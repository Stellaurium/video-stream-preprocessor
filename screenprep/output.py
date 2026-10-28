from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import io
import json
import threading
import time
import cv2
import numpy as np

class OutputServer:
    """Loopback-only latest-frame service; bounded clients and no growing queues."""
    def __init__(self, engine, port=8765):
        self.engine = engine
        self.port = port
        self.http = None
        self.stop_event = threading.Event()
        self.clients = threading.BoundedSemaphore(8)
        self.cache_lock = threading.Lock()
        self.cache_key = None
        self.cache_jpeg = None

    def jpeg(self, frame):
        key = (frame.sequence, frame.revision, frame.received_monotonic)
        with self.cache_lock:
            if self.cache_key != key:
                ok, data = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                self.cache_jpeg, self.cache_key = data.tobytes(), key
            return self.cache_jpeg

    def start(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def reply(self, data, mime, status=200, frame=None):
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if frame:
                    for k,v in self.metadata(frame).items():
                        self.send_header(k,v)
                self.end_headers()
                self.wfile.write(data)

            def metadata(self, frame):
                return {"X-Sequence":str(frame.sequence), "X-Revision":str(frame.revision),
                        "X-Source-Seconds":str(frame.source_seconds), "X-Received-Monotonic":str(frame.received_monotonic),
                        "X-Width":str(frame.image.shape[1]), "X-Height":str(frame.image.shape[0]), "X-Pixel-Format":"bgr24"}

            def do_GET(self):
                if not owner.clients.acquire(blocking=False):
                    self.reply(b"Too many clients", "text/plain", 503)
                    return
                try:
                    self.route(urlsplit(self.path).path)
                except (ConnectionError, OSError, TimeoutError):
                    pass
                finally:
                    owner.clients.release()

            def route(self, path):
                if path == "/status":
                    self.reply(json.dumps(owner.engine.status()).encode(), "application/json")
                    return
                if path == "/calibration.json":
                    pipeline = owner.engine.current_pipeline()
                    self.reply(json.dumps(pipeline.to_dict() if pipeline else {}).encode(), "application/json")
                    return
                if path == "/":
                    self.reply(b'<!doctype html><meta charset="utf-8"><title>ScreenPrep Output</title><style>body{background:#101522;color:#dce5ef;font-family:system-ui;margin:30px}img{max-width:100%;max-height:85vh}a{color:#4de0c1}</style><h2>ScreenPrep / Live output</h2><p><a href="/status">Status</a> &middot; <a href="/calibration.json">Calibration</a> &middot; <a href="/frame.png">Lossless PNG</a></p><img src="/stream.mjpg">', "text/html; charset=utf-8")
                    return
                if path == "/stream.mjpg":
                    self.stream()
                    return
                if path not in ("/frame.jpg", "/frame.png", "/frame.npy"):
                    self.reply(b"Not found", "text/plain", 404)
                    return
                frame = owner.engine.output.snapshot()
                if frame is None or not owner.engine.status()["connected"]:
                    self.reply(b"No live frame", "text/plain", 503)
                    return
                if path == "/frame.jpg":
                    self.reply(owner.jpeg(frame), "image/jpeg", frame=frame)
                elif path == "/frame.png":
                    ok, encoded = cv2.imencode(".png", frame.image)
                    if not ok:
                        self.reply(b"Encoding failed", "text/plain", 500)
                        return
                    self.reply(encoded.tobytes(), "image/png", frame=frame)
                else:
                    buffer = io.BytesIO()
                    np.save(buffer, frame.image, allow_pickle=False)
                    self.reply(buffer.getvalue(), "application/octet-stream", frame=frame)

            def stream(self):
                if not owner.engine.status()["connected"]:
                    self.reply(b"No live stream", "text/plain", 503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                generation = -1
                last_key = None
                while not owner.stop_event.is_set():
                    generation, frame = owner.engine.output.wait(generation, 1)
                    if frame is None or not owner.engine.status()["connected"]:
                        break
                    key = (frame.sequence, frame.revision, frame.received_monotonic)
                    if key == last_key:
                        continue
                    data = owner.jpeg(frame)
                    headers = "\r\n".join(f"{k}: {v}" for k,v in self.metadata(frame).items())
                    self.wfile.write((f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n{headers}\r\n\r\n").encode()+data+b"\r\n")
                    self.wfile.flush()
                    last_key = key
                self.close_connection = True
        self.http = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.http.daemon_threads = True
        self.port = self.http.server_address[1]
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True, name="http-output")
        self.thread.start()
        return self.port

    def stop(self):
        self.stop_event.set()
        self.engine.output.publish(self.engine.output.snapshot())
        if self.http:
            self.http.shutdown()
            self.http.server_close()
            self.thread.join(timeout=2)
            self.http = None
