from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import av
import cv2
import numpy as np
from .geometry import Pipeline
from .calibration import generate_board

log = logging.getLogger(__name__)
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

def root_dir():
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent

def ffmpeg_path():
    roots = [root_dir() / "bin" / "ffmpeg.exe"]
    if getattr(sys, "frozen", False):
        roots.append(Path(sys._MEIPASS) / "bin" / "ffmpeg.exe")
    for path in roots:
        if path.exists():
            return str(path)
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("未找到 FFmpeg，请将 ffmpeg.exe 放到程序的 bin 文件夹")
    return executable

def camera_names():
    result = subprocess.run([ffmpeg_path(), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                            capture_output=True, timeout=10, creationflags=NO_WINDOW)
    return re.findall(r'"([^"\r\n]+)" \(video\)', result.stderr.decode("utf-8", errors="replace"))

@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    sequence: int
    source_seconds: float | None
    received_monotonic: float
    processed_monotonic: float = 0.0
    revision: int = 0
    playback_seconds: float | None = None

class LatestFrame:
    """Single-slot frame mailbox. A slow consumer skips older frames."""
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.generation = 0

    def publish(self, frame):
        with self.condition:
            self.frame = frame
            self.generation += 1
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return self.frame

    def wait(self, after, timeout=1):
        with self.condition:
            self.condition.wait_for(lambda: self.generation != after, timeout)
            return self.generation, self.frame

class PipeReader:
    """Expose a non-seekable pipe to FFmpeg/PyAV on Windows."""
    def __init__(self, pipe):
        self.pipe = pipe

    def read(self, size):
        return self.pipe.read(size)

class Capture:
    def __init__(self, source, mailbox, fps=30, camera_size=(1920,1080)):
        self.source, self.mailbox, self.fps = source, mailbox, fps
        self.camera_size = camera_size
        self.backend = "DirectShow" if source.startswith("camera:") else ("Demo" if source=="demo://screen" else "PyAV / FFmpeg")
        self.stop_event = threading.Event()
        self.is_local_file = not source.startswith(("camera:", "demo:")) and Path(source).is_file()
        self.playback_lock = threading.Lock()
        self.playback_duration = 0.0
        self.playback_position = 0.0
        self.seek_request = None
        self.seeking = False
        self.playback_wake = threading.Event()
        self.process = None
        self.error = ""
        self.finished = False
        self.diagnostics = []
        self.thread = threading.Thread(target=self._run, daemon=True, name="capture")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.playback_wake.set()
        if self.process and self.process.poll() is None:
            self.process.kill()
        self.thread.join(timeout=2)

    def _run(self):
        try:
            if self.source == "demo://screen":
                self._demo()
            elif self.source.startswith("camera:"):
                self._camera()
            else:
                self._av()
        except Exception as exc:
            if not self.stop_event.is_set():
                # Do not log stream URLs or potentially credential-bearing FFmpeg diagnostics.
                self.error = "视频连接中断或无法读取，请检查来源、OBS 虚拟摄像头和网络连接。"
                log.warning("Capture failed: %s", type(exc).__name__)
        finally:
            self.finished = True
            if self.process:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=3)
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()

    def _camera(self):
        names = camera_names()
        name = self.source[7:]
        if name not in names:
            raise ValueError("Selected camera is not available")
        # Both enumerations use the Windows DirectShow video-input category.
        cap = cv2.VideoCapture(names.index(name), cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise RuntimeError("DirectShow device could not be opened")
            if self.camera_size:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            sequence = 0
            start = time.perf_counter()
            last_published = -1.0
            while not self.stop_event.is_set():
                success, image = cap.read()
                if self.stop_event.is_set():
                    break
                now = time.perf_counter()
                if not success:
                    raise RuntimeError("Camera read failed")
                # Drain capture at device rate; publish at the requested processing rate.
                if now-last_published < 0.9/self.fps:
                    continue
                sequence += 1
                last_published = now
                self.mailbox.publish(Frame(image, sequence, now-start, time.monotonic()))
        finally:
            cap.release()

    def playback_status(self):
        with self.playback_lock:
            return dict(is_file=self.is_local_file, duration_seconds=self.playback_duration,
                        position_seconds=self.playback_position,
                        seekable=self.is_local_file and self.playback_duration>0,
                        seeking=self.seeking)

    def seek(self, seconds):
        # Only the decoding thread touches the container. New requests replace
        # pending ones, so scrubbing never builds a queue of obsolete seeks.
        with self.playback_lock:
            if not self.is_local_file or self.playback_duration<=0 or not np.isfinite(seconds):
                return False
            self.seek_request = max(0.0,min(float(seconds),self.playback_duration))
            self.seeking = True
            self.playback_wake.set()
        return True

    def _av(self):
        # Decode directly with libav, avoiding raw-pixel transfer via pipes.
        options = {"rtsp_transport":"tcp"} if self.source.lower().startswith("rtsp://") else {}
        with av.open(self.source, mode="r", options=options, timeout=(5,2)) as container:
            if not container.streams.video:
                raise ValueError("Source has no video stream")
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            if self.is_local_file:
                self._file_playback(container,stream)
                return
            last_published = -1.0
            sequence = 0
            for frame in container.decode(stream):
                if self.stop_event.is_set():
                    return
                now = time.perf_counter()
                if now-last_published < 0.9/self.fps:
                    continue
                seconds = float(frame.pts*frame.time_base) if frame.pts is not None else None
                image = frame.to_ndarray(format="bgr24")
                if self.stop_event.is_set():
                    return
                sequence += 1
                last_published = now
                self.mailbox.publish(Frame(image,sequence,seconds,time.monotonic()))
            if not self.stop_event.is_set():
                raise RuntimeError("Video stream ended")

    def _file_playback(self,container,stream):
        frame_step = 1/float(stream.average_rate or self.fps)
        duration = (float(stream.duration*stream.time_base) if stream.duration is not None else
                    float(container.duration/av.time_base) if container.duration is not None else
                    float(stream.frames)*frame_step if stream.frames else 0.0)
        duration = max(0.0,duration)
        with self.playback_lock:
            self.playback_duration = duration
        origin = float(stream.start_time*stream.time_base) if stream.start_time is not None else None
        sequence = 0
        loop_offset = 0.0
        base_clock = time.perf_counter()
        last_published = -1.0
        skip_before = 0.0
        while not self.stop_event.is_set():
            with self.playback_lock:
                request = self.seek_request
                self.seek_request = None
                self.playback_wake.clear()
            if request is not None:
                # The endpoint means the final displayable frame, not immediate EOF/loop.
                target = min(request,max(0.0,duration-frame_step))
                timestamp = (origin or 0.0)+target
                container.seek(int(timestamp/stream.time_base),stream=stream,backward=True,any_frame=False)
                skip_before = target
                base_clock = time.perf_counter()-target
                last_published = -1.0
            decoded_any = False
            last_relative = 0.0
            for frame in container.decode(stream):
                if self.stop_event.is_set() or self.playback_wake.is_set():
                    break
                decoded_any = True
                seconds = float(frame.pts*frame.time_base) if frame.pts is not None else None
                if origin is None:
                    origin = seconds or 0.0
                relative = max(0.0,seconds-origin) if seconds is not None else last_relative+frame_step
                last_relative = relative
                # Seek lands on an earlier keyframe; decode/discard to the requested frame.
                if relative+1e-6 < skip_before:
                    continue
                if self.playback_wake.wait(max(0,base_clock+relative-time.perf_counter())):
                    break
                if self.stop_event.is_set():
                    break
                now = time.perf_counter()
                if now-last_published < 0.9/self.fps:
                    continue
                image = frame.to_ndarray(format="bgr24")
                with self.playback_lock:
                    if self.stop_event.is_set() or self.playback_wake.is_set():
                        break
                    sequence += 1
                    self.playback_position = relative
                    self.seeking = False
                    last_published = now
                    self.mailbox.publish(Frame(image,sequence,loop_offset+relative,time.monotonic(),
                                               playback_seconds=relative))
            if self.stop_event.is_set():
                break
            if self.playback_wake.is_set():
                continue
            if not decoded_any:
                raise RuntimeError("Video file has no decodable frames")
            # Keep the final frame visible for its frame interval before looping;
            # otherwise an end seek can be overwritten by frame zero immediately.
            if self.playback_wake.wait(max(0,base_clock+last_relative+frame_step-time.perf_counter())):
                continue
            loop_offset += duration or last_relative+frame_step
            container.seek(int((origin or 0.0)/stream.time_base),stream=stream,backward=True)
            skip_before = 0.0
            base_clock = time.perf_counter()
            last_published = -1.0

    def _stderr(self):
        try:
            for line in iter(self.process.stderr.readline, b""):
                # Retain bounded private diagnostics in memory only, never expose URLs in HTTP.
                self.diagnostics.append(line.decode("utf-8", errors="replace").rstrip())
                self.diagnostics[:] = self.diagnostics[-12:]
        except (OSError, ValueError):
            pass

    def _ffmpeg(self):
        args = [ffmpeg_path(), "-hide_banner", "-loglevel", "warning", "-nostdin"]
        if self.source.startswith("camera:"):
            args += ["-thread_queue_size", "4", "-rtbufsize", "32M", "-f", "dshow", "-i", "video=" + self.source[7:]]
        elif Path(self.source).is_file():
            args += ["-re", "-stream_loop", "-1", "-i", self.source]
        else:
            if self.source.lower().startswith("rtsp://"):
                args += ["-rtsp_transport", "tcp"]
            args += ["-rw_timeout", "5000000", "-i", self.source]
        args += ["-map", "0:v:0", "-an", "-vf", f"fps={self.fps}", "-pix_fmt", "bgr24", "-c:v", "rawvideo", "-threads", "1", "-f", "nut", "-flush_packets", "1", "pipe:1"]
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        stdin=subprocess.DEVNULL, creationflags=NO_WINDOW, bufsize=0)
        threading.Thread(target=self._stderr, daemon=True).start()
        with av.open(PipeReader(self.process.stdout), format="nut", mode="r") as container:
            for sequence, frame in enumerate(container.decode(video=0), 1):
                if self.stop_event.is_set():
                    break
                image = frame.to_ndarray(format="bgr24")
                seconds = float(frame.pts * frame.time_base) if frame.pts is not None else None
                self.mailbox.publish(Frame(image, sequence, seconds, time.monotonic()))
        if not self.stop_event.is_set():
            raise RuntimeError("视频输入已结束")

    def _demo(self):
        board = generate_board()
        target = np.array([[210,110],[1120,60],[1190,700],[120,740]], np.float32)
        source = np.array([[0,0],[1919,0],[1919,1079],[0,1079]], np.float32)
        matrix = cv2.getPerspectiveTransform(source, target)
        image = cv2.warpPerspective(board, matrix, (1280,800))
        sequence = 0
        started = time.monotonic()
        while not self.stop_event.is_set():
            sequence += 1
            current = image.copy()
            cv2.putText(current, f"DEMO  {sequence:06d}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90,220,250), 2)
            self.mailbox.publish(Frame(current, sequence, time.monotonic()-started, time.monotonic()))
            self.stop_event.wait(max(0, started+sequence/self.fps-time.monotonic()))

class Engine:
    def __init__(self):
        self.raw = LatestFrame()
        self.output = LatestFrame()
        self.lock = threading.RLock()
        self.pipeline = None
        self.render_options = Pipeline((2,2)).render_options
        self.revision = 0
        self.error = ""
        self.capture = None
        self.stop_event = threading.Event()
        self.fps = 0.0
        self.processing_ms = 0.0
        self.preview_metrics = {}
        self.thread = threading.Thread(target=self._process, daemon=True, name="geometry")
        self.thread.start()

    def connect(self, source, fps=30, pipeline=None, camera_size=(1920,1080)):
        self.disconnect()
        with self.lock:
            self.pipeline = pipeline.clone() if pipeline else None
            if pipeline:
                self.render_options = pipeline.render_options
            self.revision += 1
            self.error = ""
        self.capture = Capture(source, self.raw, fps, camera_size)
        self.capture.start()

    def disconnect(self):
        if self.capture:
            self.capture.stop()
            self.capture = None
        # Synchronize with in-flight rendering; no old-source frame may publish after disconnect.
        with self.lock:
            self.raw.publish(None)
            self.output.publish(None)
            self.revision += 1

    def update(self, pipeline):
        with self.lock:
            current = self.raw.snapshot()
            if current is not None and pipeline.input_size != (current.image.shape[1], current.image.shape[0]):
                raise ValueError("标定文件的输入分辨率与当前视频不一致")
            self.pipeline = pipeline.clone()
            self.render_options = pipeline.render_options
            self.revision += 1
            self.error = ""

    def set_render_options(self,interpolation,sharpen_amount):
        options = Pipeline((2,2),interpolation,sharpen_amount).render_options
        with self.lock:
            self.render_options = options
            if self.pipeline:
                self.pipeline.set_render_options(**options)
            self.revision += 1
            self.error = ""

    def current_pipeline(self):
        with self.lock:
            return self.pipeline.clone() if self.pipeline else None

    def _process(self):
        generation = -1
        count, started = 0, time.monotonic()
        while not self.stop_event.is_set():
            generation, frame = self.raw.wait(generation, 0.05)
            if frame is None:
                continue
            with self.lock:
                if self.raw.snapshot() is not frame:
                    continue
                if self.pipeline is None:
                    self.pipeline = Pipeline((frame.image.shape[1],frame.image.shape[0]),**self.render_options)
                previous = self.output.snapshot()
                if previous and previous.sequence == frame.sequence and previous.revision == self.revision:
                    continue
                begin = time.perf_counter()
                try:
                    image = self.pipeline.render(frame.image)
                    now = time.monotonic()
                    self.processing_ms = (time.perf_counter()-begin)*1000
                    self.output.publish(Frame(image, frame.sequence, frame.source_seconds,
                                              frame.received_monotonic, now, self.revision, frame.playback_seconds))
                    self.error = ""
                    count += 1
                    if now-started >= 1:
                        self.fps = count/(now-started)
                        count, started = 0, now
                except Exception as exc:
                    self.error = str(exc)
                    self.output.publish(None)
            if self.error:
                self.stop_event.wait(0.1)

    def seek(self, seconds):
        return self.capture.seek(seconds) if self.capture else False

    def status(self):
        with self.lock:
            frame = self.output.snapshot()
            age = time.monotonic()-frame.received_monotonic if frame else None
            receiving = bool(frame and age < 2.5 and self.capture and not self.capture.finished)
            return dict(rendering=self.render_options.copy(),playback=self.capture.playback_status() if self.capture else {}, backend=self.capture.backend if self.capture else None, preview=self.preview_metrics.copy(), connected=receiving, sequence=frame.sequence if frame else None,
                        revision=self.revision, output_size=list(self.pipeline.output_size) if self.pipeline else None,
                        pixel_format="bgr24", fps=round(self.fps,1) if receiving else 0,
                        processing_ms=round(self.processing_ms,2), frame_age_seconds=round(age,3) if age is not None else None,
                        source_seconds=frame.source_seconds if frame else None,
                        error=self.error or (self.capture.error if self.capture else ""))

    def close(self):
        self.disconnect()
        self.stop_event.set()
        self.raw.publish(None)
        self.thread.join(timeout=2)
