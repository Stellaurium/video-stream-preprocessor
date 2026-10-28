"""Minimal lossless client; no Qt or OpenCV dependency is needed by this client."""
from __future__ import annotations
from dataclasses import dataclass
import io
import json
import threading
import time
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np

@dataclass(frozen=True)
class ReceivedFrame:
    image: np.ndarray
    sequence: int
    revision: int
    source_seconds: float | None
    received_monotonic: float

class FrameClient:
    def __init__(self, base_url="http://127.0.0.1:8765", timeout=3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def read(self):
        with urlopen(self.base_url+"/frame.npy",timeout=self.timeout) as response:
            image = np.load(io.BytesIO(response.read()),allow_pickle=False)
            seconds = response.headers.get("X-Source-Seconds","None")
            return ReceivedFrame(image,int(response.headers["X-Sequence"]),int(response.headers["X-Revision"]),
                                 None if seconds=="None" else float(seconds),float(response.headers["X-Received-Monotonic"]))

    def status(self):
        with urlopen(self.base_url+"/status",timeout=self.timeout) as response:
            return json.load(response)

    def frames(self, fps=30, stop_event=None):
        if fps <= 0:
            raise ValueError("fps must be positive")
        stop_event = stop_event or threading.Event()
        previous = None
        while not stop_event.is_set():
            start = time.perf_counter()
            try:
                frame = self.read()
                key = (frame.sequence,frame.revision,frame.received_monotonic)
                if key != previous:
                    previous = key
                    yield frame
            except HTTPError as exc:
                if exc.code != 503:
                    raise
            stop_event.wait(max(0,1/fps-(time.perf_counter()-start)))
