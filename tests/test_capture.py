from pathlib import Path
import time
import cv2
import numpy as np
import pytest
from screenprep.media import Engine,Capture


@pytest.mark.parametrize("backend",["direct","pipe"])
def test_real_video_decode_and_stop(tmp_path,monkeypatch,backend):
    if backend=="pipe":
        monkeypatch.setattr(Capture,"_av",Capture._ffmpeg)
    path = tmp_path/'test-source.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),15,(160,120))
    assert writer.isOpened()
    for index in range(45):
        frame=np.full((120,160,3),(index*5)%255,np.uint8)
        writer.write(frame)
    writer.release()
    engine=Engine()
    capture=None
    try:
        engine.connect(str(path),15)
        capture=engine.capture
        start=time.perf_counter()
        while time.perf_counter()-start<8:
            packet=engine.output.snapshot()
            if packet is not None and packet.sequence>=5:
                assert packet.image.shape==(120,160,3)
                assert packet.source_seconds>0
                break
            if capture.error:
                raise AssertionError(capture.error)
            time.sleep(.03)
        else:
            raise AssertionError('FFmpeg pipe did not deliver frames')
    finally:
        engine.close()
    assert not capture.thread.is_alive()
    if capture.process:
        assert capture.process.poll() is not None
    assert engine.output.snapshot() is None
