from pathlib import Path
import time
import av
import cv2
import numpy as np
from screenprep.media import Engine, Capture, LatestFrame


def make_seek_video(path):
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=20)
        stream.width=160
        stream.height=120
        stream.pix_fmt='yuv420p'
        stream.codec_context.gop_size=50
        stream.codec_context.max_b_frames=2
        for index in range(180):
            pixels=np.full((120,160,3),index+20,np.uint8)
            frame=av.VideoFrame.from_ndarray(pixels,format='bgr24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def wait_for(predicate,timeout=3):
    started=time.perf_counter()
    while time.perf_counter()-started<timeout:
        value=predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('Timed out waiting for seek result')


def test_file_seek_forward_backward_end_and_preserve_crop(tmp_path):
    path=tmp_path/'seek.mp4'
    make_seek_video(path)
    engine=Engine()
    try:
        engine.connect(str(path),30)
        wait_for(lambda:engine.status()['connected'])
        playback=engine.status()['playback']
        assert playback['is_file'] and playback['seekable']
        assert abs(playback['duration_seconds']-9)<.1
        pipeline=engine.current_pipeline()
        pipeline.crop((20,20,80,60))
        engine.update(pipeline)
        revision=engine.revision
        sequence=engine.output.snapshot().sequence
        for target in [6.2,.4,8.95,2.25,0.0]:
            started=time.perf_counter()
            assert engine.seek(target)
            def reached():
                packet=engine.output.snapshot()
                return packet if (packet and packet.sequence>sequence and packet.playback_seconds is not None
                                  and target-.06 <= packet.playback_seconds <= target+.2) else None
            packet=wait_for(reached)
            assert time.perf_counter()-started<1.5
            assert packet.image.shape==(60,80,3)
            expected=round(packet.playback_seconds*20)+20
            assert abs(float(packet.image[10,10].mean())-expected)<5
            assert engine.revision==revision
            assert not engine.status()['playback']['seeking']
            sequence=packet.sequence
        assert engine.seek(7)
        assert engine.seek(1.5)
        packet=wait_for(lambda:(p if (p:=engine.output.snapshot()) and p.sequence>sequence and 1.45<=p.playback_seconds<1.8 else None))
        assert packet.playback_seconds is not None
    finally:
        engine.close()


def test_camera_and_live_inputs_cannot_be_seeked():
    for source in ['camera:OBS Virtual Camera','rtsp://127.0.0.1/live','demo://screen']:
        capture=Capture(source,LatestFrame())
        assert not capture.playback_status()['is_file']
        assert not capture.seek(10)
