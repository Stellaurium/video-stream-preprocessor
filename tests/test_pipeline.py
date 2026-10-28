import json
import time
import io
import urllib.request
import urllib.error
import cv2
import numpy as np
import pytest
from screenprep.geometry import Pipeline
from screenprep.calibration import generate_board, detect_board
from screenprep.media import Engine
from screenprep.output import OutputServer


def wait_until(callback,timeout=8):
    start = time.monotonic()
    while time.monotonic()-start < timeout:
        result = callback()
        if result:
            return result
        time.sleep(0.03)
    raise AssertionError("Timed out waiting for live frame")


def test_repeated_crop_matches_pixels_without_resampling():
    rng = np.random.default_rng(12)
    image = rng.integers(0,256,(360,640,3),dtype=np.uint8)
    p = Pipeline((640,360))
    p.crop((50,30,500,280))
    p.crop((17,19,200,120))
    np.testing.assert_array_equal(p.render(image),image[49:169,67:267])
    p.undo()
    assert p.output_size == (500,280)
    p.redo()
    np.testing.assert_array_equal(p.render(image),image[49:169,67:267])
    p.undo()
    p.crop((0,0,100,80))
    assert not p.redo_stack


def test_perspective_crop_and_save_reload(tmp_path):
    p = Pipeline((800,600))
    points = np.array([[100,90],[680,40],[740,510],[60,550]],np.float32)
    p.perspective(points,(640,360))
    mapped = cv2.perspectiveTransform(points.reshape(-1,1,2),p.matrix).reshape(-1,2)
    np.testing.assert_allclose(mapped,[[0,0],[639,0],[639,359],[0,359]],atol=0.001)
    p.crop((80,60,400,200))
    p.perspective([[10,10],[380,20],[390,180],[20,190]],(480,270))
    file = tmp_path/"配置.json"
    p.save(file)
    loaded = Pipeline.load(file)
    np.testing.assert_allclose(loaded.matrix,p.matrix,atol=1e-9)
    image = np.tile(np.arange(800,dtype=np.uint8)[None,:,None],(600,1,3))
    np.testing.assert_array_equal(loaded.render(image),p.render(image))
    assert loaded.output_size == (480,270)
    assert loaded.to_dict()["lens_correction"] is None


@pytest.mark.parametrize("points",[
    [[0,0],[100,100],[100,0],[0,100]],
    [[10,10],[10,10],[100,100],[0,100]],
    [[-10,0],[200,0],[200,100],[0,100]],
    [[0,0],[1,0],[1,1],[0,1]],
])
def test_invalid_quads_do_not_change_pipeline(points):
    p = Pipeline((640,360))
    with pytest.raises(ValueError):
        p.perspective(points,(640,360))
    assert p.operations == []


def test_reject_bad_config_and_resolution(tmp_path):
    p = Pipeline((640,360))
    p.crop((10,10,300,200))
    doc = p.to_dict()
    doc["operations"][0]["input_size"] = [200,100]
    with pytest.raises(ValueError):
        Pipeline.from_dict(doc)
    with pytest.raises(ValueError):
        p.render(np.zeros((720,1280,3),np.uint8))
    with pytest.raises(ValueError):
        p.crop((0,0,400,400))
    with pytest.raises(ValueError):
        p.perspective([[0,0],[299,0],[299,199],[0,199]],(8192,8192))
    with pytest.raises(ValueError):
        p.crop((1.5,0,20,20))


def test_automatic_board_on_oblique_blurred_camera_image():
    board = generate_board()
    source = np.float32([[0,0],[1919,0],[1919,1079],[0,1079]])
    expected = np.float32([[210,110],[1120,60],[1190,700],[120,740]])
    matrix = cv2.getPerspectiveTransform(source,expected)
    camera = cv2.warpPerspective(board,matrix,(1280,800))
    camera = cv2.GaussianBlur(camera,(3,3),0.6)
    points,size,info = detect_board(camera,1280)
    assert size == (1280,720)
    assert info["inliers"] >= 12
    np.testing.assert_allclose(points,expected,atol=3)
    p = Pipeline((1280,800))
    p.perspective(points,size)
    restored = p.render(camera)
    # Independently run detector after correction; all corners remain measurable.
    check,_,_ = detect_board(restored)
    np.testing.assert_allclose(check,[[0,0],[1279,0],[1279,719],[0,719]],atol=3)


def test_missing_marker_fails_without_false_success():
    board = generate_board()
    board[0:330,0:330] = 255
    with pytest.raises(ValueError,match="缺少标记 0"):
        detect_board(board)
    with pytest.raises(ValueError):
        detect_board(np.zeros((720,1280,3),np.uint8))


def test_stream_lossless_metadata_live_changes_and_disconnect():
    engine = Engine()
    server = OutputServer(engine,0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        engine.connect("demo://screen",15)
        wait_until(lambda:engine.status()["connected"])
        with urllib.request.urlopen(base+"/frame.npy",timeout=3) as response:
            image = np.load(io.BytesIO(response.read()),allow_pickle=False)
            assert image.shape == (800,1280,3)
            assert response.headers["X-Pixel-Format"] == "bgr24"
            sequence = int(response.headers["X-Sequence"])
        with urllib.request.urlopen(base+"/stream.mjpg",timeout=4) as response:
            assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
            assert response.readline().strip() == b"--frame"
            headers = {}
            while (line := response.readline().strip()):
                k,v = line.decode().split(":",1)
                headers[k] = v.strip()
            data = response.read(int(headers["Content-Length"]))
            decoded = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            assert decoded.shape == (800,1280,3)
        p = engine.current_pipeline()
        points,size,details = detect_board(engine.output.snapshot().image)
        p.perspective(points,size,"自动",details)
        p.crop((100,100,500,300))
        engine.update(p)
        wait_until(lambda:engine.output.snapshot() and engine.output.snapshot().revision==engine.revision)
        with urllib.request.urlopen(base+"/frame.png",timeout=3) as response:
            decoded = cv2.imdecode(np.frombuffer(response.read(),np.uint8),cv2.IMREAD_COLOR)
            assert decoded.shape == (300,500,3)
            assert int(response.headers["X-Sequence"])>=sequence
        with urllib.request.urlopen(base+"/status",timeout=3) as response:
            status = json.load(response)
            assert status["connected"] and status["output_size"] == [500,300]
        engine.disconnect()
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base+"/frame.npy",timeout=3)
        assert error.value.code == 503
        assert engine.output.snapshot() is None
    finally:
        server.stop()
        engine.close()
