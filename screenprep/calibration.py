from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np

BOARD_SIZE = (1920, 1080)
MARKER_SIZE = 210
MARGIN = 72

def marker_layout():
    w, h = BOARD_SIZE
    s, m = MARKER_SIZE, MARGIN
    xy = [(m, m), (w-m-s, m), (w-m-s, h-m-s), (m, h-m-s)]
    return {i: np.array([[x,y], [x+s-1,y], [x+s-1,y+s-1], [x,y+s-1]], np.float32)
            for i, (x,y) in enumerate(xy)}

def dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def generate_board():
    w, h = BOARD_SIZE
    board = np.full((h, w, 3), 255, np.uint8)
    for x in range(360, w-320, 120):
        cv2.line(board, (x, 330), (x, h-330), (205, 215, 222), 2)
    for y in range(330, h-300, 105):
        cv2.line(board, (360, y), (w-360, y), (205, 215, 222), 2)
    cv2.circle(board, (w//2,h//2), 145, (70, 90, 110), 4, cv2.LINE_AA)
    for marker_id, corners in marker_layout().items():
        x, y = corners[0].astype(int)
        marker = cv2.aruco.generateImageMarker(dictionary(), marker_id, MARKER_SIZE)
        board[y:y+MARKER_SIZE, x:x+MARKER_SIZE] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        cv2.putText(board, str(marker_id), (x+85,y+MARKER_SIZE+39), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40,40,40), 2, cv2.LINE_AA)
    cv2.putText(board, "SCREENPREP / CALIBRATION", (520, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (30,50,70), 3, cv2.LINE_AA)
    cv2.putText(board, "1920 x 1080  |  16:9  |  KEEP ASPECT", (570, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (90,100,110), 2, cv2.LINE_AA)
    cv2.putText(board, "Keep all four markers visible. Do not crop or stretch.", (455, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60,70,80), 2, cv2.LINE_AA)
    return board

def export_board(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ScreenPrep-Calibration-1920x1080.png"
    ok, encoded = cv2.imencode(".png", generate_board())
    if not ok:
        raise RuntimeError("无法生成标定图")
    encoded.tofile(str(path))
    (directory / "calibration-board.json").write_text(json.dumps({
        "version":1, "size":BOARD_SIZE, "dictionary":"DICT_4X4_50",
        "marker_corners":{str(k):v.tolist() for k,v in marker_layout().items()}}, indent=2), encoding="utf-8")
    return path

def detect_board(frame, output_width=1920):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary(), params)
    corners, ids, _ = detector.detectMarkers(gray)
    found = {} if ids is None else {int(i): c.reshape(4,2) for i,c in zip(ids.ravel(), corners)}
    missing = sorted(set(range(4)) - set(found))
    if missing:
        raise ValueError("未完整识别标定图，缺少标记 " + ", ".join(map(str,missing)) + "。请完整显示标定图（可窗口显示，保持原比例），让四个标记完整露出并清晰对焦。")
    source = np.concatenate([found[i] for i in range(4)])
    target = np.concatenate([marker_layout()[i] for i in range(4)])
    # Fit in source-camera pixels so the RANSAC threshold does not depend on output size.
    board_to_frame, inliers = cv2.findHomography(target, source, cv2.RANSAC, 2.5)
    if board_to_frame is None or inliers is None or int(inliers.sum()) < 12:
        raise ValueError("标定点一致性不足，请放大屏幕在画面中的占比或改善对焦后重试")
    predicted = cv2.perspectiveTransform(target.reshape(-1,1,2), board_to_frame).reshape(-1,2)
    error = np.linalg.norm(predicted-source, axis=1)
    rms = float(np.sqrt(np.mean(error ** 2)))
    if rms > 2.5 or float(error.max()) > 6:
        raise ValueError(f"标定误差较大（{rms:.2f} 像素），请检查反光、对焦或镜头弯曲畸变")
    w, h = BOARD_SIZE
    full = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], np.float32)
    screen = cv2.perspectiveTransform(full.reshape(-1,1,2), board_to_frame).reshape(-1,2)
    output_width = int(output_width)
    output_height = max(2, round(output_width * 9 / 16))
    return screen, (output_width, output_height), {"board":"screenprep-16x9-v1", "rms_source_pixels":rms, "inliers":int(inliers.sum()), "points":16}
