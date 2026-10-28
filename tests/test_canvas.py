import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from screenprep.gui import VideoCanvas


def test_letterboxing_selection_coordinates_and_reverse_drag():
    app = QApplication.instance() or QApplication([])
    canvas = VideoCanvas()
    canvas.resize(1000,800)
    canvas.set_image(np.zeros((360,640,3),np.uint8))
    rect = canvas.image_rect()
    assert rect.width()/rect.height() == 640/360
    assert canvas.image_point(QPointF(5,5)) is None
    center = canvas.image_point(rect.center())
    np.testing.assert_allclose(center,[320,180])
    for point in [(0,0),(639,359),(123,215)]:
        np.testing.assert_allclose(canvas.image_point(canvas.display_point(point)),point,atol=0.0001)
    canvas.start = (400,250)
    canvas.end = (100,50)
    assert canvas.crop_rect() == (100,50,301,201)
