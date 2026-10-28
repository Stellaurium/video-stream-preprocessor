from __future__ import annotations
import json
import math
import os
from collections import deque
from pathlib import Path
import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QDesktopServices, QShortcut, QKeySequence, QFontDatabase
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QGroupBox, QListWidget, QFileDialog, QMessageBox, QFrame, QScrollArea, QSlider, QStyle, QStyleOptionSlider)
from .media import Engine, camera_names, root_dir
from .geometry import Pipeline
from .calibration import detect_board, export_board, generate_board
from .output import OutputServer

TEAL = QColor("#45dfbf")

class SeekSlider(QSlider):
    """A small timeline: click anywhere to jump, or drag the handle."""
    def __init__(self,parent=None):
        super().__init__(Qt.Orientation.Horizontal,parent)
        self.setRange(0,10000)
        self.setSingleStep(10)
        self.setPageStep(500)

    def _move_to(self,position):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(QStyle.ComplexControl.CC_Slider,option,QStyle.SubControl.SC_SliderGroove,self)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider,option,QStyle.SubControl.SC_SliderHandle,self)
        span = max(1,groove.width()-handle.width())
        offset = round(position.x()-groove.left()-handle.width()/2)
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(),self.maximum(),max(0,min(span,offset)),span,option.upsideDown))

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:
            self.setFocus()
            self.setSliderDown(True)
            self._move_to(event.position())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.isSliderDown():
            self._move_to(event.position())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton and self.isSliderDown():
            self._move_to(event.position())
            self.setSliderDown(False)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class VideoCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self._display_cache = None
        self._display_cache_key = None
        self.presentation_mode = False
        self.serial = 0
        self.painted_serial = -1
        self.paint_times = deque(maxlen=180)
        self.last_paint_ms = 0.0
        self.mode = None
        self.points = []
        self.start = None
        self.end = None
        self.dragging = False
        self.on_change = lambda: None
        self.setMinimumSize(420, 280)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image):
        self.image = image
        self._display_cache = None
        self._display_cache_key = None
        self.serial += 1
        self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.points = []
        self.start = self.end = None
        self.dragging = False
        self.setCursor(Qt.CursorShape.CrossCursor if mode else Qt.CursorShape.ArrowCursor)
        self.update()
        self.on_change()

    def image_rect(self):
        if self.image is None:
            return QRectF()
        h, w = self.image.shape[:2]
        margin = 0 if self.presentation_mode else 18
        area = self.rect().adjusted(margin,margin,-margin,-margin)
        scale = min(area.width()/w, area.height()/h)
        return QRectF((self.width()-w*scale)/2, (self.height()-h*scale)/2, w*scale, h*scale)

    def image_point(self, position, clamp=False):
        rect = self.image_rect()
        if rect.isEmpty() or (not clamp and not rect.contains(position)):
            return None
        h, w = self.image.shape[:2]
        return (max(0,min(w-1,(position.x()-rect.x())/rect.width()*w)),
                max(0,min(h-1,(position.y()-rect.y())/rect.height()*h)))

    def display_point(self, point):
        rect = self.image_rect()
        h, w = self.image.shape[:2]
        return QPointF(rect.x()+point[0]/w*rect.width(), rect.y()+point[1]/h*rect.height())

    def crop_rect(self):
        if self.start is None or self.end is None or self.image is None:
            return None
        h, w = self.image.shape[:2]
        x0 = max(0, math.floor(min(self.start[0],self.end[0])))
        y0 = max(0, math.floor(min(self.start[1],self.end[1])))
        x1 = min(w, math.ceil(max(self.start[0],self.end[0]))+1)
        y1 = min(h, math.ceil(max(self.start[1],self.end[1]))+1)
        return (x0,y0,x1-x0,y1-y0) if x1-x0 >= 2 and y1-y0 >= 2 else None

    def mousePressEvent(self, event):
        if not self.mode:
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.points:
                self.points.pop()
            self.start = self.end = None
        else:
            point = self.image_point(event.position())
            if point is None:
                return
            if self.mode == "perspective":
                if len(self.points) < 4:
                    self.points.append(point)
            elif self.mode == "crop":
                self.start = self.end = point
                self.dragging = True
        self.on_change()
        self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "crop" and self.dragging:
            self.end = self.image_point(event.position(), clamp=True)
            self.on_change()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == "crop" and self.dragging:
            self.end = self.image_point(event.position(), clamp=True)
            self.dragging = False
            self.on_change()
            self.update()

    def paintEvent(self, event):
        paint_started = time.perf_counter()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b1018"))
        if self.image is None:
            painter.setPen(QColor("#8391a6"))
            painter.setFont(QFont("Microsoft YaHei UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待视频输入\n\n选择 OBS 虚拟摄像头、视频文件或串流地址后连接")
            return
        h, w = self.image.shape[:2]
        rect = self.image_rect()
        # Prepare exactly the physical preview size. Area reduction avoids losing
        # thin strokes to Qt's default nearest-neighbour window scaling.
        dpr = self.devicePixelRatioF()
        dw,dh = max(1,round(rect.width()*dpr)),max(1,round(rect.height()*dpr))
        key = (self.serial,dw,dh)
        if key!=self._display_cache_key:
            self._display_cache = (np.ascontiguousarray(self.image) if (dw,dh)==(w,h) else
                                   cv2.resize(self.image,(dw,dh),interpolation=cv2.INTER_AREA if dw<w else cv2.INTER_CUBIC))
            self._display_cache_key = key
        image = self._display_cache
        qimage = QImage(image.data, dw, dh, image.strides[0], QImage.Format.Format_BGR888)
        painter.drawImage(rect, qimage)
        if not self.presentation_mode:
            painter.setPen(QPen(QColor("#273348"),1))
            painter.drawRect(rect)
        if self.mode == "crop" and self.start and self.end:
            a, b = self.display_point(self.start), self.display_point(self.end)
            selected = QRectF(a,b).normalized()
            shade = QColor(0,0,0,145)
            painter.fillRect(QRectF(rect.left(),rect.top(),rect.width(),max(0,selected.top()-rect.top())),shade)
            painter.fillRect(QRectF(rect.left(),selected.bottom(),rect.width(),max(0,rect.bottom()-selected.bottom())),shade)
            painter.fillRect(QRectF(rect.left(),selected.top(),max(0,selected.left()-rect.left()),selected.height()),shade)
            painter.fillRect(QRectF(selected.right(),selected.top(),max(0,rect.right()-selected.right()),selected.height()),shade)
            painter.setPen(QPen(TEAL,2))
            painter.drawRect(selected)
            roi = self.crop_rect()
            if roi:
                painter.setPen(TEAL)
                painter.drawText(selected.topLeft()+QPointF(8,22),f"{roi[2]} × {roi[3]}")
        if self.mode == "perspective":
            points = [self.display_point(p) for p in self.points]
            painter.setPen(QPen(TEAL,2))
            for i in range(len(points)-1):
                painter.drawLine(points[i],points[i+1])
            if len(points) == 4:
                painter.drawLine(points[-1],points[0])
            names = ["1 左上","2 右上","3 右下","4 左下"]
            for i,p in enumerate(points):
                painter.setBrush(TEAL)
                painter.setPen(QPen(QColor("#051f1a"),2))
                painter.drawEllipse(p,6,6)
                bubble = QRectF(p.x()+10,p.y()-31,74,24)
                painter.fillRect(bubble,QColor("#082a24"))
                painter.setPen(TEAL)
                painter.drawText(bubble,Qt.AlignmentFlag.AlignCenter,names[i])
        if self.mode:
            painter.setPen(QColor("#ffd68b"))
            painter.drawText(QRectF(28,20,300,28),"选区预览已冻结 · 视频接收继续")
        self.last_paint_ms = (time.perf_counter()-paint_started)*1000
        if self.serial != self.painted_serial:
            self.paint_times.append(time.perf_counter())
            self.painted_serial = self.serial

    def paint_fps(self):
        now = time.perf_counter()
        while self.paint_times and now-self.paint_times[0]>2:
            self.paint_times.popleft()
        return (len(self.paint_times)-1)/(self.paint_times[-1]-self.paint_times[0]) if len(self.paint_times)>1 else 0.0

STYLE = """
QMainWindow, QWidget { background: #111824; color: #dce5f0; font-family: 'Microsoft YaHei UI'; font-size: 12px; }
QWidget#sidebar { background: #17202e; border-right: 1px solid #283348; }
QLabel#brand { font-size: 24px; font-weight: 700; color: #f3f7fc; }
QLabel#muted { color: #8c9bb0; }
QLabel#headline { font-size: 19px; font-weight: 600; }
QLabel#badge { color: #45dfbf; background: #15362f; border-radius: 6px; padding: 5px 9px; }
QGroupBox { border: 1px solid #2d3a4f; border-radius: 8px; margin-top: 12px; padding: 13px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #afbed0; }
QPushButton { background: #243248; border: 1px solid #34455d; border-radius: 6px; padding: 8px 10px; min-height: 18px; }
QPushButton:hover { background: #30445e; border-color: #517090; }
QPushButton:disabled { background: #1b2534; color: #58687d; border-color: #273348; }
QPushButton#primary { background: #35caa9; border-color: #35caa9; color: #082b23; font-weight: 700; }
QPushButton#primary:hover { background: #63e4c8; }
QPushButton#primary:disabled { background: #21483f; color: #6b9389; border-color: #21483f; }
QPushButton#flat { background: transparent; border-color: #35465e; }
QComboBox, QSpinBox { background: #101823; border: 1px solid #36465d; border-radius: 5px; padding: 6px; min-height: 18px; selection-background-color: #286b60; }
QComboBox QAbstractItemView { background: #172535; selection-background-color: #286b60; }
QListWidget { background: #101823; border: 1px solid #2b3b50; border-radius: 5px; padding: 5px; }
QListWidget::item { padding: 7px 3px; }
QScrollArea { border: 0; }
QStatusBar { background: #0c131e; color: #9badc3; border-top: 1px solid #283348; }
QSlider::groove:horizontal { background: #2a394e; height: 4px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #45dfbf; border-radius: 2px; }
QSlider::handle:horizontal { background: #b0f6e6; border: 1px solid #45dfbf; width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; }
QToolTip { background: #23364a; color: #f3f7fc; border: 1px solid #4c657d; }
"""

class MainWindow(QMainWindow):
    def __init__(self, demo=False, port=8765):
        super().__init__()
        fonts = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"
        for name in ("msyh.ttc", "msyhbd.ttc"):
            if (fonts/name).exists():
                QFontDatabase.addApplicationFont(str(fonts/name))
        self.setWindowTitle("ScreenPrep · 屏幕校正工作台")
        self.resize(1400,880)
        self.setMinimumSize(1050,720)
        self.engine = Engine()
        self.server = None
        self.data_dir = root_dir()/"data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = root_dir()/"assets"
        self.pending_pipeline = None
        self.last_key = None
        self.frozen = None
        self.last_notice = ""
        self.last_health = ""
        self.shortcuts = []
        self.presentation_mode = False
        self._presentation_fullscreen = False
        self._build()
        self.setStyleSheet(STYLE)
        self.canvas.on_change = self.selection_changed
        self._shortcut("Ctrl+Z", lambda:self.history("undo"))
        self._shortcut("Ctrl+Shift+Z", lambda:self.history("redo"))
        self._shortcut("Ctrl+S", self.save)
        self._shortcut("Escape", self.escape_action)
        self._shortcut("F11", self.toggle_fullscreen)
        self._shortcut("F10", self.toggle_windowed_view)
        self._shortcut("Return", self.apply_selection)
        try:
            names = camera_names()
        except Exception:
            names = ["OBS Virtual Camera"]
        for name in sorted(names,key=lambda n: "OBS" not in n):
            self.source.addItem("camera:"+name)
        self.source.addItem("demo://screen")
        settings = {}
        try:
            settings = json.loads((self.data_dir/"session.json").read_text(encoding="utf-8"))
        except (OSError,ValueError):
            pass
        self.source.setCurrentText("demo://screen" if demo else settings.get("source", "camera:OBS Virtual Camera"))
        if not demo and (self.data_dir/"autosave.json").exists():
            try:
                self.pending_pipeline = Pipeline.load(self.data_dir/"autosave.json")
            except Exception:
                self.notify("上次自动保存的配置无法读取，请手动载入配置",error=True)
        export_board(self.assets_dir)
        self.port.setValue(port)
        self.toggle_server()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.pump)
        self.timer.start(16)
        QTimer.singleShot(250,self.connect_source)

    def _shortcut(self,key,callback):
        shortcut = QShortcut(QKeySequence(key),self)
        shortcut.activated.connect(callback)
        self.shortcuts.append(shortcut)

    def button(self,text,callback,primary=False):
        button = QPushButton(text)
        button.setObjectName("primary" if primary else "flat")
        button.clicked.connect(callback)
        return button

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        self.root_layout = root
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(292)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(18,20,18,16)
        title = QLabel("ScreenPrep")
        title.setObjectName("brand")
        layout.addWidget(title)
        subtitle = QLabel("屏幕校正工作台  /  0.1.1")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        input_box = QGroupBox("01  视频来源")
        v = QVBoxLayout(input_box)
        self.source = QComboBox()
        self.source.setEditable(True)
        self.source.setToolTip("camera:OBS Virtual Camera，或粘贴 rtsp/http/rtmp/srt 地址，或选择本地视频")
        v.addWidget(self.source)
        row = QHBoxLayout()
        row.addWidget(self.button("选择视频",self.open_video))
        row.addWidget(self.button("刷新设备",self.refresh_devices))
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("处理帧率"))
        self.capture_fps = QComboBox()
        self.capture_fps.addItems(["15","30","60"])
        self.capture_fps.setCurrentText("15")
        row.addWidget(self.capture_fps)
        v.addLayout(row)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("摄像头输入尺寸"))
        self.camera_size = QComboBox()
        self.camera_size.addItems(["1920×1080", "1280×720", "640×480", "设备默认"])
        self.camera_size.setToolTip("仅摄像头输入有效；OBS 建议与其基础画布分辨率一致，实际尺寸以底部状态为准")
        size_row.addWidget(self.camera_size)
        v.addLayout(size_row)
        self.connect_button = self.button("连接 / 重新连接",self.connect_source,True)
        v.addWidget(self.connect_button)
        layout.addWidget(input_box)
        tools = QGroupBox("02  校正与裁剪")
        v = QVBoxLayout(tools)
        self.auto_button = self.button("识别标准图 · 自动校正",self.auto_correct,True)
        v.addWidget(self.auto_button)
        v.addWidget(self.button("导出标准标定图",self.export_target))
        framing_row = QHBoxLayout()
        framing_row.addWidget(QLabel("自动校正范围"))
        self.auto_framing = QComboBox()
        self.auto_framing.addItem("保留整幅画面","canvas")
        self.auto_framing.addItem("向外留边","margin")
        self.auto_framing.addItem("仅标定区域","screen")
        self.auto_framing.setToolTip("标准图可在小窗口中完整显示；保留整幅画面后，可再手动裁剪")
        framing_row.addWidget(self.auto_framing)
        v.addLayout(framing_row)
        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("每侧留边"))
        self.auto_margin = QSpinBox()
        self.auto_margin.setRange(0,200)
        self.auto_margin.setSuffix(" %")
        self.auto_margin.setValue(20)
        self.auto_margin.setToolTip("左右各按标定区域宽度留边，上下各按标定区域高度留边。例如 20% 表示总宽高均为原来的 1.4 倍")
        self.auto_margin.setEnabled(False)
        self.auto_framing.currentIndexChanged.connect(lambda:self.auto_margin.setEnabled(self.auto_framing.currentData()=="margin"))
        margin_row.addWidget(self.auto_margin)
        v.addLayout(margin_row)
        row = QHBoxLayout()
        self.point_button = self.button("四点校正",lambda:self.begin_selection("perspective"))
        self.crop_button = self.button("框选裁剪",lambda:self.begin_selection("crop"))
        row.addWidget(self.point_button)
        row.addWidget(self.crop_button)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("手动宽高比"))
        self.ratio = QComboBox()
        self.ratio.addItems(["16:9","16:10","4:3","1:1","9:16"])
        row.addWidget(self.ratio)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("校正输出宽度"))
        self.output_width = QSpinBox()
        self.output_width.setRange(160,3840)
        self.output_width.setSingleStep(160)
        self.output_width.setValue(1920)
        self.output_width.setToolTip("保留整幅画面时为最终画布宽度；向外留边或仅标定区域时为标定图区域的校正宽度")
        row.addWidget(self.output_width)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("图像采样"))
        self.sampling = QComboBox()
        self.sampling.addItem("高质量 · Lanczos","lanczos4")
        self.sampling.addItem("均衡 · 双三次","cubic")
        self.sampling.addItem("快速 · 双线性（原版）","linear")
        self.sampling.setToolTip("直接从原始帧进行一次重采样。Lanczos 更适合精细文字，但不会恢复原始画面缺失的细节")
        row.addWidget(self.sampling)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("文字轻锐化"))
        self.sharpen = QSpinBox()
        self.sharpen.setRange(0,60)
        self.sharpen.setSuffix(" %")
        self.sharpen.setValue(20)
        self.sharpen.setToolTip("默认 20%，增强边缘对比；0% 关闭。提高后可能放大噪点、光晕。原图和纯裁剪不锐化")
        row.addWidget(self.sharpen)
        v.addLayout(row)
        self.sampling.currentIndexChanged.connect(self.change_rendering)
        self.sharpen.valueChanged.connect(self.change_rendering)
        self.apply_button = self.button("应用当前选区  ↵",self.apply_selection,True)
        self.cancel_button = self.button("取消选择  Esc",self.cancel_selection)
        v.addWidget(self.apply_button)
        v.addWidget(self.cancel_button)
        layout.addWidget(tools)
        history_box = QGroupBox("03  操作记录")
        v = QVBoxLayout(history_box)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(110)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        v.addWidget(self.history_list)
        row = QHBoxLayout()
        self.undo_button = self.button("撤销",lambda:self.history("undo"))
        self.redo_button = self.button("重做",lambda:self.history("redo"))
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        row.addWidget(self.button("重置",self.reset))
        v.addLayout(row)
        layout.addWidget(history_box)
        row = QHBoxLayout()
        row.addWidget(self.button("保存配置",self.save,True))
        row.addWidget(self.button("载入配置",self.load))
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.button("另存为…",self.save_as))
        row.addWidget(self.button("配置目录",self.open_data))
        layout.addLayout(row)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(312)
        scroll.setWidget(side)
        root.addWidget(scroll)
        self.sidebar = scroll
        content = QVBoxLayout()
        self.content_layout = content
        content.setContentsMargins(24,22,24,16)
        content.setSpacing(14)
        row = QHBoxLayout()
        heading = QLabel("实时画面")
        heading.setObjectName("headline")
        row.addWidget(heading)
        row.addStretch()
        self.badge = QLabel("等待连接")
        self.badge.setObjectName("badge")
        row.addWidget(self.badge)
        self.windowed_view_button = self.button("纯画面 F10",self.toggle_windowed_view)
        self.windowed_view_button.setToolTip("隐藏控制区，画面占满当前窗口，保留窗口边框和任务栏；F10 或 Esc 恢复")
        row.addWidget(self.windowed_view_button)
        self.fullscreen_button = self.button("全屏 F11",self.toggle_fullscreen)
        self.fullscreen_button.setToolTip("只显示处理后的画面；F11 或 Esc 退出全屏")
        row.addWidget(self.fullscreen_button)
        row.addWidget(self.button("载入配置",self.load))
        row.addWidget(self.button("保存配置",self.save,True))
        self.header = QWidget()
        row.setContentsMargins(0,0,0,0)
        self.header.setLayout(row)
        content.addWidget(self.header)
        self.hint = QLabel("连接视频后，可先裁掉黑边，再进行自动校正或四点校正。")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        content.addWidget(self.hint)
        self.canvas = VideoCanvas()
        content.addWidget(self.canvas,1)
        self.timeline = QWidget()
        timeline_row = QHBoxLayout(self.timeline)
        timeline_row.setContentsMargins(0,0,0,0)
        timeline_row.setSpacing(12)
        self.seek_slider = SeekSlider()
        self.seek_slider.setAccessibleName("视频播放进度")
        self.seek_slider.setToolTip("点击或拖动进度条跳转到指定时间")
        self.seek_slider.valueChanged.connect(self.timeline_changed)
        self.seek_slider.sliderReleased.connect(self.seek_video)
        timeline_row.addWidget(self.seek_slider,1)
        self.playback_time = QLabel("00:00 / 00:00")
        self.playback_time.setObjectName("muted")
        self.playback_time.setMinimumWidth(110)
        self.playback_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        timeline_row.addWidget(self.playback_time)
        self.timeline.setFixedHeight(24)
        self.timeline.hide()
        content.addWidget(self.timeline)
        self.metrics = QLabel("输入 —    输出 —    处理 —")
        self.metrics.setObjectName("muted")
        content.addWidget(self.metrics)
        output_box = QGroupBox("实时输出 · 仅本机访问")
        v = QVBoxLayout(output_box)
        row = QHBoxLayout()
        self.output_url = QLabel("输出未启动")
        self.output_url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.output_url,1)
        self.port = QSpinBox()
        self.port.setRange(1024,65535)
        self.port.setValue(8765)
        self.port.setFixedWidth(90)
        row.addWidget(self.port)
        self.server_button = self.button("启动输出",self.toggle_server)
        row.addWidget(self.server_button)
        row.addWidget(self.button("复制地址",self.copy_url))
        v.addLayout(row)
        text = QLabel("视频流：MJPEG  ·  无损单帧：/frame.png 或 /frame.npy  ·  状态：/status")
        text.setObjectName("muted")
        v.addWidget(text)
        content.addWidget(output_box)
        self.output_box = output_box
        self.notice = QLabel("配置自动保存在程序目录的 data 文件夹；保存配置可导出独立 JSON。")
        self.notice.setWordWrap(True)
        self.notice.setMinimumHeight(38)
        content.addWidget(self.notice)
        root.addLayout(content,1)
        self.statusBar().showMessage("准备连接视频")
        self.selection_changed()

    def escape_action(self):
        if self.presentation_mode:
            self._exit_presentation()
        else:
            self.cancel_selection()

    def _remember_viewer_window(self):
        self._preview_window_geometry = self.normalGeometry() if self.isMaximized() else self.geometry()
        self._preview_was_maximized = self.isMaximized()

    def _restore_viewer_window(self):
        self.showNormal()
        self.setGeometry(self._preview_window_geometry)
        if self._preview_was_maximized:
            self.showMaximized()

    def toggle_fullscreen(self):
        self._toggle_presentation(fullscreen=True)

    def toggle_windowed_view(self):
        self._toggle_presentation(fullscreen=False)

    def _toggle_presentation(self,fullscreen):
        if self.presentation_mode:
            if self._presentation_fullscreen==fullscreen:
                self._exit_presentation()
                return
            # Switch modes without flashing or restoring the controls.
            if fullscreen:
                self._remember_viewer_window()
                self.showFullScreen()
            else:
                self._restore_viewer_window()
            self._presentation_fullscreen = fullscreen
        else:
            if fullscreen:
                self._remember_viewer_window()
            self._preview_minimum_size = self.minimumSize()
            margins = self.content_layout.contentsMargins()
            self._preview_margins = (margins.left(),margins.top(),margins.right(),margins.bottom())
            self._preview_spacing = self.content_layout.spacing()
            chrome = [self.sidebar,self.header,self.hint,self.metrics,self.output_box,self.notice,self.statusBar()]
            self._preview_chrome = [(widget,not widget.isHidden()) for widget in chrome]
            self.cancel_selection()
            self.presentation_mode = True
            self._presentation_fullscreen = fullscreen
            self.canvas.presentation_mode = True
            for widget,_ in self._preview_chrome:
                widget.hide()
            self.timeline.hide()
            self.content_layout.setContentsMargins(0,0,0,0)
            self.content_layout.setSpacing(0)
            self.setMinimumSize(0,0)
            if fullscreen:
                self.showFullScreen()
        self.canvas.setFocus()
        self.canvas.update()

    def _exit_presentation(self):
        if self._presentation_fullscreen:
            self._restore_viewer_window()
        self.presentation_mode = False
        self._presentation_fullscreen = False
        self.canvas.presentation_mode = False
        self.content_layout.setContentsMargins(*self._preview_margins)
        self.content_layout.setSpacing(self._preview_spacing)
        for widget,visible in self._preview_chrome:
            widget.setVisible(visible)
        self.setMinimumSize(self._preview_minimum_size)
        self.update_timeline(self.engine.status().get("playback",{}))
        self.canvas.update()

    def notify(self,text,error=False):
        self.last_notice = text
        self.notice.setText(text)
        self.notice.setStyleSheet("color: #ffc27d;" if error else "color: #78dbc5;")
        self.statusBar().showMessage(text)

    def refresh_devices(self):
        try:
            current = self.source.currentText()
            self.source.clear()
            self.source.addItems(["camera:"+n for n in camera_names()]+["demo://screen"])
            self.source.setCurrentText(current)
            self.notify("摄像头列表已刷新")
        except Exception as exc:
            self.notify(str(exc),True)

    def open_video(self):
        path,_ = QFileDialog.getOpenFileName(self,"选择视频文件","","视频文件 (*.mp4 *.mkv *.mov *.avi *.ts *.webm);;所有文件 (*)")
        if path:
            self.source.setCurrentText(path)
            self.connect_source()

    def sync_render_controls(self,pipeline):
        for widget,value in ((self.sampling,self.sampling.findData(pipeline.interpolation)),
                             (self.sharpen,round(pipeline.sharpen_amount*100))):
            previous = widget.blockSignals(True)
            if widget is self.sampling:
                widget.setCurrentIndex(value)
            else:
                widget.setValue(value)
            widget.blockSignals(previous)

    def change_rendering(self):
        self.engine.set_render_options(self.sampling.currentData(),self.sharpen.value()/100)
        self.cancel_selection()
        pipeline = self.engine.current_pipeline()
        if pipeline:
            pipeline.save(self.data_dir/"autosave.json")
        self.notify(f"已切换为{self.sampling.currentText()}，轻锐化 {self.sharpen.value()}%；直接从原始帧重新生成画面")

    def connect_source(self):
        source = self.source.currentText().strip()
        if not source:
            self.notify("请先选择摄像头、视频文件或填写串流地址",True)
            return
        self.cancel_selection()
        pending = self.pending_pipeline
        self.pending_pipeline = None
        camera_size = None if self.camera_size.currentText()=="设备默认" else tuple(map(int,self.camera_size.currentText().split("×")))
        self.timeline.hide()
        self.engine.connect(source,int(self.capture_fps.currentText()),pending,camera_size)
        if pending:
            self.sync_render_controls(pending)
        self.canvas.set_image(None)
        self.last_key = None
        self.refresh_history()
        self.notify("正在连接视频…若首次连接失败，请确认 OBS 虚拟摄像头已启动。")
        self.connection_started = time.monotonic()
        (self.data_dir/"session.json").write_text(json.dumps({"source":source},ensure_ascii=False),encoding="utf-8")

    @staticmethod
    def format_time(seconds):
        total = max(0,int(seconds))
        hours,remainder = divmod(total,3600)
        minutes,seconds = divmod(remainder,60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def update_timeline(self,playback):
        self.timeline.setVisible(bool(playback.get("is_file")) and not self.presentation_mode)
        self.seek_slider.setEnabled(bool(playback.get("seekable")))
        duration = playback.get("duration_seconds",0)
        if not self.seek_slider.isSliderDown() and not playback.get("seeking"):
            position = playback.get("position_seconds",0)
            blocked = self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(round(position/duration*10000) if duration>0 else 0)
            self.seek_slider.blockSignals(blocked)
            self.playback_time.setText(f"{self.format_time(position)} / {self.format_time(duration) if duration>0 else '--:--'}")

    def timeline_changed(self,value):
        playback = self.engine.status().get("playback",{})
        duration = playback.get("duration_seconds",0)
        self.playback_time.setText(f"{self.format_time(value/10000*duration)} / {self.format_time(duration)}")
        if not self.seek_slider.isSliderDown():
            self.seek_video()

    def seek_video(self):
        playback = self.engine.status().get("playback",{})
        if not playback.get("seekable"):
            return
        seconds = self.seek_slider.value()/10000*playback["duration_seconds"]
        if self.engine.seek(seconds):
            # A selection describes a frozen frame; show the requested moment immediately.
            self.cancel_selection()

    def pump(self):
        status = self.engine.status()
        self.update_timeline(status.get("playback",{}))
        frame = self.engine.output.snapshot()
        live = status["connected"]
        for button in (self.auto_button,self.point_button,self.crop_button):
            button.setEnabled(live)
        if self.canvas.mode:
            self.apply_button.setEnabled(live and ((self.canvas.mode=="perspective" and len(self.canvas.points)==4) or (self.canvas.mode=="crop" and self.canvas.crop_rect() is not None)))
        self.badge.setText("选区已冻结" if self.canvas.mode else (("● 播放" if status.get("playback",{}).get("is_file") else "● 实时") if live else "等待视频"))
        if frame is not None:
            key = (frame.sequence,frame.revision,frame.received_monotonic)
            if key != self.last_key and not self.canvas.mode:
                self.canvas.set_image(frame.image)
                self.last_key = key
            raw = self.engine.raw.snapshot()
            input_size = f"{raw.image.shape[1]} × {raw.image.shape[0]}" if raw else "—"
            size = f"{frame.image.shape[1]} × {frame.image.shape[0]}"
            preview_fps = self.canvas.paint_fps()
            backend = self.engine.capture.backend if self.engine.capture else "—"
            self.engine.preview_metrics = {"paint_fps":round(preview_fps,1), "paint_ms":round(self.canvas.last_paint_ms,2), "frozen":bool(self.canvas.mode)}
            self.metrics.setText(f"输入 {input_size}    输出 {size}    处理 {status['fps']:.1f} fps    预览 {preview_fps:.1f} fps    几何 {status['processing_ms']:.1f} ms    {backend}")
            if live and self.history_list.count()==0:
                self.refresh_history()
        if self.engine.error or (self.engine.capture and self.engine.capture.error):
            error = status["error"]
            if error != self.last_health:
                self.notify(error,True)
                self.last_health = error
        elif live:
            self.last_health = ""
        elif self.engine.capture and time.monotonic()-getattr(self,"connection_started",time.monotonic())>10:
            if self.last_health != "timeout":
                self.notify("超过 10 秒未收到新画面，请检查串流地址或 OBS 虚拟摄像头，然后重新连接。",True)
                self.last_health = "timeout"

    def begin_selection(self,mode):
        frame = self.engine.output.snapshot()
        if frame is None or not self.engine.status()["connected"]:
            return
        self.frozen = frame.image.copy()
        self.canvas.set_image(self.frozen)
        self.canvas.set_mode(mode)
        self.hint.setText("按顺序点击：①左上 → ②右上 → ③右下 → ④左下。右键撤回一个点，选好后点击应用。" if mode=="perspective" else "按住鼠标左键拖出需要保留的矩形，点击应用裁剪。右键清除选区。")

    def cancel_selection(self):
        self.canvas.set_mode(None)
        self.frozen = None
        self.last_key = None
        self.hint.setText("连接视频后，可先裁掉黑边，再进行自动校正或四点校正。")

    def selection_changed(self):
        mode = self.canvas.mode
        valid = (mode=="perspective" and len(self.canvas.points)==4) or (mode=="crop" and self.canvas.crop_rect() is not None)
        self.apply_button.setVisible(bool(mode))
        self.cancel_button.setVisible(bool(mode))
        self.apply_button.setEnabled(bool(valid))
        self.cancel_button.setEnabled(bool(mode))
        self.ratio.setEnabled(mode!="crop")

    def commit(self,pipeline,message):
        self.engine.update(pipeline)
        self.sync_render_controls(pipeline)
        self.cancel_selection()
        pipeline.save(self.data_dir/"autosave.json")
        self.refresh_history()
        self.notify(message+"；已自动保存，可继续叠加操作。")

    def apply_selection(self):
        if not self.canvas.mode:
            return
        try:
            if not self.engine.status()["connected"]:
                raise ValueError("视频已断开，请重新连接后再应用选区")
            pipeline = self.engine.current_pipeline()
            if pipeline is None:
                return
            if self.canvas.mode=="crop":
                roi = self.canvas.crop_rect()
                if roi is None:
                    raise ValueError("请先拖动框选裁剪区域")
                pipeline.crop(roi)
                message = f"已裁剪为 {roi[2]} × {roi[3]}"
            else:
                a,b = map(int,self.ratio.currentText().split(":"))
                w = self.output_width.value()
                size = (w,round(w*b/a))
                pipeline.perspective(self.canvas.points,size)
                message = f"已完成四点校正，输出 {size[0]} × {size[1]}"
            self.commit(pipeline,message)
        except Exception as exc:
            self.notify(str(exc),True)

    def auto_correct(self):
        frame = self.engine.output.snapshot()
        if frame is None or not self.engine.status()["connected"]:
            return
        self.cancel_selection()
        try:
            points,size,details = detect_board(frame.image,self.output_width.value())
            pipeline = self.engine.current_pipeline()
            framing = self.auto_framing.currentData()
            pipeline.perspective(points,size,"标准图自动校正",details,
                                 framing=framing,margin_ratio=self.auto_margin.value()/100 if framing=="margin" else 0)
            self.commit(pipeline,f"已自动校正 · {self.auto_framing.currentText()}，拟合误差 {details['rms_source_pixels']:.2f} 像素；可继续框选裁剪")
        except Exception as exc:
            self.notify(str(exc),True)

    def history(self,action):
        pipeline = self.engine.current_pipeline()
        if pipeline:
            getattr(pipeline,action)()
            self.commit(pipeline,"已撤销上一步" if action=="undo" else "已重做")

    def reset(self):
        pipeline = self.engine.current_pipeline()
        if pipeline:
            # Reset is reversible in this session using a saved snapshot.
            self.reset_snapshot = pipeline
            self.commit(Pipeline(pipeline.input_size,**pipeline.render_options),"已恢复原始画面；原配置保存在 reset-backup.json")
            pipeline.save(self.data_dir/"reset-backup.json")

    def refresh_history(self):
        self.history_list.clear()
        pipeline = self.engine.current_pipeline()
        if pipeline is None:
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)
            return
        self.history_list.addItem(f"原始画面  {pipeline.input_size[0]} × {pipeline.input_size[1]}")
        for i,op in enumerate(pipeline.operations,1):
            name = "裁剪" if op["kind"]=="crop" else op.get("label","四点校正")
            w,h = op["output_size"]
            self.history_list.addItem(f"{i:02d}  {name}  →  {w} × {h}")
        self.history_list.scrollToBottom()
        self.undo_button.setEnabled(bool(pipeline.operations))
        self.redo_button.setEnabled(bool(pipeline.redo_stack))

    def save(self):
        self._save_to(self.data_dir/"calibration.json")

    def _save_to(self,path):
        try:
            pipeline = self.engine.current_pipeline()
            if pipeline is None:
                raise ValueError("请先连接视频")
            if self.canvas.mode:
                raise ValueError("当前选区尚未应用，请先应用选区或取消选择再保存")
            pipeline.save(path)
            self.notify(f"校正配置已导出：{path}")
        except Exception as exc:
            self.notify(str(exc),True)

    def save_as(self):
        path,_ = QFileDialog.getSaveFileName(self,"导出校正配置",str(self.data_dir/"calibration.json"),"校正配置 (*.json)")
        if path:
            self._save_to(path)

    def load(self):
        path,_ = QFileDialog.getOpenFileName(self,"载入校正配置",str(self.data_dir),"校正配置 (*.json)")
        if not path:
            return
        try:
            pipeline = Pipeline.load(path)
            if self.engine.raw.snapshot() is None:
                self.pending_pipeline = pipeline
                self.sync_render_controls(pipeline)
                self.notify("配置已读取，下次连接时应用；输入分辨率必须匹配。")
            else:
                self.commit(pipeline,"已载入校正配置")
        except Exception as exc:
            self.notify(str(exc),True)

    def export_target(self):
        try:
            path = export_board(self.assets_dir)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.assets_dir)))
            self.notify(f"标定图已导出：{path.name}。请在被拍摄屏幕上完整显示原图（可窗口显示，保持原比例）。")
        except Exception as exc:
            self.notify(str(exc),True)

    def open_data(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.data_dir)))

    def toggle_server(self):
        if self.server:
            self.server.stop()
            self.server = None
            self.server_button.setText("启动输出")
            self.port.setEnabled(True)
            self.output_url.setText("输出已停止")
            return
        server = OutputServer(self.engine,self.port.value())
        try:
            server.start()
            self.server = server
            self.output_url.setText(f"http://127.0.0.1:{server.port}/stream.mjpg")
            self.server_button.setText("停止输出")
            self.port.setEnabled(False)
        except OSError:
            server.stop()
            self.notify("输出端口被占用，请更换右侧端口后启动输出。",True)

    def copy_url(self):
        if self.server:
            QApplication.clipboard().setText(f"http://127.0.0.1:{self.server.port}/stream.mjpg")
            self.notify("已复制视频流地址")

    def closeEvent(self,event):
        self.timer.stop()
        pipeline = self.engine.current_pipeline()
        if pipeline:
            try:
                pipeline.save(self.data_dir/"autosave.json")
            except OSError:
                pass
        if self.server:
            self.server.stop()
        self.engine.close()
        super().closeEvent(event)
