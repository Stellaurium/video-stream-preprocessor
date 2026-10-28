import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from screenprep import gui
from screenprep.geometry import Pipeline


def test_gui_buttons_points_crop_undo_save_and_render(tmp_path,monkeypatch):
    real_root = Path(gui.__file__).resolve().parent.parent
    monkeypatch.setattr(gui,'root_dir',lambda:tmp_path)
    monkeypatch.setattr(gui,'camera_names',lambda:[])
    app = QApplication.instance() or QApplication([])
    window = gui.MainWindow(demo=True,port=18765)
    window.show()
    def wait_ready(predicate,timeout=5):
        started=time.perf_counter()
        while time.perf_counter()-started<timeout:
            app.processEvents()
            window.pump()
            if predicate():
                return
            time.sleep(.02)
        raise AssertionError('GUI state did not become ready')
    try:
        wait_ready(lambda:window.engine.status()['connected'])
        window.auto_framing.setCurrentIndex(window.auto_framing.findData("screen"))
        QTest.mouseClick(window.auto_button,Qt.MouseButton.LeftButton)
        wait_ready(lambda:window.engine.output.snapshot().revision==window.engine.revision)
        assert window.engine.current_pipeline().operations[-1]['label']=='标准图自动校正'
        assert window.engine.current_pipeline().output_size==(1920,1080)
        if os.getenv('SCREENPREP_RENDER_QA')=='1':
            window.grab().save(str(real_root/'data'/'ui-preview.png'))
        QTest.mouseClick(window.point_button,Qt.MouseButton.LeftButton)
        assert window.canvas.mode=='perspective'
        window.ratio.setCurrentText('4:3')
        window.output_width.setValue(640)
        for point in [(20,20),(1250,25),(1240,690),(25,695)]:
            QTest.mouseClick(window.canvas,Qt.MouseButton.LeftButton,pos=window.canvas.display_point(point).toPoint())
        assert len(window.canvas.points)==4
        assert window.apply_button.isEnabled()
        QTest.mouseClick(window.apply_button,Qt.MouseButton.LeftButton)
        wait_ready(lambda:window.engine.output.snapshot().revision==window.engine.revision)
        assert window.engine.current_pipeline().output_size==(640,480)
        QTest.mouseClick(window.crop_button,Qt.MouseButton.LeftButton)
        a=window.canvas.display_point((40,30)).toPoint()
        b=window.canvas.display_point((500,400)).toPoint()
        QTest.mousePress(window.canvas,Qt.MouseButton.LeftButton,pos=b)
        QTest.mouseMove(window.canvas,a)
        QTest.mouseRelease(window.canvas,Qt.MouseButton.LeftButton,pos=a)
        QTest.mouseClick(window.apply_button,Qt.MouseButton.LeftButton)
        pipeline=window.engine.current_pipeline()
        assert pipeline.operations[-1]['kind']=='crop'
        assert 459<=pipeline.output_size[0]<=463
        assert 369<=pipeline.output_size[1]<=373
        cropped_size=pipeline.output_size
        QTest.mouseClick(window.undo_button,Qt.MouseButton.LeftButton)
        assert window.engine.current_pipeline().output_size==(640,480)
        QTest.mouseClick(window.redo_button,Qt.MouseButton.LeftButton)
        assert window.engine.current_pipeline().output_size==cropped_size
        window.save()
        loaded=Pipeline.load(tmp_path/'data'/'calibration.json')
        assert loaded.output_size==cropped_size
        assert len(loaded.operations)==3
        assert window.canvas.mode is None
    finally:
        window.close()
        app.processEvents()

def test_preview_timer_keeps_up_with_30fps_source(tmp_path,monkeypatch):
    from PySide6.QtCore import QEventLoop,QTimer
    monkeypatch.setattr(gui,'root_dir',lambda:tmp_path)
    monkeypatch.setattr(gui,'camera_names',lambda:[])
    app=QApplication.instance() or QApplication([])
    window=gui.MainWindow(demo=True,port=18766)
    window.capture_fps.setCurrentText("30")
    window.show()
    try:
        loop=QEventLoop()
        QTimer.singleShot(3200,loop.quit)
        loop.exec()
        source_fps=window.engine.status()['fps']
        preview_fps=window.canvas.paint_fps()
        assert source_fps >= 26,source_fps
        assert preview_fps >= source_fps*0.85,(source_fps,preview_fps)
    finally:
        window.close()
        app.processEvents()

def test_video_timeline_click_drag_keyboard_and_hidden_for_live(tmp_path,monkeypatch):
    from PySide6.QtCore import QPoint
    from tests.test_seek import make_seek_video
    real_root=Path(gui.__file__).resolve().parent.parent
    monkeypatch.setattr(gui,'root_dir',lambda:tmp_path)
    monkeypatch.setattr(gui,'camera_names',lambda:[])
    app=QApplication.instance() or QApplication([])
    window=gui.MainWindow(demo=True,port=18767)
    window.show()
    def wait_ready(predicate,timeout=4):
        start=time.perf_counter()
        while time.perf_counter()-start<timeout:
            app.processEvents()
            window.pump()
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError('Timeline state did not become ready')
    try:
        wait_ready(lambda:window.engine.status()['connected'])
        assert window.timeline.isHidden()
        path=tmp_path/'timeline.mp4'
        make_seek_video(path)
        window.source.setCurrentText(str(path))
        window.connect_source()
        wait_ready(lambda:window.seek_slider.isEnabled() and window.timeline.isVisible() and window.engine.status()['connected'])
        assert window.playback_time.text().endswith('/ 00:09')
        window.begin_selection('crop')
        assert window.canvas.mode=='crop'
        slider=window.seek_slider
        point=QPoint(round(slider.width()*.8),slider.height()//2)
        QTest.mousePress(slider,Qt.MouseButton.LeftButton,pos=point)
        assert slider.isSliderDown() and slider.value()>7500
        value=slider.value()
        QTest.qWait(100)
        window.pump()
        assert slider.value()==value  # Playback updates cannot fight a dragged handle.
        target=value/10000*9
        QTest.mouseRelease(slider,Qt.MouseButton.LeftButton,pos=point)
        assert window.canvas.mode is None
        wait_ready(lambda:(p:=window.engine.output.snapshot()) is not None and p.playback_seconds is not None and target-.06<=p.playback_seconds<=target+.25)
        assert window.playback_time.text().startswith('00:07')
        if os.getenv('SCREENPREP_RENDER_QA')=='1':
            window.grab().save(str(real_root/'data'/'ui-video-timeline.png'))
        QTest.keyClick(slider,Qt.Key.Key_Home)
        wait_ready(lambda:(p:=window.engine.output.snapshot()) is not None and p.playback_seconds is not None and p.playback_seconds<.2)
        # Reconnect to a live/demo source; the file-only row must disappear.
        window.source.setCurrentText('demo://screen')
        window.connect_source()
        wait_ready(lambda:window.engine.status()['connected'])
        assert window.timeline.isHidden()
        assert not window.engine.status()['playback']['is_file']
    finally:
        window.close()
        app.processEvents()

def test_fullscreen_shortcuts_keep_video_playing_and_restore_layout(tmp_path,monkeypatch):
    from PySide6.QtCore import QPoint
    from tests.test_seek import make_seek_video
    real_root=Path(gui.__file__).resolve().parent.parent
    monkeypatch.setattr(gui,'root_dir',lambda:tmp_path)
    monkeypatch.setattr(gui,'camera_names',lambda:[])
    app=QApplication.instance() or QApplication([])
    window=gui.MainWindow(demo=True,port=18769)
    window.resize(1120,760)
    window.show()
    window.activateWindow()
    def wait_ready(predicate,timeout=4):
        start=time.perf_counter()
        while time.perf_counter()-start<timeout:
            app.processEvents()
            window.pump()
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError('Fullscreen state did not become ready')
    try:
        wait_ready(lambda:window.engine.status()['connected'])
        path=tmp_path/'fullscreen.mp4'
        make_seek_video(path)
        window.source.setCurrentText(str(path))
        window.connect_source()
        wait_ready(lambda:window.timeline.isVisible() and window.engine.status()['connected'])
        pipeline=window.engine.current_pipeline()
        pipeline.crop((20,20,80,60))
        window.commit(pipeline,'test crop')
        wait_ready(lambda:window.engine.output.snapshot().revision==window.engine.revision)
        old_geometry=window.geometry()
        old_minimum=window.minimumSize()
        revision=window.engine.revision
        window.begin_selection('crop')
        QTest.keyClick(window.canvas,Qt.Key.Key_F11)
        wait_ready(lambda:window.isFullScreen())
        assert window.presentation_mode and window.canvas.mode is None
        for widget in [window.sidebar,window.header,window.hint,window.timeline,window.metrics,window.output_box,window.notice,window.statusBar()]:
            assert widget.isHidden()
        # The recurring video progress updates must not make controls reappear.
        sequence=window.engine.output.snapshot().sequence
        wait_ready(lambda:window.engine.output.snapshot().sequence>=sequence+3)
        assert window.timeline.isHidden()
        assert window.engine.revision==revision
        assert window.engine.output.snapshot().image.shape==(60,80,3)
        assert window.canvas.mapTo(window.centralWidget(),QPoint(0,0))==QPoint(0,0)
        assert window.canvas.size()==window.centralWidget().size()
        image_rect=window.canvas.image_rect()
        assert abs(image_rect.width()/image_rect.height()-4/3)<.001
        assert image_rect.width()==window.canvas.width() or image_rect.height()==window.canvas.height()
        if os.getenv('SCREENPREP_RENDER_QA')=='1':
            window.grab().save(str(real_root/'data'/'ui-fullscreen.png'))
        QTest.keyClick(window.canvas,Qt.Key.Key_Escape)
        wait_ready(lambda:not window.isFullScreen())
        assert not window.presentation_mode
        assert window.geometry()==old_geometry
        assert window.minimumSize()==old_minimum
        assert window.sidebar.isVisible() and window.header.isVisible()
        assert window.timeline.isVisible() and window.statusBar().isVisible()
        # A maximized window must return to maximized, including its normal geometry.
        window.showMaximized()
        app.processEvents()
        assert window.isMaximized()
        QTest.mouseClick(window.fullscreen_button,Qt.MouseButton.LeftButton)
        wait_ready(lambda:window.isFullScreen())
        QTest.keyClick(window.canvas,Qt.Key.Key_F11)
        wait_ready(lambda:window.isMaximized() and not window.isFullScreen())
        window.showNormal()
        app.processEvents()
        assert window.geometry()==old_geometry
        window.begin_selection('crop')
        QTest.keyClick(window.canvas,Qt.Key.Key_Escape)
        assert window.canvas.mode is None
    finally:
        window.close()
        app.processEvents()

def test_windowed_view_preserves_frame_geometry_and_switches_to_fullscreen(tmp_path,monkeypatch):
    from PySide6.QtCore import QPoint
    monkeypatch.setattr(gui,'root_dir',lambda:tmp_path)
    monkeypatch.setattr(gui,'camera_names',lambda:[])
    app=QApplication.instance() or QApplication([])
    window=gui.MainWindow(demo=True,port=18770)
    window.resize(1120,760)
    window.show()
    window.activateWindow()
    def settle():
        app.processEvents()
        window.pump()
        app.processEvents()
    try:
        deadline=time.perf_counter()+4
        while not window.engine.status()['connected'] and time.perf_counter()<deadline:
            settle()
            time.sleep(.01)
        assert window.engine.status()['connected']
        geometry=window.geometry()
        flags=window.windowFlags()
        QTest.keyClick(window.canvas,Qt.Key.Key_F10)
        settle()
        assert window.presentation_mode and not window.isFullScreen()
        assert window.geometry()==geometry and window.windowFlags()==flags
        assert not window.isMaximized()
        assert window.sidebar.isHidden() and window.header.isHidden()
        assert window.canvas.mapTo(window.centralWidget(),QPoint(0,0))==QPoint(0,0)
        assert window.canvas.size()==window.centralWidget().size()
        # Switching to real fullscreen and back must recover the same windowed view.
        QTest.keyClick(window.canvas,Qt.Key.Key_F11)
        settle()
        assert window.isFullScreen()
        QTest.keyClick(window.canvas,Qt.Key.Key_F10)
        settle()
        assert window.presentation_mode and not window.isFullScreen()
        assert window.geometry()==geometry and window.sidebar.isHidden()
        # User resizing in the windowed view is retained on exit.
        window.resize(1200,800)
        settle()
        resized=window.geometry()
        QTest.keyClick(window.canvas,Qt.Key.Key_Escape)
        settle()
        assert not window.presentation_mode and window.geometry()==resized
        assert window.sidebar.isVisible() and window.header.isVisible()
        window.showMaximized()
        settle()
        assert window.isMaximized()
        QTest.mouseClick(window.windowed_view_button,Qt.MouseButton.LeftButton)
        settle()
        assert window.presentation_mode and window.isMaximized() and not window.isFullScreen()
        QTest.keyClick(window.canvas,Qt.Key.Key_F10)
        settle()
        assert not window.presentation_mode and window.isMaximized()
    finally:
        window.close()
        app.processEvents()

def test_automatic_framing_controls_keep_small_window_surroundings(tmp_path,monkeypatch):
    from screenprep.media import Capture,Frame
    from tests.test_framing import small_window_scene
    camera,_=small_window_scene()
    def synthetic_camera(self):
        sequence=0
        while not self.stop_event.is_set():
            sequence+=1
            self.mailbox.publish(Frame(camera.copy(),sequence,sequence/self.fps,time.monotonic()))
            self.stop_event.wait(1/self.fps)
    real_root=Path(gui.__file__).resolve().parent.parent
    monkeypatch.setattr(gui,'root_dir',lambda:tmp_path)
    monkeypatch.setattr(gui,'camera_names',lambda:[])
    monkeypatch.setattr(Capture,'_demo',synthetic_camera)
    app=QApplication.instance() or QApplication([])
    window=gui.MainWindow(demo=True,port=18771)
    window.show()
    def wait_ready(predicate,timeout=4):
        start=time.perf_counter()
        while time.perf_counter()-start<timeout:
            app.processEvents()
            window.pump()
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError('Automatic framing state did not become ready: '+window.last_notice)
    def wait_rendered():
        wait_ready(lambda:window.engine.output.snapshot() is not None and window.engine.output.snapshot().revision==window.engine.revision)
    try:
        wait_ready(lambda:window.engine.status()['connected'])
        assert window.output_width.value()==1920
        assert window.auto_framing.currentData()=='canvas'
        assert not window.auto_margin.isEnabled()
        QTest.mouseClick(window.auto_button,Qt.MouseButton.LeftButton)
        wait_rendered()
        pipeline=window.engine.current_pipeline()
        assert pipeline.operations[-1]['framing']=='canvas'
        output=window.engine.output.snapshot().image
        assert ((output[:,:,1]>190)&(output[:,:,0]<50)).any()  # Notes outside the standard image survive.
        # Manually crop the retained canvas using the existing workflow.
        window.begin_selection('crop')
        h,w=output.shape[:2]
        a=window.canvas.display_point((10,10)).toPoint()
        b=window.canvas.display_point((w-20,h-20)).toPoint()
        QTest.mousePress(window.canvas,Qt.MouseButton.LeftButton,pos=a)
        QTest.mouseMove(window.canvas,b)
        QTest.mouseRelease(window.canvas,Qt.MouseButton.LeftButton,pos=b)
        QTest.mouseClick(window.apply_button,Qt.MouseButton.LeftButton)
        wait_rendered()
        assert window.engine.current_pipeline().operations[-1]['kind']=='crop'
        window.history('undo')
        window.history('undo')
        wait_rendered()
        assert not window.engine.current_pipeline().operations
        window.auto_framing.setCurrentIndex(window.auto_framing.findData('margin'))
        assert window.auto_margin.isEnabled()
        window.auto_margin.setValue(25)
        QTest.mouseClick(window.auto_button,Qt.MouseButton.LeftButton)
        wait_rendered()
        pipeline=window.engine.current_pipeline()
        assert pipeline.output_size==(2880,1620)
        assert pipeline.operations[-1]['margin_ratio']==.25
        window.save()
        reloaded=Pipeline.load(tmp_path/'data'/'calibration.json')
        np.testing.assert_array_equal(reloaded.render(camera),pipeline.render(camera))
        if os.getenv('SCREENPREP_RENDER_QA')=='1':
            window.grab().save(str(real_root/'data'/'ui-auto-framing.png'))
        window.history('undo')
        wait_rendered()
        window.auto_framing.setCurrentIndex(window.auto_framing.findData('screen'))
        assert not window.auto_margin.isEnabled()
        QTest.mouseClick(window.auto_button,Qt.MouseButton.LeftButton)
        wait_rendered()
        assert window.engine.current_pipeline().output_size==(1920,1080)
    finally:
        window.close()
        app.processEvents()


def test_quality_controls_apply_without_reconnect_and_save(tmp_path,monkeypatch):
    monkeypatch.setattr(gui,'root_dir',lambda:tmp_path)
    monkeypatch.setattr(gui,'camera_names',lambda:[])
    app=QApplication.instance() or QApplication([])
    window=gui.MainWindow(demo=True,port=18772)
    window.show()
    def wait_ready(predicate,timeout=6):
        start=time.perf_counter()
        while time.perf_counter()-start<timeout:
            app.processEvents();window.pump()
            if predicate():return
            time.sleep(.01)
        raise AssertionError(window.last_notice)
    try:
        assert window.capture_fps.currentText()=='15'
        assert window.sampling.currentData()=='lanczos4'
        assert window.sharpen.value()==20
        wait_ready(lambda:window.engine.status()['connected'])
        window.auto_framing.setCurrentIndex(window.auto_framing.findData('screen'))
        QTest.mouseClick(window.auto_button,Qt.MouseButton.LeftButton)
        wait_ready(lambda:window.engine.output.snapshot().revision==window.engine.revision)
        capture=window.engine.capture
        original=window.engine.current_pipeline()
        window.sampling.setCurrentIndex(window.sampling.findData('linear'))
        window.sharpen.setValue(0)
        wait_ready(lambda:window.engine.output.snapshot().revision==window.engine.revision)
        current=window.engine.current_pipeline()
        assert window.engine.capture is capture
        assert current.render_options==dict(interpolation='linear',sharpen_amount=0.0)
        np.testing.assert_array_equal(original.matrix,current.matrix)
        assert current.output_size==original.output_size
        window.save()
        assert Pipeline.load(tmp_path/'data'/'calibration.json').render_options==current.render_options
        window.commit(original,'恢复高质量配置')
        assert window.sampling.currentData()=='lanczos4' and window.sharpen.value()==20
        if os.getenv('SCREENPREP_RENDER_QA')=='1':
            root=Path(gui.__file__).resolve().parent.parent
            window.grab().save(str(root/'data'/'quality-comparison'/'ui-quality.png'))
    finally:
        window.close();app.processEvents()
