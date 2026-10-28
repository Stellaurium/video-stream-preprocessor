from __future__ import annotations
import argparse
import logging
import sys
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QLockFile
from screenprep.media import root_dir
from screenprep.gui import MainWindow

def main():
    parser = argparse.ArgumentParser(description="ScreenPrep screen rectification workstation")
    parser.add_argument("--demo",action="store_true",help="Use the synthetic demo stream")
    parser.add_argument("--port",type=int,default=8765)
    args = parser.parse_args()
    data = root_dir()/"data"
    data.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=str(data/"screenprep.log"),level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",encoding="utf-8")
    app = QApplication(sys.argv)
    app.setApplicationName("ScreenPrep")
    instance_lock = QLockFile(str(data/"screenprep.lock"))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(0):
        QMessageBox.information(None,"ScreenPrep","此程序已在运行，请切换到现有窗口。")
        return 0
    def exception_hook(kind,exc,tb):
        logging.error("Unhandled error",exc_info=(kind,exc,tb))
        QMessageBox.critical(None,"ScreenPrep",f"发生错误：{exc}\n日志位置：{data / 'screenprep.log'}")
    sys.excepthook = exception_hook
    window = MainWindow(demo=args.demo,port=args.port)
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
