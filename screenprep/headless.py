"""Run the same processor without the desktop UI."""
import argparse
import time
from .geometry import Pipeline
from .media import Engine
from .output import OutputServer

def main():
    p = argparse.ArgumentParser(description="ScreenPrep headless stream processor")
    p.add_argument("--source",default="camera:OBS Virtual Camera")
    p.add_argument("--config",required=True)
    p.add_argument("--fps",type=int,choices=[15,30,60],default=15)
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--camera-size",default="1920x1080")
    args = p.parse_args()
    pipeline = Pipeline.load(args.config)
    size = tuple(map(int,args.camera_size.lower().split('x')))
    engine = Engine()
    server = OutputServer(engine,args.port)
    try:
        engine.connect(args.source,args.fps,pipeline,size)
        server.start()
        print(f"MJPEG: http://127.0.0.1:{server.port}/stream.mjpg",flush=True)
        print("Lossless: /frame.npy | /frame.png | Status: /status",flush=True)
        while True:
            if engine.error or (engine.capture and engine.capture.error):
                raise RuntimeError(engine.status()["error"])
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        engine.close()

if __name__=="__main__":
    main()
