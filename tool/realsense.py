import cv2
import numpy as np
import pyrealsense2 as rs


def frame2numpy(frame: rs.frame) -> np.ndarray:
    return np.asanyarray(frame.get_data())


def run(w, h, fps=30, show=True):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

    pipe = rs.pipeline()
    pipe.start(cfg)

    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame: continue
        color_image = cv2.cvtColor(frame2numpy(color_frame), cv2.COLOR_RGB2BGR)
        depth_image = frame2numpy(depth_frame)

        if show:
            images = np.hstack((color_image, cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET)))
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)


if __name__ == '__main__':
    devices = rs.context().query_devices()
    for i in devices: print(i)

    run(640, 480)
