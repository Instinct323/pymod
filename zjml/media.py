import cv2 as cv
import mediapipe as mp

from pymod.zjdl.utils.utils import Capture

draw_utils = mp.solutions.drawing_utils
draw_styles = mp.solutions.drawing_styles


def hand_gesture(*args, **kwargs):
    ''' 手掌关键点检测'''
    hands = mp.solutions.hands
    draw_args = [
        hands.HAND_CONNECTIONS,
        draw_styles.get_default_hand_landmarks_style(),
        draw_styles.get_default_hand_connections_style()
    ]
    # 启动检测器
    with hands.Hands(*args, **kwargs) as detector:
        # 逐帧进行识别
        for img in Capture():
            result = detector.process(img)
            # 绘制检测结果
            if result.multi_hand_landmarks:
                for landmark in result.multi_hand_landmarks:
                    draw_utils.draw_landmarks(img, landmark, *draw_args)
                # multi_hand_landmarks: 关键点绝对坐标
                # multi_hand_world_landmarks: 关键点相对坐标
                # multi_handedness: 手性检测结果
                yield result
            cv.imshow('Hand Gesture', img)
            cv.waitKey(1)


def face_detect(*args, **kwargs):
    ''' 人脸关键点检测'''
    fd = mp.solutions.face_detection
    # 启动检测器
    with fd.FaceDetection(*args, **kwargs) as detector:
        # 逐帧进行识别
        for img in Capture():
            result = detector.process(img).detections
            # 绘制检测结果
            if result:
                for detection in result:
                    draw_utils.draw_detection(img, detection)
                yield result
            cv.imshow('Face Detect', img)
            cv.waitKey(1)


if __name__ == '__main__':
    for i in hand_gesture(): pass
