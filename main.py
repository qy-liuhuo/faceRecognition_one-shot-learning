from yunNet.yunNet import YunNet
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import *
from nets.Recognizer import Recognizer


if __name__ == '__main__':
    # 初始化检测器，用于框选人脸
    faceDetectionModel = YunNet()
    device_id = 0
    cap = cv.VideoCapture(device_id)
    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    faceDetectionModel.resetInputSize(frame_w, frame_h)
    # 初始化识别器，用于识别身份
    faceRecognizer = Recognizer()
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            print('No frames grabbed!')
        tm.start()
        _, faces = faceDetectionModel.detect(frame)
        image = frame.copy()
        # 判断是否存在人脸
        if faces is not None:
            try:
                face = faces[0]
                coords = face[:-1].astype(np.int32)
                cropped = frame[coords[1]:coords[1] + coords[3],coords[0]:coords[0] + coords[2]]
                # 识别结果
                result=faceRecognizer.predict(Image.fromarray(cv.cvtColor(cropped, cv.COLOR_BGR2RGB)))
                # Draw face bounding box
                cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (255, 0, 0), 1)

                cv.putText(image, result[0]+'{:.4f}'.format(result[-1]), (coords[0], coords[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 255))
            except:
                continue
        tm.stop()
        cv.putText(image, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv.imshow('demo', image)
        tm.reset()