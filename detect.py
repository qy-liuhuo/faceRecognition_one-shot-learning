from yunNet.yunNet import YunNet
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import *
from nets.Recognizer import Recognizer
import argparse



parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--path', '-p', help='请输入待检测照片的路径', required=True)
args = parser.parse_args()


if __name__ == '__main__':
    # 初始化检测器，用于框选人脸
    faceDetectionModel = YunNet()
    device_id = 0
    cap = cv.VideoCapture(device_id)
    testImg = cv.imread(args.path)
    faceDetectionModel.resetInputSize(testImg.shape[1], testImg.shape[0])
    # 初始化识别器，用于识别身份
    faceRecognizer = Recognizer()
    tm = cv.TickMeter()

    _, faces = faceDetectionModel.detect(testImg)

    if faces is not None:
        try:
            face = faces[0]
            coords = face[:-1].astype(np.int32)
            cropped = testImg[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
            # 识别结果
            result = faceRecognizer.predict(Image.fromarray(cv.cvtColor(cropped, cv.COLOR_BGR2RGB)))
            # Draw face bounding box
            cv.rectangle(testImg, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (255, 0, 0), 1)

            cv.putText(testImg, result[0] + '{:.4f}'.format(result[-1]), (coords[0], coords[1] - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 255))
            cv.imshow("result",testImg)
            cv.waitKey(0)
        except:
            print("识别失败")
    else:
        print("未检测到人脸")