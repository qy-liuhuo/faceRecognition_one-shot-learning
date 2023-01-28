import cv2 as cv


class YunNet:

    def __init__(self, model=r"./weights/face_detection_yunet.onnx", top_k=1):
        self.yunet = cv.FaceDetectorYN.create(
            model=model,
            config='',
            input_size=(640, 640),
            score_threshold=0.9,
            nms_threshold=0.6,
            top_k=top_k,
            backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv.dnn.DNN_TARGET_CPU
        )

    # 重制输入大小
    def resetInputSize(self, width, height):
        self.yunet.setInputSize([width, height])
        return self.yunet.getInputSize()

    def detect(self, image):
        return self.yunet.detect(image)
