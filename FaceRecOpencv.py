# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : FaceRecOpencv.py
# @Explain : 使用Opencv人脸识别

import json
import cv2

from TrainModel import Model

class Face_recognition():
    def __init__(self):
        with open('data/contrast_table', 'r') as f:
            self.contrast_table = json.loads(f.read())
        self.model = Model()
        self.model.load_model(file_path='./model/face.model')
        self.color = (0, 255, 0)
        self.cap = cv2.VideoCapture(0)
        self.cascade_path = "./data/haarcascade_frontalface_default.xml"
    def recongition(self):
        while True:
            ret, frame = self.cap.read()

            if ret is True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            cascade = cv2.CascadeClassifier(self.cascade_path)
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    if image.size == 0:
                        break
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    probability, name_number = self.model.face_predict(image)
                    print(name_number)
                    name = self.contrast_table[str(name_number)]
                    # print('name_number:', name_number)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), self.color, thickness=2)
                    #cv2.putText(frame, name, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    if probability > 0.9:
                        cv2.putText(frame, name, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    else:
                        cv2.putText(frame, 'unknow', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow("face_recognition", frame)
            k = cv2.waitKey(10)
            if k & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

def face():
    fr = Face_recognition()
    fr.recongition()

if __name__ == '__main__':
    fr = Face_recognition()
    fr.recongition()

