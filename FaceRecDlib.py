# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : FacRecDlib.py
# @Explain : 使用Dlib人脸识别

import json
import cv2
import dlib

from TrainModel import Model


def recongition():
    with open('data/contrast_table', 'r') as f:
        contrast_table = json.loads(f.read())
    model = Model()
    model.load_model(file_path='./model/face.model')
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    index = 1
    color = (0, 255, 0)

    while True:
        ret, frame = cap.read()

        if ret is True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        dets = detector(frame_gray, 1)
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = frame[x1-10:y1+10, x2-10:y2+10]
            if face.size == 0:
                break
            #cv2.imshow("op", face)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            probability, name_number = model.face_predict(face)
            print(name_number)
            name = contrast_table[str(name_number)]
            # print('name_number:', name_number)
            cv2.rectangle(frame, (x2 - 10, x1 - 10), (y2 + 10, y1 + 10), color, 2)
            #cv2.putText(frame, name, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if probability > 0.50:
                cv2.putText(frame, name, (x2 + 30, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                cv2.putText(frame, 'unknow', (x2 + 30, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("face_recognition", frame)
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def face():
    recongition()

if __name__ == '__main__':
    recongition()

