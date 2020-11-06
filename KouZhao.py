# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : KouZhao.py
# @Explain : 测试

import cv2
import Utils

detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
mask_detector = cv2.CascadeClassifier('./data/kouzhao.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = img[y:y + h, x:x + w]
        mask_face = mask_detector.detectMultiScale(gray, 1.1, 5)
        if len(mask_face) > 0:
            for (x2, y2, w2, h2) in mask_face:
                img = Utils.cv2ImgAddText(img, "佩戴了口罩", x + h * 0.3, y, (0,255,0), 20)
        else:
            img = Utils.cv2ImgAddText(img, "没有佩戴口罩", x + h * 0.3, y, (255,0,0), 20)

    cv2.imshow('Cheney', img)
    cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()