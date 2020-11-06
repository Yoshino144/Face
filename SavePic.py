# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : SavePic.py
# @Explain : 读取图片

import os
import cv2
import sys
import Utils
import dlib
import random
from PIL import Image


def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    # image = []
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


def SavePicFromVideoByDlib(pic_num, path,re):
    if not os.path.exists(path):
        os.makedirs(path)
    detector = dlib.get_frontal_face_detector()
    camera = cv2.VideoCapture(0)
    index = 1
    color = (0, 255, 0)
    while camera.isOpened():
        if (index <= pic_num):
            success, img = camera.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_img, 1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                if re:
                    face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                face = cv2.resize(face, (64, 64))
                cv2.rectangle(img, (x2 - 10, x1 - 10), (y2 + 10, y1 + 10), color, 2)
                cv2.imshow('face', face)
                #face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(path + '/' + str(index) + '.jpg', face)
                print(path + '/' + str(index) + '.jpg')
                index += 1
            img = Utils.cv2ImgAddText(img, "人脸识别 BY PC", 0, img.shape[0] - 20, (0, 255, 0), 18)
            cv2.imshow('video', img)
        else:
            print('Finished!')
            break
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


def SavePicFromVideoByOpencv(pic_num, path):
    if not os.path.exists(path):
        os.mkdir(path)
    cap = cv2.VideoCapture(0)
    classfier = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                img_name = '%s/%d.jpg' % (path, num)
                print(img_name)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)
                cv2.imshow('face', image)
                num += 1
                if num > pic_num:
                    break
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                frame = Utils.cv2ImgAddText(frame, 'num:%d' % (num), x + 3, y + 3, (255, 0, 255), 20)
        if num > pic_num: break
        frame = Utils.cv2ImgAddText(frame, "人脸识别 BY PC", 0, frame.shape[0] - 20, (0, 255, 0), 18)
        cv2.imshow("123", frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    SavePicFromVideoByDlib(2, "./pic/pengcheng",0)
    #SavePicFromVideoByOpencv(1000, "./pic/pcc")
    print("main")