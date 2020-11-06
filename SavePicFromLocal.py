# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : SavePicFromLocal.py
# @Explain : 本地处理图片

import sys
import os
import cv2
import dlib
import Utils as ut
import numpy as np

def change(name):

    input_dir = './local'
    output_dir = './pic/'+name
    size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = dlib.get_frontal_face_detector()

    index = 1
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                print('Being processed picture %s' % index)
                img_path = path+'/'+filename
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_img, 1)
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:y1,x2:y2]
                    face = cv2.resize(face, (size,size))
                    cv2.imshow('image',face)
                    cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                    index += 1
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)

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

def changeZeng(name):

    input_dir = './local'
    output_dir = './pic/'+name
    size = 128

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = dlib.get_frontal_face_detector()

    index = 1
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                print('Being processed picture %s' % index)
                img_path = path+'/'+filename
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_img, 1)
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0

                    for s in range(-6,7,2):
                        M = cv2.getRotationMatrix2D(( ((y2-x2)/2)+x2 , ((y1-x1)/2)+x1 ), s, 1)
                        rows, cols,_ = img.shape
                        MM = cv2.warpAffine(img,M,(2*cols,2*rows))
                        faceM = MM[x1:y1, x2:y2]
                        faceM = cv2.resize(faceM, (size, size))
                        for d in range(1, 3):
                            for l in range(10, 30, 10):
                                iii = ut.contrast_brightness(faceM, d, l)
                                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', iii)
                                index += 1
                                cv2.imshow('image', iii)

                    face3 = img[x1 - 20:y1 - 20, x2 - 20:y2 - 20]
                    face3 = cv2.resize(face3, (size, size))
                    cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face3)
                    index += 1

                    face2 = img[x1-10:y1-10,x2-10:y2-10]
                    face2 = cv2.resize(face2, (size,size))
                    cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face2)
                    index += 1

                    face = img[x1:y1,x2:y2]
                    face = cv2.resize(face, (size,size))
                    cv2.imshow('image',face)
                    cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                    index += 1

                    for d in range(1,3):
                        for l in range(-30,60,10):
                            iii = ut.contrast_brightness(face,d,l)
                            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', iii)
                            index += 1
                            cv2.imshow('image',iii)


                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)

if __name__ == '__main__':
    #change("xiaoman")
    changeZeng("pengcheng")
    print("main")