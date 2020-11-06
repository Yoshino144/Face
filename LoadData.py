# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : LoadData.py
# @Explain : 加载类

import json
import os
import cv2
import numpy as np

IMAGE_SIZE = 128

def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))

images = []
labels = []

def read_path(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                print(str(full_path))
                try:
                    #image = cv2.imread(full_path)
                    image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception as e:
                    print(str(full_path)+str(e))
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                if "\\" in path_name:
                    labels.append(path_name.split('\\')[-1])
                else :
                    labels.append(path_name.split('/')[-1])
    return images, labels

def load_dataset(path_name):
    images, labels = read_path(path_name)
    print('labels:', labels)
    images = np.array(images)
    print(images.shape)
    labels1 = list(set(labels))
    face_num = len(labels1)
    print('face_num:',face_num)
    num = [i for i in range(face_num)]
    contrast_table = dict(zip(num, labels1))
    with open('./data/contrast_table', 'w') as f:
        f.write(json.dumps(contrast_table))
    # print('contrast_table:', contrast_table)
    for index, name in contrast_table.items():
        for i in range(len(labels)):
            if labels[i] == name:
                labels[i] = index
    # print(labels)
    labels = np.array(labels)
    return images, labels, face_num

if __name__ == '__main__':
    images, labels, face_num = load_dataset("./pic")
