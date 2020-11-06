# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : main.py
# @Explain : 主函数

import Utils as ut
import SavePic as sp
import TrainModel as tm
import FaceRecOpencv as fo
import FaceRecDlib as fd
import SavePicFromLocal as sl
import Expression as ex
import FaceRecDlibAddExp as fde
import KouZhao2 as kz

while True:
    print("-----------------------Face recognition----------------------")
    print("1:录入目标人脸.(Opencv)")
    print("2:录入目标人脸.(Dlib)")
    print("3:本地加载图片识别人脸")
    print("4:训练模型")
    print("5.人脸识别.(Opencv)")
    print("6.人脸识别.(Dlib)")
    print("7:本地加载图片识别人脸-数据增强")
    print("8:表情识别")
    print("9:人脸识别+表情识别.(Dlib)")
    print("10:口罩识别 TensorFlow")
    print("c:检查Python库版本")
    print("q:退出.")
    print("-----------------------Face recognition----------------------")
    str = input("请输入序号：")
    if str == "1":
        pic_name = input("请输入该人物标签:")
        pic_size = input("请输入录入张数(int):")
        sp.SavePicFromVideoByOpencv(int(pic_size), "./pic/"+pic_name)
    elif str == "2":
        pic_name = input("请输入该人物标签:")
        pic_size = input("请输入录入张数(int):")
        relight = input("是否开启亮度变化:")
        sp.SavePicFromVideoByDlib(int(pic_size), "./pic/"+pic_name,int(relight))
    elif str == "3":
        pic_name = input("请输入该人物标签:")
        sl.change(pic_name)
    elif str == "4":
        tm.train()
    elif str == "5":
        fo.face()
    elif str == "6":
        fd.face()
    elif str == "7":
        pic_name = input("请输入该人物标签:")
        sl.change(pic_name)
    elif str == "8":
        ex.test()
    elif str == "9":
        fde.recongition()
    elif str == "10":
        kz.kou()
    elif str == "c":
        ut.CheckVersion()
    elif str == "q":
        exit()