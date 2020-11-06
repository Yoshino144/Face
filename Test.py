import cv2
import numpy as np
from matplotlib import pyplot as plt

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    cv2.imshow('img', img)
    rows, cols, ch = img.shape
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow('dst', dst)
    c = cv2.waitKey(1)
    if c == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()