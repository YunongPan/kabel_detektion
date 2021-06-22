#!/usr/bin/env python
import cv2  
import numpy as np 
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('VID_20210507_091819.mp4')

while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,100,0])
    upper_red = np.array([10,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    img_Guassian_mask = cv2.GaussianBlur(mask,(13,13),0)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    img_Guassian = cv2.GaussianBlur(res_rgb,(5,5),0)

    gray_img = cv2.cvtColor(img_Guassian, cv2.COLOR_BGR2GRAY)
    ret,thresh2 = cv2.threshold(gray_img,70,255,cv2.THRESH_BINARY)

    res_2 = cv2.bitwise_and(frame,frame, mask= thresh2)

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000,1000)
    cv2.imshow('image',res_2)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()






