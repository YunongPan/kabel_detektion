#!/usr/bin/env python
import cv2  
import numpy as np 
import matplotlib.pyplot as plt

frame = cv2.imread("./DSC00088.JPG")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_red = np.array([1, 100, 0])
upper_red = np.array([20, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

res = cv2.bitwise_and(frame, frame, mask = mask)

res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

res_rgb_Gaussian = cv2.GaussianBlur(res_rgb, (5, 5), 0)


gray_img = cv2.cvtColor(res_rgb_Gaussian, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)

res_2 = cv2.bitwise_and(frame,frame, mask= thresh)
res_2_Gaussian = cv2.GaussianBlur(res_2, (5, 5), 0)
res_2_rgb = cv2.cvtColor(res_2_Gaussian, cv2.COLOR_BGR2RGB)

plt.imshow(res_2_rgb)
plt.show()



