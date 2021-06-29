#!/usr/bin/env python
import cv2  
import numpy as np 


cap = cv2.VideoCapture('VID_20210507_093251.mp4')


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('2.mp4', fourcc, 20, (width, height))

while(1):
    ret, frame = cap.read()


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 0])
    upper_red = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    img_Gaussian_mask = cv2.GaussianBlur(mask, (13, 13), 0)

    res = cv2.bitwise_and(frame, frame, mask = mask)

    res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    gray_img = cv2.cvtColor(res_rgb, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

    res_2 = cv2.bitwise_and(frame,frame, mask = thresh)
    res_2_Gaussian = cv2.GaussianBlur(res_2, (5, 5), 0)

    out.write(res_2_Gaussian)
    # show a frame
    cv2.imshow("capture", res_2_Gaussian)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows() 





