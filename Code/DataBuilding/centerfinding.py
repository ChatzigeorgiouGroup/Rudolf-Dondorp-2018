# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:28:35 2017

@author: ddo003
"""

import os
import cv2
import numpy as np

#
#f = "wt_chorionated49hr_animal1_5min.avi"
#
#cap = cv2.VideoCapture(f)
def find_center(f):
    cap = cv2.VideoCapture(f)
    ret = True
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.medianBlur(gray, 31)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 1000, param1 = 5, param2 = 5, minRadius =200, maxRadius = 480)
    circles = np.uint16(np.around(circles))
    
    avg = [circles[0,0,:]]
    
    n = 0
    while ret == True and n <10:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #    blur = cv2.GaussianBlur(gray, (5,5), 0)
        blur = cv2.medianBlur(gray, 31)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 1000, param1 = 1, param2 = 1, minRadius =300, maxRadius = 600)
        circles = np.uint16(np.around(circles))
        
        avg.append(circles[0,0,:])
        if len(avg) > 10:
            avg = avg[-10:]
        a = sum(avg)//len(avg)
    
        cv2.circle(frame, (a[0], a[1]), a[2], (0,255,0),3)
        cv2.circle(frame, (a[0], a[1]), 3, (0,0,255), 5)
        
    #    cv2.circle(frame, (circles[0][0][0], circles[0][0][1]), circles[0][0][2], (0,255,0),3)
    #    cv2.circle(frame, (circles[0][0][0], circles[0][0][1]), 3, (0,0,255), 10)
        cv2.imshow("Finding Well", frame)
        #cv2.imshow("Blurred", blur)
        n += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return a