import cv2
import numpy as np
import os
f_name = "Hevea_Brasilinsis"
directory = os.fsencode(f_name)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        
        img = cv2.imread(f_name+"/"+filename)
        ## (1) Convert to gray, and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        ## (2) Morph-op to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        ## (3) Find the max-area contour
        _, cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        ## (4) Crop and save it
        x,y,w,h = cv2.boundingRect(cnt)
        dst = img[y:y+h, x:x+w]
        cv2.imwrite(f_name+"/"+filename, dst)