# hitam.py
# buang background putih
# pakai adaptive thresholding
import cv2
import numpy as np

img = cv2.imread("yangdicoba.jpg")

## (1) Convert to gray, and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# th, threshed = cv2.threshold(gray, 163, 255, cv2.THRESH_BINARY_INV)
threshed2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,251,6)

kernel = np.ones((8,8), np.uint8)

dilation = cv2.dilate(threshed2,kernel,iterations = 10)
inverted = cv2.bitwise_not(dilation)
backtorgb = cv2.cvtColor(inverted,cv2.COLOR_GRAY2RGB)

hasil = cv2.subtract(img,backtorgb)
cv2.imwrite("clos.jpg",hasil)
