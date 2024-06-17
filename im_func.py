import cv2
import numpy as np


img = cv2.imread('test_img.jpg')
cv2.imshow('img', img)
cv2.waitKey()

img2 = cv2.resize(img, (32, 32))
cv2.imshow('img2', img2)
cv2.waitKey()

hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
## Slice the green
imask = mask>0
img3 = np.zeros_like(img2, np.uint8)
img3[imask] = img2[imask]
cv2.imshow('img3', img3)
print(img3.shape)


img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', img4)
print(img4.shape)

img5 = cv2.threshold(img4, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('Binary', img5)

cv2.waitKey()
cv2.destroyAllWindows()
