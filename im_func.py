import cv2
import numpy as np



def preprocess_image(img, new_size, green=True):
    img = cv2.resize(img, new_size)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
    if green:
        mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
    else:
        m1 = cv2.inRange(hsv, (0, 70, 50), (10, 255,255))
        m2 = cv2.inRange(hsv, (170, 70, 50), (180, 255,255))
        mask = m1 | m2

    ## Slice the green
    imask = mask>0
    masked_img = np.zeros_like(img, np.uint8)
    masked_img[imask] = img[imask]

    grayscale_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
    binary_img = cv2.threshold(grayscale_img, 70, 255, cv2.THRESH_BINARY)[1]
    return binary_img

img = cv2.imread('red.jpg')
# img = cv2.imread('test_img.jpg')
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

bimg = preprocess_image(img, (100, 100), green=False)
# bimg = preprocess_image(img, (100, 100), hsv_range=((170, 70, 50), (180, 255, 255)))
# bimg = preprocess_image(img, (48, 48), hsv_range=((36, 25, 25), (70, 255,255)))

cv2.imshow('After', bimg)
cv2.waitKey()
cv2.destroyAllWindows()




