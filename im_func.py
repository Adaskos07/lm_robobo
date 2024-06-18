import cv2
import numpy as np


img = cv2.imread('test_img_2.png')
RESIZE_DIMENSIONS = (128, 128)


def find_center_of_objects(img):

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    if not contours:
        return None, None, img
    # Calculate moments for each contour
    for i, contour in enumerate(contours):
        # Calculate moments
        M = cv2.moments(contour)
        
        # Calculate centroid
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        centers.append((cX, cY))
        
    return centers

def preprocess_image():

    
    cv2.imshow('img', img)
    cv2.waitKey()
    img2 = cv2.resize(img, RESIZE_DIMENSIONS)


    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
    ## Slice the green
    imask = mask>0
    img3 = np.zeros_like(img2, np.uint8)
    img3[imask] = img2[imask]
    cv2.imshow('img3', img3)



    img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)



    img5 = cv2.threshold(img4, 100, 255, cv2.THRESH_BINARY)[1]


    cv2.imshow('Binary', img5)

    return img5, img2

def find_distance_obj_center(img, centers):
    "finds distance from middle of object to center of image"
    
    distances = []
    _,width = img.shape
    x_center = width // 2 
    for center in centers:
        distance = abs(center[0] - x_center)
        distances.append(distance)

    return distances


processed_img, resized_img = preprocess_image()
centers = find_center_of_objects(processed_img)
distances = find_distance_obj_center(processed_img, centers)

"""
for center in centers:
    cv2.circle(resized_img, (center), 5, (255, 0, 0), -1)
    cv2.imshow('Filled centers', resized_img)
    print(center)
    cv2.waitKey()"""

print(distances)
cv2.imshow('Filled centers', resized_img)
cv2.waitKey()
cv2.destroyAllWindows()



