import cv2
import numpy as np


def get_dist(img, size):
    processed_img = preprocess_image(img, new_size=(size, size))
    centers = find_center_of_objects(processed_img)
    distances = find_distance_obj_center(processed_img, centers)

    if distances == []:
        return None

    return min(distances)


def find_center_of_objects(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    # Calculate moments for each contour
    for contour in contours:
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


def preprocess_image(img, new_size):
    img = cv2.resize(img, new_size)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    ## Slice the green
    imask = mask>0
    masked_img = np.zeros_like(img, np.uint8)
    masked_img[imask] = img[imask]

    grayscale_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.threshold(grayscale_img, 100, 255, cv2.THRESH_BINARY)[1]

    return binary_img


def find_distance_obj_center(img, centers):
    "finds distance from middle of object to center of image"
    distances = []
    _, width = img.shape
    x_center = width // 2 

    for center in centers:
        distance = abs(center[0] - x_center)
        distances.append(distance)

    return distances

