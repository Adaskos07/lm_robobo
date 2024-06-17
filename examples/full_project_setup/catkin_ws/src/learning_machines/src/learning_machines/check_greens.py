def try_hsv(img,name):
     ## Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_lower_bound = (36, 25, 25)
    green_upper_bound = (70, 255,255)

    
    mask = cv2.inRange(hsv, green_lower_bound , green_upper_bound)

    ## Slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    #Calculate percantage of non-black e.g. green pixels
    height, width, _ = img.shape
    number_of_pixels = (height*width)
    green_pixels = np.sum(np.any(green != [0, 0, 0], axis=-1))
    print("Green image:", green)
    print("total_pixels", number_of_pixels)
    percentage = (green_pixels / number_of_pixels) * 100
    cv2.imwrite(str(FIGRURES_DIR / f"green_{name}.png"), green)
    return green_pixels, percentage

    
## Save 
    


def get_green_percentages(image):
    "Takes in an image, divides it into three blocks, left right and middle, and returns the percentage of pixels that are green in each block as a dict"
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Calculate the width of each block
    block_width = width // 3

    #Number of pixels in a block
    number_of_pixels = ((height*width) // 3)
    
    left = image[:, :block_width]
    middle = image[:, block_width:2*block_width]
    right = image[:, 2*block_width:]

    blocks = {
    "left_block" : left,
    "middle_block" : middle,
    "right_block" : right,
    }

    percentages ={}
    #Print out values, save images and store percentages in dict
    for name, img in blocks.items():
        green_count, green_percentage = try_hsv(img,name)
        print(f"Number of green pixels in {name}", green_count)
        print(f"Percent of green pixels in {name}", green_percentage)
        
        percentages[name] = green_percentage
        cv2.imwrite(str(FIGRURES_DIR / f"photo_{name}.png"), img)
    return percentages