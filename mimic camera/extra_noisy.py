import cv2
import numpy as np
import os


img = cv2.imread('datasets/noisy/input_image.png')
width = 440
height = 320

def gaussblured(img):
    if img is not None:
        blurred = cv2.GaussianBlur(img, (5, 5), 0.0)
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0.0)
        cv2.imwrite('datasets/noisy/random_block.png', blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Failed to load image file.')
    return blurred

def random_block(img, width, height, left_upper, right_bottom):
    num_blocks = 200

    img_size = (width, height)

    # Create a black image
    color_list = [16 * i for i in range(8, 17)]
    colors = []
    for i in color_list:
        colors.append((i, i, i))

    # Generate random block coordinates, shapes, and colors
    for i in range(num_blocks):
        x = np.random.randint(0, img_size[0])
        while x > left_upper[0] and x < right_bottom[0]:
            x = np.random.randint(0, img_size[0])
        y = np.random.randint(0, img_size[1])
        while y > left_upper[1] and y < right_bottom[0]:
            y = np.random.randint(0, img_size[1])
        #random color
        color = colors[np.random.choice([i for i in range(0, 9)])]
        
        #random shape
        shape = np.random.choice(['rectangle', 'triangle'])
        
        # random size
        size = np.random.randint(1, 9)
        
        if shape == 'rectangle':
            width = np.random.randint(1, size+1)
            height = np.random.randint(1, size+1)
            x1 = x - width//2
            y1 = y - height//2
            x2 = x + width//2
            y2 = y + height//2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
        elif shape == 'triangle':
            pts = np.array([[x, y-size//2], [x-size//2, y+size//2], [x+size//2, y+size//2]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img, [pts], color)
            
    cv2.imshow('Noisy Image', img)
    cv2.imwrite('datasets/noisy/random_block.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def for_every_image(folder_path):

    for item in os.listdir(folder_path):
        # get full path of item
        img_address = os.path.join(folder_path, item)

        img = cv2.imread(img_address)

        # check if image was loaded successfully
        if img is not None:
            # perform any desired actions on the image
            img = gaussblured(img)
            img = random_block(img, width, height)
            cv2.imshow('image', img)
            cv2.imwrite("haha", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Error: Image not loaded.')

for_every_image('noisy')

gaussblured(img)

