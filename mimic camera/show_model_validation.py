from tensorflow.keras.models import Sequential, load_model
from generate_image import draw_new_rectangle
from coordinates_distortion import distorded_cordinates
import numpy as np
import os
import cv2

width = 320
height = 440

rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]

# Create a black image
img = np.zeros((height, width, 3), np.uint8)

# Draw the white rectangle on the black background, 100:60
background_start_point = (170,170)#(120, 220)
background_end_point = (370, 290)#(320, 240)
color = (255,255,255)
thickness = -1 # negative thickness fills the rectangle
original_rectangle = cv2.rectangle(img, background_start_point, background_end_point, color, thickness)

def column_to_array(df):
    first_list = []
    for x in df.columns:
        first_list = x[0]
    return first_list

def subtract(path1, path2):
    # Load two images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    diff = img1.astype(np.int16) - img2.astype(np.int16)

    # Create masks for negative and positive differences
    negative_mask = diff < 0
    positive_mask = diff > 0
    zero_mask = diff == 0

    # Define color maps for negative (blue), positive (red), and zero (black) differences
    blue_map = np.zeros((256, 1, 3), dtype=np.uint8)
    blue_map[:, 0, 0] = np.arange(256)

    red_map = np.zeros((256, 1, 3), dtype=np.uint8)
    red_map[:, 0, 2] = np.arange(256)

    black_map = np.zeros((256, 1, 3), dtype=np.uint8)

    # Apply the color maps to the negative, positive, and zero differences separately
    diff_negative = cv2.normalize(np.abs(diff * negative_mask), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    diff_positive = cv2.normalize(diff * positive_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    diff_color_negative = cv2.applyColorMap(diff_negative, blue_map)
    diff_color_positive = cv2.applyColorMap(diff_positive, red_map)
    diff_color_zero = diff.astype(np.uint8) * zero_mask

    # Merge the color mapped negative, positive, and zero differences
    diff_color = cv2.add(diff_color_negative, diff_color_positive)
    diff_color = cv2.add(diff_color, diff_color_zero)

    # Display the resulting image
    cv2.imshow('Difference', diff_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return diff_color
'''
image_info_list = []
for file_name in os.listdir("datasets/validation_set"):
    img = cv2.imread(os.path.join("datasets/validation_set", file_name))
    img_dict = {"name": file_name, "img": img}
    image_info_list.append(img_dict)

loaded_model = load_model('loss.h5')

for im_inf in image_info_list:
    print(im_inf["img"].shape)
    img = im_inf["img"]
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output = loaded_model.predict(img)[0]
    new_coordinates = distorded_cordinates([output[1],output[2],output[3]], [output[4],output[5],output[6]], rectangle_coordinates)
    print(new_coordinates)
    draw_new_rectangle(new_coordinates, "datasets/validation_set/reverse" + im_inf["name"])
    result = subtract("datasets/validation_set/" + im_inf["name"], "datasets/validation_set/reverse" + im_inf["name"])
    cv2.imwrite("datasets/validation_set/subtract" + im_inf["name"], img)
'''

name_list = ["distortion", "distortion+translation", "one corner out", "translation", "two corner out"]
image_info_list = []
for file_name in name_list:
    result = subtract("datasets/validation_set/" + file_name + ".png", "datasets/validation_set/reverse " + file_name + ".png")
    cv2.imwrite("datasets/validation_set/subtract" + file_name + ".png", result)
#path1 = 'C:/Users/Yun Li/Downloads/pvmod-main/pvmod-main/YUn/github/YunPV/mimic camera/datasets/validation_set/'


#path2 = 'C:/Users/Yun Li/Downloads/pvmod-main/pvmod-main/YUn/github/YunPV/mimic camera/datasets/validation_set/pitch_angle = 0.png'

#subtract(path1, path2)


    