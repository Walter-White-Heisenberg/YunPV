import numpy as np
import cv2
import os
from math import pi
import itertools
from coordinates_distortion import distorded_cordinates
import random



# Define the dimensions of the rectangle
width = 440
height = 320

rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]


def draw_new_rectangle(quad_coordinates, image_name):
    # Create a black image
    img = np.zeros((height, width, 3), np.uint8)
    # Reshape the coordinates to the required format for polylines
    quad_coordinates = np.array(quad_coordinates, dtype=np.int32).reshape((-1, 1, 2))

    index1 = 0
    index2 = 1
    m = quad_coordinates[index1].copy()


    # Swap the items
    quad_coordinates[index1] = quad_coordinates[index2]
    quad_coordinates[index2] = m
    ###########print(quad_coordinates)


    # Draw the white quadrilateral on the black background
    color = (255, 255, 255)
    thickness = -1 # negative thickness fills the quadrilateral
    is_closed = True

    result = cv2.fillPoly(img, [quad_coordinates], color)
    
    average_color = filter_by_average_color(result)[0]

    if average_color > 42.5:
        new_image_path = f"datasets/Testset/{image_name}.png"
    elif average_color > 35:
        new_image_path = f"datasets/Testset/one_corner_out/{image_name}.png"
    elif average_color > 26:
        new_image_path = f"datasets/Testset/two_corner_out/{image_name}.png"
    else:
        new_image_path = f"datasets/Testset/nightmare/{image_name}.png"
    
    cv2.imwrite(new_image_path, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def distortion_and_translation():
    yaw = list(range(0, 181, 30))
    pitch = list(range(0, 18, 2))
    roll = list(range(0, 18, 2))
    camera_angle_list = list(itertools.product(yaw, pitch, roll))

    x_position = list(range(160, 280, 30))
    y_position = list(range(100, 280, 30))
    positions = list(itertools.product(x_position, y_position))

    # add the constant to each combination using a list comprehension
    position_list = [[item[0], item[1], 100] for item in positions]

    inputs = list(itertools.product(position_list, camera_angle_list,))
    input_list = [[item[0], item[1], 100] for item in inputs]

    # Randomly select 8000 elements without replacement
    input_list = random.sample(input_list, 8000)
    return input_list

def generate_image(input_list):
    rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]
    for input in input_list:
        result = np.array(distorded_cordinates(input[1], input[0], rectangle_coordinates), np.int32)
        draw_new_rectangle(result, f"{input[1][0]} {input[1][1]} {input[1][2]}   {input[0][0]} {input[0][1]} {input[0][2]} {100}")

def only_translation():
    x_position = list(range(100, 350, 10))
    y_position = list(range(100, 220, 10))
    positions = list(itertools.product(x_position, y_position))
    print(positions[0])
    position_list = [[item[0], item[1], 100] for item in positions]
    print(position_list[0])
    inputs = list([item, [0, 0, 0]] for item in position_list)
    print(inputs[0])
    input_list = [[item[0], item[1], 100] for item in inputs]
    return input_list

def only_distortion():
    yaw = list(range(0, 181, 10))
    pitch = list(range(0, 7, 1))
    roll = list(range(0, 7, 1))
    camera_angle_list = list(itertools.product(yaw, pitch, roll))
    inputs = list([[0, 0, 100], item] for item in camera_angle_list)
    input_list = [[item[0], item[1], 100] for item in inputs]
    return input_list


def big_distortion():
    yaw = list(range(0, 180, 30))
    pitch = list(range(0, 45, 9))
    roll = list(range(0, 45, 9))
    camera_angle_list = list(itertools.product(yaw, pitch, roll))

    x_position = list(range(-100, 60, 40)) + list(range(380, 480, 40))
    y_position = list(range(-100, 60, 40)) + list(range(260, 360, 40))
    positions = list(itertools.product(x_position, y_position))

    # add the constant to each combination using a list comprehension
    position_list = [[item[0], item[1], 100] for item in positions]

    inputs = list(itertools.product(camera_angle_list, position_list))
    input_list = [[item[0], item[1], 100] for item in inputs]

    return input_list

# Calculate the mean of the pixel values in each color channel\
def filter_by_average_color(img):
    avg_color_per_row = np.mean(img, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return avg_color


def draw_many_rectangle(input_list):
    rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]
    for input in input_list:
        # Define the coordinates of the quadrilateral
        quad_coordinates = np.array(distorded_cordinates(input[0], input[1], rectangle_coordinates), np.int32)
        draw_new_rectangle(quad_coordinates, f"{input[0][0]} {input[0][1]} {input[0][2]} {input[1][0]} {input[1][1]} {input[1][2]} {100}")

#draw_many_rectangle(only_translation())
#draw_many_rectangle(only_distortion())
#draw_many_rectangle(distortion_and_translation())
#distor
#draw_many_rectangle(big_distortion())




'''
for i in range(0, 180):
    camera_position = (220, 180, 100)
    rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]
    camera_angle = (2, i, 2)
    results = np.array(distorded_cordinates(camera_position, camera_angle, rectangle_coordinates), np.int32)
    draw_new_rectangle(results, f"0 0 0 pitch_angle = {i}")

    '''
