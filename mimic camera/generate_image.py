import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import pi
import itertools
from coordinates_distortion import distorded_cordinates
import random
from skimage.measure import regionprops
import skimage.measure
import skimage.morphology
import os



folder_name_l = ["two_corner_out","one_corner_out","nightmare", "distortion+translation", "big_distortion_big_translation", "only_translation", "only_distortion", "test"]
def make_folder(dir_name):
    for i in folder_name_l:
        if not os.path.exists("datasets/Testset/" + i):
            print(folder_name_l)
            os.makedirs("datasets/Testset/" + i)


def draw_rectangle_with_grid(image, rectangle_coordinates, grid_size, color=(0, 0, 0), thickness=1):

    # Draw the rectangle
    cv2.fillPoly(image, [np.array(rectangle_coordinates).astype(int)], (255, 255, 255))

    # Draw horizontal lines
    for i in range(rectangle_coordinates[0][1], rectangle_coordinates[1][1], grid_size):
        cv2.line(image, (rectangle_coordinates[0][0], i), (rectangle_coordinates[2][0], i), color, thickness)

    # Draw vertical lines
    for j in range(rectangle_coordinates[0][0], rectangle_coordinates[2][0], grid_size):
        cv2.line(image, (j, rectangle_coordinates[0][1]), (j, rectangle_coordinates[1][1]), color, thickness)

    return image


# Define the dimensions of the rectangle
width = 440
height = 360

rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]
rectangle_coordinates_draw = [[120,120], [120, 240], [320, 240], [320, 120]]

def howmanYcorner(quad_coordinates):
    counter = 0
    for point in quad_coordinates:
        if point[0] <= 440 and point[0] >= 0 and point[1] <= 360 and point[1] >= 0:
            counter = counter + 1
    return counter
    

def draw_new_rectangle(quad_coordinates, folder_name, image_name, img):
    original_average_color = filter_by_average_color(img)[0]
    corner = howmanYcorner(quad_coordinates)
    # Reshape the coordinates to the required format for polylines
    quad_coordinates = np.array(quad_coordinates, dtype=np.int32).reshape((-1, 1, 2))
    M = cv2.getPerspectiveTransform(np.float32(rectangle_coordinates), np.float32(quad_coordinates))
    result = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    average_color = filter_by_average_color(result)[0]
    if corner == 4:
        new_image_path = f"datasets/Testset/{folder_name}/{image_name}.png"
    elif corner == 3:
        new_image_path = f"datasets/Testset/one_corner_out/{image_name}.png"
    elif average_color/original_average_color < 0.1 or average_color/original_average_color > 2:
        new_image_path = f"datasets/Testset/nightmare/{image_name}.png"
    else:
        new_image_path = f"datasets/Testset/two_corner_out/{image_name}.png"


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
    return input_list, "distortion+translation"


def only_translation():
    x_position = list(range(100, 350, 10))
    y_position = list(range(100, 220, 10))
    positions = list(itertools.product(x_position, y_position))
    print("@@@@@@")
    position_list = [[item[0], item[1], 100] for item in positions]
    inputs = list([item, [0, 0, 0]] for item in position_list)
    input_list = [[item[0], item[1], 100] for item in inputs]
    return input_list, "only_translation"

def only_distortion():
    yaw = list(range(0, 181, 10))
    pitch = list(range(0, 7, 1))
    roll = list(range(0, 7, 1))
    camera_angle_list = list(itertools.product(yaw, pitch, roll))
    inputs = list([[0, 0, 100], item] for item in camera_angle_list)
    input_list = [[item[0], item[1], 100] for item in inputs]
    return input_list, "only_distortion"

def test():
    input_list = []
    input_list.append()
    input_list.append("22131 0 0 100 180 3 6 100") 
    input_list.append("4243 120 36 36 20 -20 100 100")
    input_list.append("20435 160 100 100 180 14 6 100") 
    input_list.append("6765 190 100 100 180 0 0 100") 
    input_list.append("14843 190 190 100 0 14 0 100")
    input_list.append("35582 220 100 100 150 4 14 100")
    input_list.append("15273 220 220 100 0 4 4 100")
    return input_list, "test"



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

    return input_list, "big_distortion_big_translation"

# Calculate the mean of the pixel values in each color channel\
def filter_by_average_color(img):
    avg_color_per_row = np.mean(img, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return avg_color


def draw_many_rectangle(input_list, folder_name, case, img):
    rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]
    for input in input_list:
        # Define the coordinates of the quadrilateral
        quad_coordinates = np.array(distorded_cordinates(input[0], input[1], rectangle_coordinates), np.int32)
        draw_new_rectangle(quad_coordinates, folder_name, f"/{case}/{input[0][0]} {input[0][1]} {input[0][2]} {input[1][0]} {input[1][1]} {input[1][2]} {100}", img)


def random_color_distribution(original):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    image_color = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    image_color[original == 0] = [0, 0, 0]

    lmask = skimage.measure.label(original, background = 1)
    

    props = regionprops(lmask, intensity_image=original)
    image_color = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

    # Set your desired mean and standard deviation
    mean = 0.7
    std_dev = 0.25

    # Set the number of colors you want to generate
    num_colors = len(props)

    # Generate the list of random numbers
    rand_nums = np.random.normal(mean, std_dev, num_colors)

    # Ensure the numbers stay within the range [0, 1]
    rand_nums = np.clip(rand_nums, 0, 1)

    # Now, use these numbers to generate colors
    for i, prop in enumerate(props):
        # Skip background region
        if prop.label == 0 or i == 0:
            continue

        # Generate a color with varying whiteness
        color = (int(rand_nums[i] * 255), int(rand_nums[i] * 255), int(rand_nums[i] * 255))
        
        # Assign the color to the region
        image_color[lmask == prop.label] = color

    return image_color

for name in folder_name_l:
    make_folder(name)

original_img = np.zeros((height, width, 3), np.uint8)
white_rectangle = cv2.fillPoly(original_img.copy(), [np.array(rectangle_coordinates_draw).astype(int)], (255, 255, 255))
three_five = draw_rectangle_with_grid(original_img.copy(), rectangle_coordinates_draw, 40)
six_ten = draw_rectangle_with_grid(original_img.copy(), rectangle_coordinates_draw, 20)
twelve_twenty = draw_rectangle_with_grid(original_img.copy(), rectangle_coordinates_draw, 10)
the24_40 = draw_rectangle_with_grid(original_img.copy(), rectangle_coordinates_draw, 10)

colored1 = random_color_distribution(six_ten)
colored2 = random_color_distribution(six_ten)
colored3 = random_color_distribution(six_ten)
colored4 = random_color_distribution(six_ten)
colored5 = random_color_distribution(six_ten)
cv2.imshow('image window', colored1)
#original_img = cv2.fillPoly(original_img, [np.array([[120,120], [120, 240], [320, 240], [320, 120]]).astype(int)], (255, 255, 255))

original_img = draw_rectangle_with_grid(original_img, [[120,120], [120, 240], [320, 240], [320, 120]], 20)
original_img = random_color_distribution(original_img)

#original_img = draw_rectangle_with_grid(original_img, [[120,120], [120, 240], [320, 240], [320, 120]], 10)

#original_img = draw_rectangle_with_grid(original_img, [[120,120], [120, 240], [320, 240], [320, 120]], 5)


'''
distortion_and_translation()
big_distortion()
only_distortion()
only_translation()
'''


def generate_dataset(original_img, case):
    input_list, folder_name = only_translation()
    draw_many_rectangle(input_list, folder_name, case, original_img)
    input_list, folder_name = only_distortion()
    draw_many_rectangle(input_list, folder_name, case, original_img)
    input_list, folder_name = big_distortion()
    draw_many_rectangle(input_list, folder_name, case, original_img)
    input_list, folder_name = distortion_and_translation()
    draw_many_rectangle(input_list, folder_name, case, original_img)

'''
# "6 by 10/original", "6 by 10/color distribution/1", "12 by 20", "24 by 40", "original"
case = ''
generate_dataset(the24_40)
input_list, folder_name = only_translation()
draw_many_rectangle(input_list, folder_name, original_img)
input_list, folder_name = only_distortion()
draw_many_rectangle(input_list, folder_name, original_img)
input_list, folder_name = big_distortion()
draw_many_rectangle(input_list, folder_name, original_img)
input_list, folder_name = distortion_and_translation()
draw_many_rectangle(input_list, folder_name, original_img)
'''




'''
for i in range(0, 180):
    camera_position = (220, 180, 100)
    rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]
    camera_angle = (2, i, 2)
    results = np.array(distorded_cordinates(camera_position, camera_angle, rectangle_coordinates), np.int32)
    draw_new_rectangle(results, f"0 0 0 pitch_angle = {i}")

    '''
