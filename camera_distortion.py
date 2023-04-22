import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# Define the dimensions of the rectangle
width = 540
height = 460

short_side = 120
long_side = 200

# Create a black image
img = np.zeros((height, width, 3), np.uint8)

# Draw the white rectangle on the black background
background_start_point = (120, 120)
background_end_point = (120 + short_side, 120 + long_side)


color = (255,255,255)
thickness = -1 # negative thickness fills the rectangle
original_rectangle = cv2.rectangle(img, background_start_point, background_end_point, color, thickness)

camera_position = (270, 230, 100)
camera_angle = (90, 90, 90)
rectangle_coordinates = [[120,120], [320, 240], [120, 240], [320, 120]]

def get_rotation_matrix(yaw, pitch, roll):
    cy = np.cos(np.radians(yaw))
    sy = np.sin(np.radians(yaw))
    cp = np.cos(np.radians(pitch))
    sp = np.sin(np.radians(pitch))
    cr = np.cos(np.radians(roll))
    sr = np.sin(np.radians(roll))

    R_yaw = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    R = R_yaw @ R_pitch @ R_roll

    return R

def distorded_cordinates(camera_position, camera_angles, rectangle_coordinates):
    R = get_rotation_matrix(camera_angles[0], camera_angles[1], camera_angles[2])
    
    distorted_coords = []
    for x, y in rectangle_coordinates:
        # 1. Translate image coordinates to camera coordinate system
        translated_coord = np.array([x - camera_position[0], y - camera_position[1], -camera_position[2]])

        # 2. Apply the rotation matrix to the translated coordinates
        rotated_coord = R @ translated_coord

        # 3. Project the rotated coordinates onto the image plane
        projected_x = focal_length * (rotated_coord[0] / rotated_coord[2]) + camera_position[0]
        projected_y = focal_length * (rotated_coord[1] / rotated_coord[2]) + camera_position[1]
        
        distorted_coords.append((projected_x, projected_y))

    return distorted_coords

rectangle_coordinates = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
camera_position = (x, y, z)
camera_angles = (yaw, pitch, roll)
focal_length = 1000  # You can adjust this value based on the camera's focal length

distorted_coords = distorded_cordinates(camera_position, camera_angles, rectangle_coordinates)
print(distorted_coords)

def image_distortion(original, distortion_matrix, width, height):
    return




'''
# Define the intrinsic camera matrix
camera_matrix = np.array([[1, 0, width/2],
                            [0, 1, height/2],
                            [0, 0, 1]], dtype= np.float32)

def camera_distortion_corner(original_image, focal_length, camera_position, camera_angle):
    # Define the camera position and orientation
    yaw = camera_angle[0]
    pitch = camera_angle[1]
    roll = camera_angle[2]
    camera_pos = np.array([camera_position[0], camera_position[1], 1], dtype=np.float64)  # (x, y, z)

    # Define the intrinsic camera matrix
    camera_matrix = np.array([[focal_length, 0, camera_position[0] - width / 2],
                              [0, focal_length, camera_position[1] - height / 2],
                              [0, 0, 1]], dtype= np.float32)

    # Define the extrinsic camera matrix
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    # Translate the image so that the center of rotation is at (0, 0)
    T1 = np.array([[1, 0, -camera_position[0]],
                   [0, 1, -camera_position[1]],
                   [0, 0, 1]], dtype=np.float32)

    # Translate the image back to its original position after rotation
    T2 = np.array([[1, 0, camera_position[0]],
                   [0, 1, camera_position[1]],
                   [0, 0, 1]], dtype=np.float32)

    # Define the extrinsic camera matrix
    extrinsic_matrix = np.dot(T2, np.dot(rotation_matrix, T1))
    result = cv2.warpPerspective(original_image, np.dot(camera_matrix, extrinsic_matrix)[:3,:3], (width, height))
    return result

warped_img = camera_distortion(img, 1, camera_position, camera_angle)




focal_length_list = np.arange(0.8, 1.2, 0.05)
camera_position_list = product(np.arange(width - 50, width + 50, 10), np.arange(height - 50, height + 50, 10))
camera_angle_list = product(np.arange(-0.1, 0.1, 0.05), np.arange(-0.1, 0.1, 0.05), np.arange(-20, 20, 10))


# create a new directory called "images"
if not os.path.exists('images'):
    os.makedirs('images')


for focal in focal_length_list:
    for position in camera_position_list:
        for angle in camera_angle_list:
            # save the image in the "images" directory
            cv2.imwrite(f'images/{focal} {position} {angle}.jpg', camera_distortion(original_rectangle, focal, position, angle))
'''


cv2.imshow('distorted image', warped_img) 

cv2.waitKey(0)
cv2.destroyAllWindows()