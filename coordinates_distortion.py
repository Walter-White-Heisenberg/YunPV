import numpy as np
import cv2
import os
import smtplib
from email.message import EmailMessage
from email.mime.image import MIMEImage

folder_name = "images"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def send_email_with_image(sender_email, receiver_email, subject, body, image_path):
    # Create the EmailMessage object
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(body)

    # Set the content type to "multipart/mixed"
    msg.make_mixed()

    # Read the image and attach it to the email
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        image = MIMEImage(img_data)
        msg.attach(image)

    # Send the email using SMTP
    with smtplib.SMTP_SSL('smtp.126.com', 465) as server:
        server.login(sender_email, '123456')
        server.send_message(msg)
        print(f"Email sent to {receiver_email}.")

# Set your email details and call the function
sender_email = "liyun8185@126.com"
receiver_email = "liyun8185@126.com"
subject = "Image attachment"
body = "Please find the attached image."
image_path = "images/input_image.png"

width = 440
height = 360

short_side = 120
long_side = 200

camera_position = (220, 180, 1000)
camera_angle = (0, 0, 0)
rectangle_coordinates = [[120,120], [120, 240], [320, 120], [320, 240]]


# Create a black image
img = np.zeros((height, width, 3), np.uint8)

# Draw the white rectangle on the black background
background_start_point = (120, 120)
background_end_point = (120 + long_side, 120 + short_side)


color = (255,255,255)
thickness = -1 # negative thickness fills the rectangle
original_rectangle = cv2.rectangle(img, background_start_point, background_end_point, color, thickness)

cv2.imwrite(image_path, img)


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

    focal_length = 1000

    for x, y in rectangle_coordinates:
        # 1. Translate image coordinates to camera coordinate system
        translated_coord = np.array([x - camera_position[0], y - camera_position[1], -camera_position[2]])

        # 2. Apply the rotation matrix to the translated coordinates
        rotated_coord = R @ translated_coord

        # 3. Project the rotated coordinates onto the image plane
        projected_x = focal_length * (rotated_coord[0] / rotated_coord[2]) + camera_position[0]
        projected_y = focal_length * (rotated_coord[1] / rotated_coord[2]) + camera_position[1]
        
        distorted_coords.append((projected_x, projected_y))

        print(distorted_coords)

    return distorted_coords



# Create a black image
img = np.zeros((height, width, 3), np.uint8)

# Define the coordinates of the quadrilateral
quad_coordinates = np.array(distorded_cordinates(camera_position, camera_angle, rectangle_coordinates), np.int32)

# Reshape the coordinates to the required format for polylines
quad_coordinates = quad_coordinates.reshape((-1, 1, 2))

index1 = 0
index2 = 1
m = quad_coordinates[index1].copy()


# Swap the items
quad_coordinates[index1] = quad_coordinates[index2]

quad_coordinates[index2] = m
print(quad_coordinates)


# Draw the white quadrilateral on the black background
color = (255, 255, 255)
thickness = -1 # negative thickness fills the quadrilateral
is_closed = True

result = cv2.fillPoly(img, [quad_coordinates], color)

new_image_path = "images/output.png"

cv2.imwrite(new_image_path, result)

cv2.waitKey(0)
cv2.destroyAllWindows()

