import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from coordinates_distortion import distorded_cordinates
from generate_image import draw_rectangle_with_grid
import numpy as np
import os
import cv2
import pickle
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
case_name = "6 by 10/original"
PATH = "datasets/Testset/" + case_name +'/checkpoint_epoch_0.pth'
original_path = "datasets/validation_set/originally generated/"

with open('{}/{}'.format("datasets/Testset/" + case_name, "params"), 'rb') as file:
    scaler = pickle.load(file)


validation_set_dict = {
    "one corner out1": [250, 250, 100, 30, 2, 8, 100],
    "one corner out2": [160, 130, 100, 30, 6, 12, 100],
    "two corner out1": [160, 100, 100, 90, 0, 0, 100],
    "two corner out2": [190, 130, 100, 60, 14, 14, 100],
    "distortion+translation1": [250, 250, 100, 180, 2, 6, 100],
    "distortion+translation2": [160, 190, 100, 120, 2, 16, 100],
    "nightmare1": [120, 36, 27, 420, 340, 100, 100],
    "nightmare2": [120, 36, 36, 20, 340, 100, 100],
    "distortion1": [0, 0, 100, 160, 5, 0, 100],
    "distortion2": [0, 0, 100, 180, 0, 6, 100],
    "translation1": [180, 210, 100, 0, 0, 0, 100],
    "translation2": [240, 130, 100, 0, 0, 0, 100],
    "original image": [220, 180, 100, 0, 0, 0, 100]
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(128000, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def load_model(model_path):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


width = 440
height = 360

rectangle_coordinates = [[120,120], [120, 240], [320, 240], [320, 120]]
original_img = np.zeros((height, width, 3), np.uint8)
white_rectangle = cv2.fillPoly(original_img.copy(), [np.array(rectangle_coordinates).astype(int)], (255, 255, 255))
#six_ten = draw_rectangle_with_grid(original_img.copy(), [[120,120], [120, 240], [320, 240], [320, 120]], 20)

for key, value in validation_set_dict.items():
    quad_coordinates = np.array(distorded_cordinates([value[0],value[1],value[2]], [value[3],value[4],value[5]], rectangle_coordinates))
    quad_coordinates = np.array(quad_coordinates, dtype=np.int32).reshape((-1, 1, 2))
    M = cv2.getPerspectiveTransform(np.float32(rectangle_coordinates), np.float32(quad_coordinates))
    result = cv2.warpPerspective(white_rectangle, M, (white_rectangle.shape[1], white_rectangle.shape[0]))
    cv2.imwrite(original_path + key +".png", result)


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

    return diff_color

def calculate_area_accuracy(image, original):
    
    original = img1 = cv2.imread(original)

    blue_mask = (image[:, :, 0] == 255)  
    red_mask = (image[:, :, 2] == 255) 
    white_mask = (original[:, :, 0] == 255) & (original[:, :, 1] == 255) & (original[:, :, 2] == 255)  # White pixels in the image


    blue = np.sum(blue_mask)
    red = np.sum(red_mask)
    white = np.sum(white_mask)

    return (blue + red) / white

image_info_list = []
for file_name in os.listdir("datasets/validation_set/originally generated"):
    img = cv2.imread(os.path.join("datasets/validation_set/originally generated", file_name))
    img_dict = {"name": file_name, "img": img}
    print(file_name)
    image_info_list.append(img_dict)

loaded_model = load_model(PATH)   # Replace 'best_model.pth' with the actual path of your PyTorch model

for im_inf in image_info_list:
    img = im_inf["img"]
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    #img_tensor = transforms.ToTensor()(img).to(device)

    scaled_output = loaded_model(img_tensor).detach().numpy()
    print(scaled_output)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    output = scaler.inverse_transform(scaled_output)[0]
    print(output)
    new_coordinates = distorded_cordinates([output[0].item(),output[1].item(),output[2].item()], [output[3].item(),output[4].item(),output[5].item()], rectangle_coordinates)
    print(new_coordinates)
    print(validation_set_dict[im_inf['name'].split(".png")[0]])
    #draw_new_rectangle(new_coordinates, "datasets/validation_set/reverse" + im_inf["name"])
    reverse = cv2.fillPoly(original_img.copy(), [np.array(new_coordinates).astype(int)], (255, 255, 255))
    cv2.imwrite("datasets/validation_set/reverse/" + im_inf["name"], reverse)
    result = subtract(original_path + im_inf["name"], "datasets/validation_set/reverse/" + im_inf["name"])
    cv2.imwrite("datasets/validation_set/subtraction/" + im_inf["name"], result)
'''    
name_list = ["distortion", "distortion+translation", "one corner out", "translation", "two corner out"]
image_info_list = []
for file_name in name_list:
    print(file_name)
    path1 = "datasets/validation_set/originally generated/" + file_name + ".png"
    path2 = "datasets/validation_set/reverse/" + file_name + ".png"
    result = subtract(path1, path2)
    cv2.imwrite("datasets/validation_set/subtraction/_" + file_name + str(calculate_area_accuracy(result, path1)) + ".png", result)
    cv2.imwrite("datasets/validation_set/subtraction/_" + file_name + str(calculate_area_accuracy(result, path1)) + ".png", result)
'''


    