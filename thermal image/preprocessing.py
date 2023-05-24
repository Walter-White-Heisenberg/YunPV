import cv2
import numpy as np
import os

def process_image(path, output_folder, lower, upper):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, lower, upper)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    print(lines)

    # Crop cells based on line segments
    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        cell = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        cv2.imwrite(os.path.join(output_folder, f"cell_{idx}.png"), cell)

    return len(cell)

rho_resolution = 1
theta_resolution = np.pi / 180
threshold = 100
min_area = 0
max_area = 1000000

max_area = 0
best_lower = 0
best_higher = 0

for lower in range(100,255):
    for higher in range(lower,255):
        num_area = process_image("exmpl-ir-imgs/DJI_0812_R.JPG", "exmpl-ir-imgs/after-preprocessing", lower, higher)
        if num_area is not None and num_area > max_area:
            print(max_area, best_higher, best_lower)
            print((max_area, best_higher, best_lower))
            max_area = num_area
            best_lower = lower
            best_higher = higher

print(max_area, best_higher, best_lower)

if not os.path.exists("exmpl-ir-imgs/after-preprocessing"):
    os.makedirs("exmpl-ir-imgs/after-preprocessing")

input_image_path = "exmpl-ir-imgs/DJI_0812_R.JPG"
output_cells_folder = "exmpl-ir-imgs/after-preprocessing"

process_image(input_image_path, output_cells_folder, best_higher, best_lower)