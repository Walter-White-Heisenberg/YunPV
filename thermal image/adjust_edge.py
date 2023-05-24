
import cv2
import numpy as np

def on_change(*args):
    pass
def find_thresholds(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Canny Edges')
    cv2.createTrackbar('Lower', 'Canny Edges', 0, 255, on_change)
    cv2.createTrackbar('Upper', 'Canny Edges', 0, 255, on_change)
    
    lower = 0
    upper = 0

    while True:
        if cv2.getWindowProperty('Canny Edges', cv2.WND_PROP_VISIBLE) >= 0:
            lower = cv2.getTrackbarPos('Lower', 'Canny Edges')
            upper = cv2.getTrackbarPos('Upper', 'Canny Edges')
        else:
            break

        edges = cv2.Canny(gray_image, lower, upper)
        cv2.imshow('Canny Edges', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return lower, upper

input_image_path = "exmpl-ir-imgs/DJI_0812_R.JPG"
lower_threshold, upper_threshold = find_thresholds(input_image_path)

print(f"Lower threshold: {lower_threshold}")
print(f"Upper threshold: {upper_threshold}")