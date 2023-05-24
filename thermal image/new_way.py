import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from PIL import Image
from skimage.measure import regionprops
import skimage.measure
import skimage.morphology
import cv2
import math
import hdbscan
import os

rootDir = 'exmpl-ir-imgs/'




def masking(clean_mask):
    threshold = skimage.filters.threshold_otsu(clean_mask)

    print("111")

    image_mask = np.ones(clean_mask.shape)

    idx = np.where(clean_mask < threshold)

    image_mask[idx] = 0

    return image_mask


def label_before_crop(image, laplacian_ksize, binary_closing_ksize):
    
    '''
    cv2.imshow('Original', image)
    cv2.waitKey(0)


    for i in [3,5,7,9,11,13,15,17,19]:
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=i)
        # Display the original image and the result
        cv2.imshow(f'Laplacian with ksize {i}', laplacian)
        cv2.waitKey(0)
    '''

    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=laplacian_ksize)

    '''
    def scharring(image):
        # or apply schar filter
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)  # x-direction
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)  # y-direction
        schar = cv2.magnitude(scharrx, scharry)
        return cv2.normalize(schar, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    scharr = scharring(image)

    def sobeling(image, KS):
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=KS)  # x-direction
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=KS)  # y-direction
        sobel = cv2.magnitude(sobelx, sobely)
        return cv2.normalize(sobel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    sobel = sobeling(image, 3)
    '''



    '''
    cv2.imshow('schar', scharr)
    cv2.imshow('sobel', sobel)
    cv2.waitKey(0)
    cv2.imshow('Laplacian after sobel', cv2.Laplacian(sobel, cv2.CV_64F, ksize=9))
    cv2.imshow('Laplacian after schar', cv2.Laplacian(scharr, cv2.CV_64F, ksize=9))
    cv2.waitKey(0)
    '''



    # Clean the binary mask using morphological operations (e.g., opening)
    clean_mask = skimage.morphology.closing(laplacian, skimage.morphology.square(1))
    #cv2.imshow('clean_mask', clean_mask)
    #cv2.waitKey(0)

    '''
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.imshow(clean_mask, cmap='bone', alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    '''



    image_mask = masking(clean_mask)

    '''
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.imshow(image_mask, cmap='bone', alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    '''

    image_mask = skimage.morphology.binary_closing(image_mask, skimage.morphology.square(binary_closing_ksize))

    '''
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.imshow(image_mask, cmap='bone', alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    '''

    lmask = skimage.measure.label(image_mask, background = 1)
    '''
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.imshow(lmask, cmap='jet', alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    '''
    return lmask


def crop_and_record(image, lmask, widths, heights, ratios):
    # label each closed part, get properties of each labeled region
    props = regionprops(lmask, intensity_image=image)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for i, prop in enumerate(props):
        cy, cx = prop.centroid
        cv2.putText(image_color, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        minr, minc, maxr, maxc = prop.bbox
        cropped_img = image[minr:maxr, minc:maxc]
        '''
        average = np.mean(prop.intensity_image[prop.image])
        SD = np.std(prop.intensity_image[prop.image])
        if  60 <  maxc - minc and maxc - minc < 80 and 31 < maxr - minr and maxr - minr < 37:
            cv2.imwrite(f"Cropped/good/{fname}_{cy}_{cx}_{average}_{SD}.jpg", cropped_img)
        else:
            cv2.imwrite(f"Cropped/abnormal/{fname}_{cy}_{cx}_{average}_{SD}.jpg", cropped_img)
        '''
        width = maxc - minc
        height = maxr - minr
        ratio = width / height if height != 0 else 0

        widths.append(width)
        heights.append(height)
        ratios.append(ratio)

        print(f"Area of cropped_img_{i}: {prop.area} pixels, width: {width}, height: {height}, ratio: {ratio}")
        
    '''
    cv2.imshow('Annotated image', image_color)
    cv2.waitKey(0)
    '''


for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    widths = []
    heights = []
    ratios = []
    for fname in fileList:
        print('\t%s' % fname)
        image = cv2.imread(os.path.join(dirName, fname), cv2.IMREAD_GRAYSCALE)
        
        lmask = label_before_crop(image, 7, 23)
        crop_and_record(image, lmask, widths, heights, ratios)


# Draw histograms for width, height and ratio
plt.figure()
counts, bins, patches = plt.hist(widths, bins=20)
for count, bin in zip(counts, bins):
    if count > 0:
        plt.text(bin, count, str(int(count)))
plt.title("Width Distribution")
plt.xlabel("Width")
plt.ylabel("Frequency")
plt.show()

plt.figure()
counts, bins, patches = plt.hist(heights, bins=20)
for count, bin in zip(counts, bins):
    if count > 0:
        plt.text(bin, count, str(int(count)))
plt.title("Height Distribution")
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.show()

plt.figure()
counts, bins, patches = plt.hist(ratios, bins=20)
for count, bin in zip(counts, bins):
    if count > 0:
        plt.text(bin, count, str(int(count)))
plt.title("Ratio Distribution")
plt.xlabel("Ratio")
plt.ylabel("Frequency")
plt.show()


# For widths
width_mean = np.mean(widths)
width_median = np.median(widths)
width_sd = np.std(widths)

# For heights
height_mean = np.mean(heights)
height_median = np.median(heights)
height_sd = np.std(heights)

# For ratios
ratio_mean = np.mean(ratios)
ratio_median = np.median(ratios)
ratio_sd = np.std(ratios)

print("Widths - Mean: ", width_mean, ", Median: ", width_median, ", SD: ", width_sd)
print("Heights - Mean: ", height_mean, ", Median: ", height_median, ", SD: ", height_sd)
print("Ratios - Mean: ", ratio_mean, ", Median: ", ratio_median, ", SD: ", ratio_sd)