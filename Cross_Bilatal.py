import cv2
import numpy as np
def cross_bilateral_filter(image1, image2, d, sigma_color, sigma_space):
    # Split the images into color channels.
    channels1 = cv2.split(image1)
    channels2 = cv2.split(image2)

    # Apply the filter to each channel.
    filtered_channels = []
    for ch1, ch2 in zip(channels1, channels2):
        filtered_channels.append(cross_bilateral_filter_gray(ch1, ch2, d, sigma_color, sigma_space))

    # Merge the channels back into a color image.
    return cv2.merge(filtered_channels)

def cross_bilateral_filter_gray(image1, image2, d, sigma_color, sigma_space):
    filtered_image = np.zeros(image1.shape, dtype=np.float64)
    width, height = image1.shape[:2]
    r = d // 2

    # Pre-compute Gaussian distance weights.
    x, y = np.mgrid[-r:r+1, -r:r+1]
    gaussian_space = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))

    for i in range(r, width-r):
        for j in range(r, height-r):
            # Extract local region from both images.
            region1 = image1[i-r:i+r+1, j-r:j+r+1]
            region2 = image2[i-r:i+r+1, j-r:j+r+1]

            # Compute Gaussian intensity weights.
            gaussian_color = np.exp(-((region1 - image1[i, j])**2 + (region2 - image2[i, j])**2) / (2 * sigma_color**2))

            # Calculate bilateral filter response.
            weights = gaussian_color * gaussian_space
            weights /= np.sum(weights)
            filtered_image[i, j] = np.sum(region1 * weights)

    return filtered_image.astype(np.uint8)
def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    
    image = cv2.imread(image_path_b)
    image0 = cv2.imread(image_path_a)
    image1 = image0
    image2 = image0
    image3 = image

# Apply cross bilateral filter.
    image = cross_bilateral_filter(image1, image2, d=7, sigma_color=85, sigma_space=75)
    return image
