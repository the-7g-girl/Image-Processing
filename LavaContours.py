import cv2
import numpy as np
def threshold_image(image, lower_bound, upper_bound):
    return cv2.inRange(image, lower_bound, upper_bound)

def apply_morphology(mask):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_empty_mask(image):
    return np.zeros_like(image)

def draw_contours(mask, contours):
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

def apply_bitwise_and(original_image, mask):
    return cv2.bitwise_and(original_image, mask)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def find_connected_components(image):
    return cv2.connectedComponents(image)

def get_largest_connected_component(labeled_image):
    return np.where(labeled_image == np.argmax(np.bincount(labeled_image.flat)[1:]) + 1, 255, 0)

def merge_channels(component):
    return cv2.merge([component, component, component])
# Usage
def solution(image_path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for hue values corresponding to lava color
    lower_hue = np.array([0, 100, 100])   # Adjust the values based on the specific lava color
    upper_hue = np.array([16, 255, 255])

    # Threshold the image using the hue channel
    mask = threshold_image(hsv_image, lower_hue, upper_hue)

    # Additional morphological operations
    closing = apply_morphology(mask)

    # Find contours in the thresholded image
    contours = find_contours(closing)

    # Create an empty mask
    mask_result = create_empty_mask(image)

    # Draw the contours on the mask
    draw_contours(mask_result, contours)

    # Bitwise AND with the original image
    lava_region = apply_bitwise_and(image, mask_result)

    # Convert back to grayscale for connected components
    lava_region_gray = convert_to_gray(lava_region)

    # Find connected components
    num_labels, labeled_image = find_connected_components(lava_region_gray)

    # Get the largest connected component
    largest_component = get_largest_connected_component(labeled_image)

    # Merge channels for the final result
    image = merge_channels(largest_component)

    return image
