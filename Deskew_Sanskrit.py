import cv2
import numpy as np

def correct_skew(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 255)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    _, thresh = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        text_contour = contours[0]

        rect = cv2.minAreaRect(text_contour)

        angle = rect[2]

        if angle > 45:
            angle -= 90
        rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1), (image.shape[1], image.shape[0]))

        return rotated

def solution(image_path):
    ############################
    ############################
    image = correct_skew(image_path)
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # image = cv2.imread(image_path)
    return image
