# image_preprocessing.py

import cv2
import numpy as np

def resize_ara_num(img, width=1280, height=819):
    """
    Resize the image to the specified width and height.
    """
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def preprocess_image(image_path):
    """
    Read and preprocess the image.
    """
    image = cv2.imread(image_path)
    img = resize_ara_num(image)
    return img

def apply_adaptive_thresholding(img, block_size=21, C=10):
    """
    Apply adaptive thresholding to the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    return th

if __name__ == "__main__":
    image = preprocess_image('test.jpg')
