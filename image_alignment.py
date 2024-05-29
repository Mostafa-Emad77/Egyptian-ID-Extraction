# image_alignment.py

import cv2
import numpy as np

def align_images(imgTest_path, imgRef_path, output_path):
    """
    Align imgTest with imgRef and save the result.
    """
    imgTest = cv2.imread(imgTest_path)
    imgRef = cv2.imread(imgRef_path)

    # Convert images to grayscale
    imgTest_grey = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    imgRef_grey = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
    height, width = imgRef_grey.shape

    # Create an AKAZE detector
    akaze_detector = cv2.AKAZE_create()

    # Detect keypoints and compute descriptors for both images
    keyPoint1, des1 = akaze_detector.detectAndCompute(imgTest_grey, None)
    keyPoint2, des2 = akaze_detector.detectAndCompute(imgRef_grey, None)

    des1_float = des1.astype(np.float32)
    des2_float = des2.astype(np.float32)

    # Match features between two images using FlannBasedMatcher
    flann_matcher = cv2.FlannBasedMatcher()
    matches = flann_matcher.match(des1_float, des2_float)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = keyPoint1[matches[i].queryIdx].pt
        p2[i, :] = keyPoint2[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    aligned_img = cv2.warpPerspective(imgTest, homography, (width, height))
    cv2.imwrite(output_path, aligned_img)
if __name__ == "__main__":
    align_images('test.jpg', 'ref.jpg', 'aligned_test.jpg')
