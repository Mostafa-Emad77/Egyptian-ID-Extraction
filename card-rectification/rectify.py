"""
ID Card Rectification Software
Version: 2.0
Author: kxie
Email: xiekai@sundear.com

This module provides functionality for rectifying ID card images by detecting and correcting
perspective distortion. It uses both traditional computer vision techniques and deep learning
approaches for edge detection and corner point estimation.
"""

import os
import sys
import cv2
import torch
import imutils
import numpy as np
from os.path import join as pjoin
from skimage import exposure, img_as_ubyte
from imutils.perspective import four_point_transform
from itertools import combinations
from torchvision import transforms
from typing import Tuple, Optional, List
from load_model import load_model

# Configuration parameters
DEBUG = True
DEBUG_DIR = 'debug/'
PROCESS_SIZE = 1000
MODEL_INPUT_SIZE = 1000

def setup_debug_dir() -> None:
    """Create debug directory if debugging is enabled."""
    if DEBUG and not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)

def detect_edge(img: np.ndarray, name: str = '') -> np.ndarray:
    """
    Detect edges in the image using traditional computer vision techniques.
    
    Args:
        img (np.ndarray): Input image
        name (str): Name prefix for debug images
        
    Returns:
        np.ndarray: Edge detection result
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    mean_gray = cv2.mean(gray)
    TH_LIGHT = 150
    if mean_gray[0] > TH_LIGHT:
        gray = exposure.adjust_gamma(gray, gamma=6)
        gray = exposure.equalize_adapthist(gray, kernel_size=None, clip_limit=0.02)
        gray = img_as_ubyte(gray)

    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.medianBlur(closing, 5)
    blurred = cv2.bilateralFilter(blurred, d=0, sigmaColor=15, sigmaSpace=10)

    edged = cv2.Canny(blurred, 75, 200)

    if DEBUG:
        cv2.imwrite(pjoin(DEBUG_DIR, f"{name}_Cannyedge.png"), edged)

    return edged

def get_card_colormap() -> np.ndarray:
    """Get the color mapping for card segmentation."""
    return np.asarray([[0, 0, 0], [255, 255, 255]])

def decode_map(label_mask: np.ndarray) -> np.ndarray:
    """
    Decode the label mask into RGB image.
    
    Args:
        label_mask (np.ndarray): Input label mask
        
    Returns:
        np.ndarray: RGB image
    """
    label_colors = get_card_colormap()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    
    for ll in range(0, 2):
        r[label_mask == ll] = label_colors[ll, 0]
        g[label_mask == ll] = label_colors[ll, 1]
        b[label_mask == ll] = label_colors[ll, 2]
        
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb.astype(np.uint8)

def detect_edge_cnn(
    img: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    name: str = ''
) -> np.ndarray:
    """
    Detect edges using the CNN model.
    
    Args:
        img (np.ndarray): Input image
        model (torch.nn.Module): Loaded PyTorch model
        device (torch.device): Device to run inference on
        name (str): Name prefix for debug images
        
    Returns:
        np.ndarray: Edge detection result
    """
    image = cv2.resize(
        img,
        (MODEL_INPUT_SIZE, int(MODEL_INPUT_SIZE * img.shape[0] / img.shape[1])),
        interpolation=cv2.INTER_LINEAR
    )
    
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.196, 0.179, 0.323], [0.257, 0.257, 0.401])
    ])
    
    image = tf(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        img_val = image.to(device)
        res = model(img_val)
        pred = np.squeeze(res.data.max(1)[1].cpu().numpy())
        edged = decode_map(pred)
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
        edged = cv2.resize(
            edged,
            (PROCESS_SIZE, int(PROCESS_SIZE * img.shape[0] / img.shape[1])),
            interpolation=cv2.INTER_NEAREST
        )
        
        if DEBUG:
            cv2.imwrite(pjoin(DEBUG_DIR, f"{name}_CNNedge.png"), edged)

    return edged

def cross_point(line1: List[float], line2: List[float]) -> List[float]:
    """
    Calculate the intersection point of two lines.
    
    Args:
        line1 (List[float]): First line parameters [x1, y1, x2, y2]
        line2 (List[float]): Second line parameters [x1, y1, x2, y2]
        
    Returns:
        List[float]: Intersection point [x, y]
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        
    if k1 is None:
        if k2 is not None:
            x = x1
            y = k2 * x1 + b2
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    else:
        raise ValueError("Lines are parallel")

    return [x, y]

def get_angle(sta_point: np.ndarray, mid_point: np.ndarray, end_point: np.ndarray) -> float:
    """
    Calculate the angle between three points.
    
    Args:
        sta_point (np.ndarray): Starting point
        mid_point (np.ndarray): Middle point
        end_point (np.ndarray): End point
        
    Returns:
        float: Angle in degrees
    """
    ma_x = sta_point[0][0] - mid_point[0][0]
    ma_y = sta_point[0][1] - mid_point[0][1]
    mb_x = end_point[0][0] - mid_point[0][0]
    mb_y = end_point[0][1] - mid_point[0][1]
    ab_x = sta_point[0][0] - end_point[0][0]
    ab_y = sta_point[0][1] - end_point[0][1]
    
    ab_val2 = ab_x * ab_x + ab_y * ab_y
    ma_val2 = ma_x * ma_x + ma_y * ma_y
    mb_val2 = mb_x * mb_x + mb_y * mb_y
    
    cos_M = (ma_val2 + mb_val2 - ab_val2) / (2 * np.sqrt(ma_val2) * np.sqrt(mb_val2))
    angleAMB = np.arccos(cos_M) / np.pi * 180
    
    return angleAMB

def checked_valid_transform(approx: np.ndarray) -> bool:
    """
    Check if the detected corners form a valid quadrilateral.
    
    Args:
        approx (np.ndarray): Detected corner points
        
    Returns:
        bool: True if corners are valid
        
    Raises:
        ValueError: If corners are invalid
    """
    hull = cv2.convexHull(approx)
    TH_ANGLE = 45
    
    if len(hull) == 4:
        for i in range(4):
            p1 = hull[(i - 1) % 4]
            p2 = hull[i]
            p3 = hull[(i + 1) % 4]
            angle = get_angle(p1, p2, p3)
            if not (90 - TH_ANGLE < angle < 90 + TH_ANGLE):
                if DEBUG:
                    print("Detection Error: The detected corners could not form a valid quadrilateral for transformation.")
                raise ValueError("Corner points invalid.")
    else:
        if DEBUG:
            print("Detection Error: Could not find four corners from the detected edge.")
        raise ValueError("Corner points less than 4.")

    return True

def get_cnt(edged: np.ndarray, img: np.ndarray, ratio: float, name: str = '') -> np.ndarray:
    """
    Get the four corners of the ID card.
    
    Args:
        edged (np.ndarray): Edge detection result
        img (np.ndarray): Original image
        ratio (float): Scale ratio
        name (str): Name prefix for debug images
        
    Returns:
        np.ndarray: Four corner points
        
    Raises:
        ValueError: If corners cannot be detected
    """
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    mask = np.zeros((edged.shape[0], edged.shape[1]), np.uint8)
    mask[10:edged.shape[0] - 10, 10:edged.shape[1] - 10] = 1
    edged = edged * mask

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2(or_better=True) else cnts[1]
    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)
    
    edgelines = np.zeros(edged.shape, np.uint8)
    cNum = 4

    for i in range(min(cNum, len(cnts))):
        TH = 1 / 20.0
        if cv2.contourArea(cnts[i]) < TH * img.shape[0] * img.shape[1]:
            cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)
        else:
            cv2.drawContours(edgelines, [cnts[i]], 0, (1, 1, 1), -1)
            edgelines = edgelines * edged
            break
        cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)

    if DEBUG:
        cv2.imwrite(pjoin(DEBUG_DIR, f"{name}_edgelines.png"), edgelines)

    lines = cv2.HoughLines(edgelines, 1, np.pi / 180, 200)

    if lines is None or len(lines) < 4:
        if DEBUG:
            print("Detection Error: Could not find enough lines (must more than 4) from the detected edge.")
        raise ValueError("Lines not found.")

    if DEBUG:
        lines_draw = np.zeros((len(lines), 4), dtype=int)
        img_draw = img.copy()
        for i in range(0, len(lines)):
            rho, theta = lines[i][0][0], lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            lines_draw[i][0] = int(x0 + 1000 * (-b))
            lines_draw[i][1] = int(y0 + 1000 * (a))
            lines_draw[i][2] = int(x0 - 1000 * (-b))
            lines_draw[i][3] = int(y0 - 1000 * (a))
            cv2.line(img_draw, (lines_draw[i][0], lines_draw[i][1]), (lines_draw[i][2], lines_draw[i][3]), (0, 255, 0), 1)
        cv2.imwrite(pjoin(DEBUG_DIR, f"{name}_hough1.png"), img_draw)

    strong_lines = np.zeros([4, 1, 2])
    n2 = 0

    for n1 in range(0, len(lines)):
        if n2 == 4:
            break
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                c1 = np.isclose(abs(rho), abs(strong_lines[0:n2, 0, 0]), atol=80)
                c2 = np.isclose(np.pi - theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                c = np.all([c1, c2], axis=0)
                if any(c):
                    continue
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=40)
                closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and n2 < 4 and theta != 0:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1

    lines1 = np.zeros((len(strong_lines), 4), dtype=int)
    for i in range(0, len(strong_lines)):
        rho, theta = strong_lines[i][0][0], strong_lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        lines1[i][0] = int(x0 + 1000 * (-b))
        lines1[i][1] = int(y0 + 1000 * (a))
        lines1[i][2] = int(x0 - 1000 * (-b))
        lines1[i][3] = int(y0 - 1000 * (a))

        if DEBUG:
            cv2.line(img, (lines1[i][0], lines1[i][1]), (lines1[i][2], lines1[i][3]), (0, 255, 0), 3)

    approx = np.zeros((len(strong_lines), 1, 2), dtype=int)
    index = 0
    combs = list((combinations(lines1, 2)))
    
    for twoLines in combs:
        x1, y1, x2, y2 = twoLines[0]
        x3, y3, x4, y4 = twoLines[1]
        [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
        if 0 < x < img.shape[1] and 0 < y < img.shape[0] and index < 4:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 3)
            approx[index] = (int(x), int(y))
            index = index + 1

    if DEBUG:
        cv2.imwrite(pjoin(DEBUG_DIR, f"{name}_hough2.png"), img)

    if checked_valid_transform(approx):
        return approx * ratio

def set_corner(img: np.ndarray, r: int) -> np.ndarray:
    """
    Set rounded corners on the image.
    
    Args:
        img (np.ndarray): Input image
        r (int): Corner radius
        
    Returns:
        np.ndarray: Image with rounded corners
    """
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    row = img.shape[0]
    col = img.shape[1]

    for i in range(0, r):
        for j in range(0, r):
            if (r - i) * (r - i) + (r - j) * (r - j) > r * r:
                alpha_channel[i][j] = 0

    for i in range(0, r):
        for j in range(col - r, col):
            if (r - i) * (r - i) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                alpha_channel[i][j] = 0

    for i in range(row - r, row):
        for j in range(0, r):
            if (r - row + i + 1) * (r - row + i + 1) + (r - j) * (r - j) > r * r:
                alpha_channel[i][j] = 0

    for i in range(row - r, row):
        for j in range(col - r, col):
            if (r - row + i + 1) * (r - row + i + 1) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                alpha_channel[i][j] = 0

    img_bgra = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_bgra

def finetune(img: np.ndarray, ratio: float) -> np.ndarray:
    """
    Fine-tune the rectified image.
    
    Args:
        img (np.ndarray): Input image
        ratio (float): Scale ratio
        
    Returns:
        np.ndarray: Fine-tuned image
    """
    offset = int(2 * ratio)
    img = img[offset + 15:img.shape[0] - offset,
          int(offset * 2):img.shape[1] - int(offset * 2), :]
          
    if img.shape[0] < img.shape[1]:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 856 * 540)))
        r = int(img.shape[1] / 856 * 31.8)
    else:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 540 * 856)))
        r = int(img.shape[1] / 540 * 31.8)
        
    img = set_corner(img, r)
    
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
        
    return img

def inference(
    input_path: str,
    output_path: str,
    trained_model: torch.nn.Module,
    device: torch.device
) -> None:
    """
    Perform inference on a single image.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save output image
        trained_model (torch.nn.Module): Loaded PyTorch model
        device (torch.device): Device to run inference on
        
    Raises:
        ValueError: If input/output paths are invalid
        RuntimeError: If processing fails
    """
    # Validate input/output paths
    image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    if not os.path.isfile(input_path):
        raise ValueError(f"Input file not found: {input_path}")
    if os.path.splitext(input_path)[1] not in image_format:
        raise ValueError(f"Invalid input format. Supported formats: {', '.join(image_format)}")

    # Setup debug directory if needed
    setup_debug_dir()
    
    # Get base name for debug images
    name = os.path.splitext(os.path.basename(input_path))[0]

    try:
        # Read and preprocess image
        image = cv2.imread(input_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {input_path}")
            
        img = cv2.resize(image, (PROCESS_SIZE, int(PROCESS_SIZE * image.shape[0] / image.shape[1])))
        ratio = image.shape[1] / PROCESS_SIZE

        # Try CNN-based edge detection first
        try:
            if DEBUG:
                print("Edge Detection: trying CNN method...")
            edged = detect_edge_cnn(image, trained_model, device, name)
            corners = get_cnt(edged, img, ratio, name)
        except Exception as e:
            if DEBUG:
                print(f"CNN method failed: {str(e)}")
                print("Edge Detection: trying traditional method...")
            edged = detect_edge(img, name)
            corners = get_cnt(edged, img, ratio, name)

        # Transform and save result
        result = four_point_transform(image, corners.reshape(4, 2))
        result = finetune(result, ratio)
        cv2.imwrite(output_path, result)
        
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {str(e)}")

def inference_all(
    input_dir: str,
    output_dir: str,
    trained_model: torch.device,
    device: torch.device
) -> Tuple[int, int]:
    """
    Process all images in a directory.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save processed images
        trained_model (torch.nn.Module): Loaded PyTorch model
        device (torch.device): Device to run inference on
        
    Returns:
        Tuple[int, int]: Number of successful and total processed images
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    file_list = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1] in image_format]
    file_list.sort()
    
    total = len(file_list)
    successful = 0
    
    for i, file_name in enumerate(file_list, 1):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".png")
        
        try:
            inference(input_path, output_path, trained_model, device)
            successful += 1
            print(f"Success! Output saved in {os.path.abspath(output_path)}")
        except Exception as e:
            print(f"Failed to process {file_name}: {str(e)}")
            
    return successful, total

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rectify.py <input_path/dir> <output_path/dir>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        trained_model, device = load_model()
        
        if os.path.isfile(input_path):
            inference(input_path, output_path, trained_model, device)
        elif os.path.isdir(input_path) and os.path.isdir(output_path):
            successful, total = inference_all(input_path, output_path, trained_model, device)
            print(f"Done! {successful}/{total} images processed successfully.")
        else:
            print("Error: Invalid input or output path.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
