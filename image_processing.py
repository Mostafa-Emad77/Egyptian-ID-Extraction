import cv2

def resize_image(img, width, height):
    """
    Resize the input image.
    """
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img
    
def extract_roi(image, x1, y1, x2, y2):
    """
    Extracts a region of interest (ROI) from the input image.
    """
    # Extract the ROI using the provided coordinates
    roi = image[y1:y2, x1:x2]

    # Return the extracted ROI
    return roi
