from ArabicOcr import arabicocr
import os
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def extract_text_from_roi(roi_image_path, output_text_path):
    """
    Extract text from an ROI image using ArabicOCR and save the results.
    """
    print("Output text path:", output_text_path)
    
    # Ensure the directory for the output text file exists
    output_directory = os.path.dirname(output_text_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    # Ensure the output image filename has a valid file extension
    output_image_path = os.path.splitext(output_text_path)[0] + ".jpg"

    # Perform OCR on the ROI image and save the output image
    results = arabicocr.arabic_ocr(roi_image_path, output_image_path)
    words = [result[1] for result in results]

    # Write extracted text to the output text file
    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(words))

def extract_text_from_roi_with_tesseract(roi_image_path, output_text_path):
    """
    Extract text from an ROI image using Pytesseract and save the results.
    """
    print("Output text path:", output_text_path)

    # Ensure the directory for the output text file exists
    output_directory = os.path.dirname(output_text_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    output_image_path = os.path.splitext(output_text_path)[0] + ".jpg"

    # Perform OCR on the ROI image using Pytesseract
    extracted_text = pytesseract.image_to_string(Image.open(roi_image_path), lang='arabic_numbers')

    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write(extracted_text)

