# Egyptian ID Extraction

This project automates the extraction of text data from Egyptian ID cards. It involves preprocessing and aligning images, extracting specific regions of interest (ROIs), and performing Optical Character Recognition (OCR) to save the extracted data as text files.

## Project Description

The project aims to streamline the process of extracting essential information from Egyptian ID cards. The workflow includes image preprocessing, alignment, ROI extraction, and OCR using ArabicOCR and Tesseract OCR. The extracted data is saved in a structured format for further use.

## File Descriptions

- **main.py**: The main script that orchestrates the entire process. It clones the necessary repository, processes the image, aligns it, extracts ROIs, and performs OCR.

- **image_processing.py**: Contains functions for resizing images and extracting regions of interest (ROIs) from images.

- **image_preprocessing.py**: Includes functions for preprocessing images, such as applying adaptive thresholding.

- **image_alignment.py**: Provides functions for aligning images based on a reference image.

- **git_utils.py**: Contains utility functions for cloning a Git repository and changing directories.

- **ocr_extraction.py**: Defines functions for performing OCR on the extracted ROIs using ArabicOCR and Tesseract OCR.

## How to Run

1. **Clone the repository**:
    ```
    git clone https://github.com/yourusername/egyptian-id-extraction.git
    cd egyptian-id-extraction
    ```

2. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```

3. **Run the main script**:
    ```
    python main.py
    ```

## Output
- *Processed Images*
- *Extracted Text* 
## Requirements

- Python 3.x
- OpenCV
- Pytesseract
- ArabicOCR
- Git

## License

This project is licensed under the MIT License.
