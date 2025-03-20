# Egyptian ID Extraction

This project automates the extraction of text data from Egyptian ID cards. It involves preprocessing images, extracting specific regions of interest (ROIs), and performing Optical Character Recognition (OCR) to save the extracted data as text files.

## Project Description

The Egyptian ID Extraction project aims to streamline the process of extracting essential information from Egyptian ID cards. The workflow includes:

- **Image Preprocessing**: Enhancing the quality of the input images to improve OCR accuracy. This includes resizing, applying adaptive thresholding, and ensuring that the images are suitable for further processing.

- **Region of Interest (ROI) Extraction**: Identifying and extracting specific areas of the ID card, such as the first name, last name, address, ID number, and face image. This is crucial for isolating the text that needs to be recognized.

- **Optical Character Recognition (OCR)**: Utilizing OCR technologies, including ArabicOCR and Tesseract OCR, to convert the extracted image regions into machine-readable text. The extracted data is then saved in a structured format (JSON) for further use.

- **Error Handling**: The application includes robust error handling to inform users if the image quality is insufficient for rectification. Users will receive clear messages indicating that the image must be of good quality and lighting, similar to the provided example.

## File Descriptions

- **main.py**: The main script for command-line execution that orchestrates the entire process. It processes the image, extracts ROIs, and performs OCR. The script requires the `card-rectification` repository to be present in the project directory.

- **gui.py**: The graphical user interface (GUI) for the application, allowing users to upload images, view previews, and display extracted information in a user-friendly manner.

- **image_processing.py**: Contains functions for resizing images and extracting regions of interest (ROIs) from images.

- **image_preprocessing.py**: Includes functions for preprocessing images, such as applying adaptive thresholding.

- **git_utils.py**: Contains utility functions for changing directories. The cloning functionality has been removed.

- **ocr_extraction.py**: Defines functions for performing OCR on the extracted ROIs using ArabicOCR and Tesseract OCR.

## How to Run

1. **Ensure the `card-rectification` repository is present** in the project directory.

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script**:
    ```bash
    python main.py
    ```

   or run the GUI:
    ```bash
    python gui.py
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
