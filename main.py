import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import cv2
from image_processing import extract_roi, resize_image
from image_preprocessing import preprocess_image, apply_adaptive_thresholding
from git_utils import change_directory, run_rectify_script
from ocr_extraction import extract_text_from_roi, extract_text_from_roi_with_tesseract
import logging
from pathlib import Path
from typing import Dict
import json
import shutil
import base64
import numpy as np

class IDCardProcessor:
    def __init__(self):
        self.setup_logging()
        # Store the original working directory
        self.original_dir = Path.cwd()
        # Make output folder in the original directory
        self.output_folder = self.original_dir / 'output'
        self.output_folder.mkdir(exist_ok=True)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('id_processing.log'),
                logging.StreamHandler()  # Added console logging
            ]
        )
        
    def process_image(self, image_path: str) -> Dict[str, str]:
        try:
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            logging.info(f"Processing image: {image_path}")
            
            # Get repository directory and run rectification first
            repo_dir = self._setup_repository()
            
            # Save current directory
            original_dir = Path.cwd()
            try:
                # Change to repository directory
                os.chdir(str(repo_dir))
                
                # Run rectification first and get the rectified image path
                logging.info("Running image rectification...")
                rectified_path = run_rectify_script(str(image_path))
                
                logging.info(f"Rectified image saved at: {rectified_path}")
                
                # Process the rectified image
                return self.process_rectified_image(rectified_path)
                
            finally:
                # Always return to original directory
                os.chdir(str(original_dir))
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            raise ValueError("The image must be of good quality and lighting, similar to the provided example.")

    def process_rectified_image(self, rectified_path: str) -> Dict[str, str]:
        try:
            logging.info(f"Processing rectified image: {rectified_path}")
            # Process image with absolute paths
            preprocessed_img = self._preprocess_image(rectified_path)
            if preprocessed_img is None:
                raise ValueError("Preprocessing failed to produce valid image")
            
            roi_results = self._extract_rois(preprocessed_img)
            logging.info(f"Extracted ROIs: {roi_results}")
            
            ocr_results = self._perform_ocr(roi_results)
            
            logging.info("OCR extraction completed successfully")
            return ocr_results
            
        except Exception as e:
            logging.error(f"Error processing rectified image: {str(e)}")
            raise

    def _setup_repository(self):
        # Simply return the path to the existing repository
        try:
            target_dir = self.original_dir / 'card-rectification'
            if not target_dir.exists():
                raise FileNotFoundError(f"Repository directory does not exist: {target_dir}")
            return target_dir
        except Exception as e:
            logging.error(f"Repository setup failed: {str(e)}")
            raise

    def _preprocess_image(self, image_path: str):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image file: {image_path}")
            
            logging.info(f"Successfully read image: {image_path}")
            
            # Create output folder if it doesn't exist
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            preprocessed = preprocess_image(image_path)
            output_path = self.output_folder / 'preprocessed_test.jpg'
            success = cv2.imwrite(str(output_path), preprocessed)
            if not success:
                raise ValueError(f"Failed to save preprocessed image to {output_path}")
            
            thresholded = apply_adaptive_thresholding(preprocessed)
            thresh_path = self.output_folder / 'thresholded_test.jpg'
            success = cv2.imwrite(str(thresh_path), thresholded)
            if not success:
                raise ValueError(f"Failed to save thresholded image to {thresh_path}")
            
            if not thresh_path.exists():
                raise FileNotFoundError(f"Failed to create thresholded image at {thresh_path}")
            
            logging.info(f"Successfully preprocessed image and saved to {thresh_path}")
            
            # Check the type of the image before returning
            if not isinstance(thresholded, (np.ndarray,)):
                raise ValueError("Preprocessed image is not a valid array")
            
            return thresholded
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise

    def _extract_rois(self, img):
        roi_configs = {
            'first_name': (500, 210, 1225, 290),
            'last_name': (500, 280, 1220, 380),
            'address': (500, 385, 1225, 550),
            'id': (550, 600, 1240, 750),
            'face': (20, 50, 350, 450)
        }
        
        rois = {}
        try:
            if img is None:
                raise ValueError("Input image is None")
                
            for name, coords in roi_configs.items():
                roi = extract_roi(img, *coords)
                output_path = self.output_folder / f'{name}_roi.jpg'
                cv2.imwrite(str(output_path), roi)
                rois[name] = output_path
            return rois
        except Exception as e:
            logging.error(f"ROI extraction failed: {str(e)}")
            raise

    def _perform_ocr(self, roi_paths: Dict[str, Path]) -> Dict[str, str]:
        results = {}
        try:
            for name, path in roi_paths.items():
                if not path.exists():
                    raise FileNotFoundError(f"ROI image not found: {path}")
                
                if name == 'face':
                    # Just copy face image to output directory
                    face_output = self.output_folder / 'face.jpg'
                    shutil.copy2(str(path), str(face_output))
                    continue  # Skip adding face to results
                else:
                    # For text fields, extract the text
                    temp_output = self.output_folder / f'temp_{name}.txt'
                    if name == 'id':
                        extract_text_from_roi_with_tesseract(str(path), str(temp_output))
                    else:
                        extract_text_from_roi(str(path), str(temp_output))
                    
                    # Read the extracted text
                    with open(temp_output, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    results[name] = text
                    
                    # Remove temporary text file
                    temp_output.unlink()
                
            # Create final JSON output (without face image)
            output_json = {
                'first_name': results.get('first_name', ''),
                'last_name': results.get('last_name', ''),
                'address': results.get('address', ''),
                'id_number': results.get('id', '')
            }
            
            # Save JSON to file
            json_path = self.output_folder / 'id_card_data.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
                
            logging.info("OCR extraction completed and JSON file created")
            return output_json
            
        except Exception as e:
            logging.error(f"OCR processing failed: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup temporary ROI images"""
        try:
            for name in ['first_name_roi.jpg', 'last_name_roi.jpg', 'address_roi.jpg', 'id_roi.jpg']:
                roi_path = self.output_folder / name
                if roi_path.exists():
                    roi_path.unlink()
            
            # Remove other temporary files
            for temp_file in self.output_folder.glob('preprocessed_*.jpg'):
                temp_file.unlink()
            for temp_file in self.output_folder.glob('thresholded_*.jpg'):
                temp_file.unlink()
                
        except Exception as e:
            logging.warning(f"Cleanup failed: {str(e)}")

def main():
    try:
        processor = IDCardProcessor()
        # Use absolute path for input image
        image_path = Path('test.jpg').resolve()
        if not image_path.exists():
            # Try alternative location
            image_path = Path('card-rectification/test.jpg').resolve()
            
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find test.jpg in current or card-rectification directory")
            
        logging.info(f"Starting processing with image: {image_path}")
        results = processor.process_image(str(image_path))
        
        # Cleanup temporary files
        processor.cleanup()
        
        logging.info("Processing completed successfully")
        print("Processing completed. Check id_card_data.json in the output folder for results.")
        return results
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()
