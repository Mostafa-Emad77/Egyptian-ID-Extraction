import os
import cv2
from image_processing import extract_roi, resize_image
from image_preprocessing import preprocess_image, apply_adaptive_thresholding
from image_alignment import align_images
from git_utils import clone_repository, change_directory, run_rectify_script
from ocr_extraction import extract_text_from_roi, extract_text_from_roi_with_tesseract

def main():
    repo_url = 'https://github.com/shakex/card-rectification'
    target_dir = 'card-rectification'
    clone_repository(repo_url, target_dir)
    
    change_directory(target_dir)

    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    image_path = 'test.jpg'
    run_rectify_script(image_path)

    preprocessed_image_path = os.path.join(output_folder, 'preprocessed_test.jpg')
    preprocessed_image = preprocess_image(image_path)
    cv2.imwrite(preprocessed_image_path, preprocessed_image)

    thresholded_image = apply_adaptive_thresholding(preprocessed_image)
    thresholded_image_path = os.path.join(output_folder, 'thresholded_test.jpg')
    cv2.imwrite(thresholded_image_path, thresholded_image)

    reference_image_path = 'ref.jpg'
    aligned_image_path = os.path.join(output_folder, 'aligned_test.jpg')
    align_images(thresholded_image_path, reference_image_path, aligned_image_path)

    print("Image alignment completed. Check the aligned image for results.")

    aligned_img = cv2.imread(aligned_image_path)

    resized_img = resize_image(aligned_img, 1280, 819)

    # Extract ROIs
    first_name_roi = extract_roi(resized_img, 500, 210, 1225, 290)
    last_name_roi = extract_roi(resized_img, 500, 280, 1220, 380)
    address_roi = extract_roi(resized_img, 500, 385, 1225, 550)
    id_roi = extract_roi(resized_img, 550, 600, 1240, 750)
    face_roi = extract_roi(resized_img, 20, 50, 350, 450)

    # Save the ROIs as images
    cv2.imwrite(os.path.join(output_folder, 'first_name_roi.jpg'), first_name_roi)
    cv2.imwrite(os.path.join(output_folder, 'last_name_roi.jpg'), last_name_roi)
    cv2.imwrite(os.path.join(output_folder, 'address_roi.jpg'), address_roi)
    cv2.imwrite(os.path.join(output_folder, 'id_roi.jpg'), id_roi)
    cv2.imwrite(os.path.join(output_folder, 'face_roi.jpg'), face_roi)
    print("ROIs extracted and saved as images.")

    # Perform OCR on the ROIs
    extract_text_from_roi(os.path.join(output_folder, 'first_name_roi.jpg'), os.path.join(output_folder, 'first_name.txt'))
    extract_text_from_roi(os.path.join(output_folder, 'last_name_roi.jpg'), os.path.join(output_folder, 'last_name.txt'))
    extract_text_from_roi(os.path.join(output_folder, 'address_roi.jpg'), os.path.join(output_folder, 'address.txt'))
    extract_text_from_roi_with_tesseract(os.path.join(output_folder, 'id_roi.jpg'), os.path.join(output_folder, 'id.txt'))

    print("OCR extraction completed. Check the output files in the 'output' folder for results.")

if __name__ == "__main__":
    main()
