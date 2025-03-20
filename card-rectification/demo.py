import os
import sys
from typing import List, Tuple
from load_model import load_model
from rectify import inference

"""
1. load_model()
根据模型类型，导入存储在硬盘中的模型文件至内存。

Parameters: 
None

Returns:
- model: {UNetRNN}
模型对象，包括模型各层结构和预训练的参数。
- device: {device}
torch.device类对象，表示分配给torch.Tensor进行运算的设备。包含设备类型（"cpu"或"cuda"）和设备序号。

Example:
from load_model import load_model
model, device = load_model()


2. inferecne(input_path, output_path, model, device)
校正推理，对单张图像进行校正处理。

Parameters:
- input_path: {str}
待校正图像路径

- output_path: {str}
图像保存路径

- model: {UNetRNN}
模型对象，包括模型各层结构和预训练的参数。

- device: {device}
torch.device类对象，表示分配给torch.Tensor进行运算的设备。包含设备类型（"cpu"或"cuda"）和设备序号。

Example:
from rectify import inference
from load_model import load_model
input = 'example/card.jpg'
output = 'result/card.png'
model, device = load_model()
inference(input, output, trained_model, device)

"""

def process_image(
    input_path: str,
    output_path: str,
    model: any,
    device: any
) -> bool:
    """
    Process a single image using the ID card rectification model.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save output image
        model: Loaded PyTorch model
        device: PyTorch device to use
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process the image
        inference(input_path, output_path, model, device)
        print(f"Successfully processed: {input_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_directory(
    input_dir: str,
    output_dir: str,
    model: any,
    device: any
) -> Tuple[int, int]:
    """
    Process all images in a directory.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save processed images
        model: Loaded PyTorch model
        device: PyTorch device to use
        
    Returns:
        Tuple[int, int]: Number of successful and total processed images
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.bmp', '.png'}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return 0, 0
        
    total = len(image_files)
    successful = 0
    
    print(f"Processing {total} images...")
    
    for i, image_file in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(
            output_dir,
            os.path.splitext(image_file)[0] + '.png'
        )
        
        if process_image(input_path, output_path, model, device):
            successful += 1
            
        print(f"Progress: {i}/{total} ({(i/total)*100:.1f}%)")
        
    return successful, total

def main():
    """
    Main function to demonstrate ID card rectification.
    """
    try:
        # Example paths
        example_dir = 'example'
        result_dir = 'result'
        
        # Load model
        print("Loading model...")
        trained_model, device = load_model()
        print(f"Model loaded successfully on device: {device}")
        
        # Process example images
        if os.path.exists(example_dir):
            print(f"\nProcessing images in {example_dir}...")
            successful, total = process_directory(
                example_dir,
                result_dir,
                trained_model,
                device
            )
            print(f"\nProcessing complete: {successful}/{total} images processed successfully")
        else:
            print(f"Example directory not found: {example_dir}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
