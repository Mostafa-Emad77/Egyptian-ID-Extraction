import os
import subprocess
import shutil

def clone_repository(repo_url, target_dir):
    """
    Clone a Git repository.
    """
    subprocess.run(['git', 'clone', repo_url, target_dir])

def change_directory(directory):
    """
    Change the current working directory.
    """
    os.chdir(directory)

def run_rectify_script(image_path):
    """
    Run the rectify.py script with the provided image path.
    Returns the path to the rectified image.
    """
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    output_path = os.path.join(result_dir, 'rectified_' + os.path.basename(image_path))
    result = subprocess.run(['python', 'rectify.py', image_path, output_path], capture_output=True)

    if result.returncode != 0:
        # Raise an exception with the error message from the script
        raise Exception("Rectification failed. Ensure the image is of good quality and lighting, similar to the provided example.")
    
    if not os.path.exists(output_path):
        raise Exception("Rectification failed to produce output image.")
        
    return output_path
