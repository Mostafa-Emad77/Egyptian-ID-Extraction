import os
import subprocess

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
    """
    subprocess.run(['python', 'rectify.py', image_path, image_path])
