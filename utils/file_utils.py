import os
import glob
from pathlib import Path

def get_image_files(input_path):
    """
    Get list of image files from input path
    
    Args:
        input_path: Can be:
            - Single file: 'image.jpg'
            - Directory: 'input' or 'input/'
            - Glob pattern: 'input/*.jpg'
    
    Returns:
        List of image file paths
    """
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # If it's a glob pattern
    if '*' in input_path:
        files = glob.glob(input_path)
        return [f for f in files if Path(f).suffix.lower() in image_extensions]
    
    # If it's a directory
    if os.path.isdir(input_path):
        files = []
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(input_path, f'*{ext}')))
            files.extend(glob.glob(os.path.join(input_path, f'*{ext.upper()}')))
        return sorted(files)
    
    # Single file
    if os.path.isfile(input_path):
        return [input_path]
    
    return []
