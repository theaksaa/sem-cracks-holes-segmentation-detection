#!/usr/bin/env python3
"""
Preprocessing script for dark microscopic TIF images.
Simple histogram stretching for clean, noise-free results.
"""

import argparse
import sys
import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from preprocessing.image_preprocessor import preprocess_microscopic_image
from utils.file_utils import get_image_files

def load_tif_image(input_path):
    """
    Load TIF image and return as numpy array.
    
    Args:
        input_path: Path to input TIF file
        
    Returns:
        numpy array of image data
    """
    try:
        # Load with PIL to handle 16-bit TIF properly
        img = Image.open(input_path)
        img_array = np.array(img)
        print(f"Loaded image: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"Value range: {img_array.min()} - {img_array.max()}")
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def save_image(img_array, output_path, quality=95):
    """
    Save processed image to file.
    
    Args:
        img_array: Image array to save
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    try:
        img = Image.fromarray(img_array)
        
        # Determine format from extension
        output_path = Path(output_path)
        ext = output_path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg']:
            img.save(output_path, 'JPEG', quality=quality)
        elif ext == '.png':
            img.save(output_path, 'PNG')
        elif ext in ['.tif', '.tiff']:
            img.save(output_path, 'TIFF')
        else:
            img.save(output_path)
        
        print(f"\nImage saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        sys.exit(1)


def preview_image(img_array, window_name="Preprocessed Image"):
    """
    Display image in a window for preview.
    Press any key to close.
    
    Args:
        img_array: Image array to display
        window_name: Name of the display window
    """
    try:
        # Resize if image is too large for screen
        height, width = img_array.shape[:2]
        max_display_size = 1200
        
        if height > max_display_size or width > max_display_size:
            scale = max_display_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_display = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"\nDisplay size: {new_width}x{new_height} (scaled for screen)")
        else:
            img_display = img_array
        
        cv2.imshow(window_name, img_display)
        print("\nPreview window opened. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error displaying image: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess microscopic TIF images with clean histogram stretching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Single image - simple histogram stretch (clean, no noise - recommended!)
            python preprocessing.py --input image.tif --out output.jpg
            
            # Process entire folder
            python preprocessing.py --input input/ --out output/
            
            # Process with glob pattern
            python preprocessing.py --input "input/*.tif" --out output/
            
            # Preview without saving
            python preprocessing.py --input image.tif --nosave
            
            # Custom percentile range
            python preprocessing.py --input image.tif --out output.jpg --lower 1 --upper 99
            
            # Add CLAHE for more local contrast
            python preprocessing.py --input image.tif --out output.jpg --clahe
            
            # Add sharpening for crack enhancement
            python preprocessing.py --input image.tif --out output.jpg --sharpen
            
            # Batch processing with all enhancements
            python preprocessing.py --input input/ --out output/ --clahe --sharpen --lower 1 --upper 99
        """
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, 
                       help='Input: single file, folder, or glob pattern (e.g., input/*.tif)')
    parser.add_argument('--out', help='Output: file or directory (required unless --nosave)')
    parser.add_argument('--nosave', action='store_true', help='Preview only, do not save')
    
    # Histogram stretching parameters
    parser.add_argument('--lower', type=float, default=0, 
                       help='Lower percentile for histogram stretch (0-100, default: 0)')
    parser.add_argument('--upper', type=float, default=100,
                       help='Upper percentile for histogram stretch (0-100, default: 100)')
    
    # Optional processing
    parser.add_argument('--clahe', action='store_true', 
                       help='Apply CLAHE for local contrast enhancement')
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                       help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--clahe-tile', type=int, default=8,
                       help='CLAHE tile size (default: 8)')
    
    parser.add_argument('--denoise', action='store_true',
                       help='Apply denoising (use if image is noisy)')
    parser.add_argument('--denoise-strength', type=int, default=10,
                       help='Denoising strength (default: 10)')
    
    parser.add_argument('--sharpen', action='store_true',
                       help='Apply sharpening to enhance cracks')
    parser.add_argument('--sharpen-amount', type=float, default=1.5,
                       help='Sharpening amount (default: 1.5)')
    
    # Output quality
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality for output (1-100, default: 95)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.nosave and not args.out:
        print("Error: --out is required unless --nosave is specified")
        sys.exit(1)
    
    if args.lower < 0 or args.lower > 100 or args.upper < 0 or args.upper > 100:
        print("Error: Percentiles must be between 0 and 100")
        sys.exit(1)
    
    if args.lower >= args.upper:
        print("Error: Lower percentile must be less than upper percentile")
        sys.exit(1)
    
    # Get input files
    image_files = get_image_files(args.input)
    
    if not image_files:
        print(f"Error: No images found at: {args.input}")
        sys.exit(1)
    
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Warning for preview mode with multiple images
    if args.nosave and len(image_files) > 1:
        print("\nWarning: Preview mode with multiple images!")
        print("Press any key on each image to continue to the next one.")
        print("Press Ctrl+C to stop processing.\n")
    
    # Determine if batch processing
    is_batch = len(image_files) > 1
    
    if is_batch and not args.nosave:
        # Create output directory
        output_dir = Path(args.out)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Process images
    success_count = 0
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(image_files)}] Processing: {Path(img_path).name}")
        print(f"{'='*80}")
        
        try:
            # Load image
            img_array = load_tif_image(img_path)
            
            # Preprocess
            processed = preprocess_microscopic_image(
                img_array,
                lower_percentile=args.lower,
                upper_percentile=args.upper,
                use_clahe=args.clahe,
                clahe_clip=args.clahe_clip,
                clahe_tile=args.clahe_tile,
                use_denoise=args.denoise,
                denoise_strength=args.denoise_strength,
                use_sharpen=args.sharpen,
                sharpen_amount=args.sharpen_amount
            )
            
            # Save or preview
            if args.nosave:
                preview_image(processed, f"Image {idx}/{len(image_files)}: {Path(img_path).name}")
            else:
                if is_batch:
                    # Generate output filename (keep original extension or use jpg)
                    img_name = Path(img_path).stem
                    output_ext = Path(args.out).suffix if not Path(args.out).is_dir() else '.jpg'
                    if output_ext == '':
                        output_ext = '.jpg'
                    output_path = output_dir / f"{img_name}{output_ext}"
                else:
                    output_path = Path(args.out)
                
                save_image(processed, output_path, quality=args.quality)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Processing complete: {success_count}/{len(image_files)} successful")
    print(f"{'='*80}")
    
    if not args.nosave and success_count > 0:
        if is_batch:
            print(f"\nDone! Check your output folder: {args.out}")
        else:
            print(f"\nDone! Check your output: {args.out}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())