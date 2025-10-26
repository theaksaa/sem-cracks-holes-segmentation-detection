import argparse
import sys
import os

from detectors.cv_detector import CrackDetector
from detectors.yolo_detector import YOLOCrackDetector
from utils.file_utils import get_image_files
from processing.image_processor import process_single_image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO mode will not be available.")
    print("Install with: pip install ultralytics")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced crack and hole detection using CV and/or YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Detection Methods:
            --method cv       : Use computer vision only (good for black holes)
            --method yolo     : Use YOLO model only (good for cracks)
            --method both     : Use both methods (comprehensive detection)

            Input Options:
            --input image.jpg           : Process single image
            --input /path/to/folder     : Process all images in folder
            --input /path/to/*.jpg      : Process images matching pattern

            Examples:
            # CV only with auto borders
            python predict_enhanced.py --method cv --input image.jpg --output result.jpg --auto-borders
            
            # YOLO only
            python predict_enhanced.py --method yolo --input image.jpg --output result.jpg \\
                --model-path runs/segment/weights/best.pt
            
            # Both methods combined
            python predict_enhanced.py --method both --input image.jpg --output result.jpg \\
                --auto-borders --model-path runs/segment/weights/best.pt
            
            # Process entire folder
            python predict_enhanced.py --method both --input input/ --output output/ \\
                --auto-borders --model-path runs/segment/weights/best.pt
            
            # Process with pattern
            python predict_enhanced.py --method cv --input "input/*.jpg" --output output/ \\
                --sensitivity high --auto-borders
        """
    )
    
    # Method selection
    parser.add_argument('--method', choices=['cv', 'yolo', 'both'], default='both',
                        help='Detection method: cv (computer vision), yolo (model), or both (default: both)')
    
    # Input/Output
    parser.add_argument('--input', required=True, 
                        help='Input: single file, folder, or glob pattern (e.g., input/*.jpg)')
    parser.add_argument('--output', required=True, 
                        help='Output: file or directory')
    parser.add_argument('--nosave', action='store_true', 
                        help='Preview only, do not save output')
    
    # YOLO parameters
    parser.add_argument('--model-path', type=str, 
                        default='runs/segment/Cracks_Segmentation_yolov8/weights/best.pt',
                        help='Path to YOLO model weights (default: runs/segment/Cracks_Segmentation_yolov8/weights/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='YOLO IOU threshold (default: 0.7)')
    
    # CV parameters
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high', 'custom', 'auto'], 
                        default='medium', help='CV detection sensitivity (default: medium)')
    parser.add_argument('--auto-borders', action='store_true',
                        help='Automatically detect rectangular box boundary for CV')
    parser.add_argument('--border-offset', type=int, default=0,
                        help='Additional pixels to move border inward (default: 0)')
    parser.add_argument('--border-left', type=int, default=50)
    parser.add_argument('--border-right', type=int, default=50)
    parser.add_argument('--border-top', type=int, default=50)
    parser.add_argument('--border-bottom', type=int, default=50)
    parser.add_argument('--show-borders', action='store_true',
                        help='Draw green border lines on output')
    
    args = parser.parse_args()
    
    # Validate method requirements
    if args.method in ['yolo', 'both'] and not YOLO_AVAILABLE:
        print("Error: YOLO mode requires ultralytics package")
        print("Install with: pip install ultralytics")
        sys.exit(1)
    
    if args.method in ['yolo', 'both'] and not os.path.exists(args.model_path):
        print(f"Error: YOLO model not found: {args.model_path}")
        sys.exit(1)
    
    # Initialize detectors
    cv_detector = None
    yolo_detector = None
    
    if args.method in ['cv', 'both']:
        print("\n=== Initializing Computer Vision Detector ===")
        cv_detector = CrackDetector(
            sensitivity=args.sensitivity,
            border_left=args.border_left,
            border_right=args.border_right,
            border_top=args.border_top,
            border_bottom=args.border_bottom,
            auto_detect_borders=args.auto_borders,
            border_offset=args.border_offset
        )
        print(f"Sensitivity: {args.sensitivity}")
        if args.auto_borders:
            print("Border detection: AUTOMATIC")
        else:
            print(f"Borders: L={args.border_left}, R={args.border_right}, "
                  f"T={args.border_top}, B={args.border_bottom}")
    
    if args.method in ['yolo', 'both']:
        print("\n=== Initializing YOLO Detector ===")
        yolo_detector = YOLOCrackDetector(
            model_path=args.model_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        print(f"Confidence threshold: {args.conf}")
        print(f"IOU threshold: {args.iou}")
    
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
    
    # Determine output paths
    is_batch = len(image_files) > 1
    
    if is_batch:
        # Create output directory
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    success_count = 0
    
    for idx, img_path in enumerate(image_files, 1):
        if is_batch:
            # Generate output filename
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, img_name)
        else:
            output_path = args.output
        
        print(f"\n[{idx}/{len(image_files)}]", end=" ")
        
        if process_single_image(img_path, output_path, args, cv_detector, yolo_detector):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Processing complete: {success_count}/{len(image_files)} successful")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()