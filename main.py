#!/usr/bin/env python3
"""
Combined preprocessing and prediction script for image processing pipeline.
Combines CLAHE enhancement, denoising, sharpening, and YOLO segmentation.

Default behavior matches:
  python preprocessing.py --input input/*.tif --out input --clahe --denoise --denoise-strength 20 --sharpen
  python predict.py --sensitivity custom --auto-borders --border-offset 30 --input input/*.jpg --output output/ --method both --model-path runs/segment/Cracks_Segmentation_yolov8/weights/best.pt --show-borders
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_preprocessing(input_pattern, output_dir, clahe=True, clahe_clip=2.0, clahe_tile=8,
                     denoise=True, denoise_strength=20, sharpen=True, sharpen_amount=1.5,
                     lower=0, upper=100, quality=95):
    """
    Run the preprocessing step with configurable parameters.
    
    Args:
        input_pattern: Input file pattern (e.g., 'input/*.tif')
        output_dir: Output directory for preprocessed images
        clahe: Apply CLAHE for local contrast enhancement
        clahe_clip: CLAHE clip limit
        clahe_tile: CLAHE tile size
        denoise: Apply denoising
        denoise_strength: Denoising strength
        sharpen: Apply sharpening
        sharpen_amount: Sharpening amount
        lower: Lower percentile for histogram stretch
        upper: Upper percentile for histogram stretch
        quality: JPEG quality for output
    """
    cmd = [
        'python', 'preprocessing.py',
        '--input', input_pattern,
        '--out', output_dir,
        '--lower', str(lower),
        '--upper', str(upper),
        '--quality', str(quality)
    ]
    
    if clahe:
        cmd.append('--clahe')
        cmd.extend(['--clahe-clip', str(clahe_clip)])
        cmd.extend(['--clahe-tile', str(clahe_tile)])
    
    if denoise:
        cmd.append('--denoise')
        cmd.extend(['--denoise-strength', str(denoise_strength)])
    
    if sharpen:
        cmd.append('--sharpen')
        cmd.extend(['--sharpen-amount', str(sharpen_amount)])
    
    print("=" * 70)
    print("STEP 1: Running Preprocessing")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nPreprocessing completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nPreprocessing failed with error code {e.returncode}\n", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("\nError: preprocessing.py not found in current directory\n", file=sys.stderr)
        return False


def run_prediction(input_pattern, output_dir, method='both', model_path=None,
                  conf=0.25, iou=0.7, sensitivity='auto', auto_borders=True,
                  border_offset=30, border_left=50, border_right=50,
                  border_top=50, border_bottom=50, show_borders=True, nosave=False):
    """
    Run the prediction step with configurable parameters.
    
    Args:
        input_pattern: Input file pattern (e.g., 'input/*.jpg')
        output_dir: Output directory for predictions
        method: Detection method ('cv', 'yolo', or 'both')
        model_path: Path to YOLO model weights
        conf: YOLO confidence threshold
        iou: YOLO IOU threshold
        sensitivity: CV detection sensitivity
        auto_borders: Automatically detect rectangular box boundary
        border_offset: Additional pixels to move border inward
        border_left: Left border in pixels
        border_right: Right border in pixels
        border_top: Top border in pixels
        border_bottom: Bottom border in pixels
        show_borders: Draw green border lines on output
        nosave: Preview only, do not save output
    """
    cmd = [
        'python', 'predict.py',
        '--input', input_pattern,
        '--output', output_dir,
        '--method', method,
        '--sensitivity', sensitivity,
        '--conf', str(conf),
        '--iou', str(iou),
        '--border-offset', str(border_offset),
        '--border-left', str(border_left),
        '--border-right', str(border_right),
        '--border-top', str(border_top),
        '--border-bottom', str(border_bottom)
    ]
    
    if model_path:
        cmd.extend(['--model-path', model_path])
    
    if auto_borders:
        cmd.append('--auto-borders')
    
    if show_borders:
        cmd.append('--show-borders')
    
    if nosave:
        cmd.append('--nosave')
    
    print("=" * 70)
    print("STEP 2: Running Prediction")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nPrediction completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nPrediction failed with error code {e.returncode}\n", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("\nError: predict.py not found in current directory\n", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Combined preprocessing and prediction pipeline for crack segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage with defaults (matches the two original commands)
            python process_and_predict.py
            
            # Custom input/output directories
            python process_and_predict.py --input-dir my_images --output-dir results
            
            # Custom preprocessing parameters
            python process_and_predict.py --denoise-strength 30 --sharpen-amount 2.0
            
            # Custom prediction parameters
            python process_and_predict.py --border-offset 40 --conf 0.3 --method yolo
            
            # Skip preprocessing (if already done)
            python process_and_predict.py --skip-preprocessing
        """
    )
    
    # Input/Output
    parser.add_argument('--input-dir', default='input',
                        help='Input directory containing .tif images (default: input)')
    parser.add_argument('--output-dir', default='output/',
                        help='Output directory for final predictions (default: output/)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing step and go directly to prediction')
    
    # Preprocessing parameters
    preprocess_group = parser.add_argument_group('preprocessing parameters')
    preprocess_group.add_argument('--no-clahe', action='store_true',
                                  help='Disable CLAHE (enabled by default)')
    preprocess_group.add_argument('--clahe-clip', type=float, default=2.0,
                                  help='CLAHE clip limit (default: 2.0)')
    preprocess_group.add_argument('--clahe-tile', type=int, default=8,
                                  help='CLAHE tile size (default: 8)')
    preprocess_group.add_argument('--no-denoise', action='store_true',
                                  help='Disable denoising (enabled by default)')
    preprocess_group.add_argument('--denoise-strength', type=int, default=20,
                                  help='Denoising strength (default: 20)')
    preprocess_group.add_argument('--no-sharpen', action='store_true',
                                  help='Disable sharpening (enabled by default)')
    preprocess_group.add_argument('--sharpen-amount', type=float, default=1.5,
                                  help='Sharpening amount (default: 1.5)')
    preprocess_group.add_argument('--lower', type=float, default=0,
                                  help='Lower percentile for histogram stretch (default: 0)')
    preprocess_group.add_argument('--upper', type=float, default=100,
                                  help='Upper percentile for histogram stretch (default: 100)')
    preprocess_group.add_argument('--quality', type=int, default=95,
                                  help='JPEG quality for output (default: 95)')
    
    # Prediction parameters
    predict_group = parser.add_argument_group('prediction parameters')
    predict_group.add_argument('--method', choices=['cv', 'yolo', 'both'], default='both',
                              help='Detection method (default: both)')
    predict_group.add_argument('--model-path',
                              default='models/best.pt',
                              help='Path to YOLO model weights')
    predict_group.add_argument('--conf', type=float, default=0.25,
                              help='YOLO confidence threshold (default: 0.25)')
    predict_group.add_argument('--iou', type=float, default=0.7,
                              help='YOLO IOU threshold (default: 0.7)')
    predict_group.add_argument('--sensitivity', 
                              choices=['low', 'medium', 'high', 'custom', 'auto'],
                              default='custom',
                              help='CV detection sensitivity (default: custom)')
    predict_group.add_argument('--no-auto-borders', action='store_true',
                              help='Disable automatic border detection (enabled by default)')
    predict_group.add_argument('--border-offset', type=int, default=30,
                              help='Additional pixels to move border inward (default: 30)')
    predict_group.add_argument('--border-left', type=int, default=50,
                              help='Left border in pixels (default: 50)')
    predict_group.add_argument('--border-right', type=int, default=50,
                              help='Right border in pixels (default: 50)')
    predict_group.add_argument('--border-top', type=int, default=50,
                              help='Top border in pixels (default: 50)')
    predict_group.add_argument('--border-bottom', type=int, default=50,
                              help='Bottom border in pixels (default: 50)')
    predict_group.add_argument('--no-show-borders', action='store_true',
                              help='Disable showing borders on output (enabled by default)')
    predict_group.add_argument('--nosave', action='store_true',
                              help='Preview only, do not save prediction output')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CRACK DETECTION PIPELINE")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model path: {args.model_path}")
    print()
    
    # Step 1: Preprocessing
    if not args.skip_preprocessing:
        input_pattern = f"{args.input_dir}/*.tif"
        success = run_preprocessing(
            input_pattern=input_pattern,
            output_dir=args.input_dir,
            clahe=not args.no_clahe,
            clahe_clip=args.clahe_clip,
            clahe_tile=args.clahe_tile,
            denoise=not args.no_denoise,
            denoise_strength=args.denoise_strength,
            sharpen=not args.no_sharpen,
            sharpen_amount=args.sharpen_amount,
            lower=args.lower,
            upper=args.upper,
            quality=args.quality
        )
        
        if not success:
            print("Pipeline failed at preprocessing step.")
            sys.exit(1)
    else:
        print("Skipping preprocessing step as requested.\n")
    
    # Step 2: Prediction
    input_pattern = f"{args.input_dir}/*.jpg"
    success = run_prediction(
        input_pattern=input_pattern,
        output_dir=args.output_dir,
        method=args.method,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        sensitivity=args.sensitivity,
        auto_borders=not args.no_auto_borders,
        border_offset=args.border_offset,
        border_left=args.border_left,
        border_right=args.border_right,
        border_top=args.border_top,
        border_bottom=args.border_bottom,
        show_borders=not args.no_show_borders,
        nosave=args.nosave
    )
    
    if not success:
        print("Pipeline failed at prediction step.")
        sys.exit(1)
    
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Results saved to: {args.output_dir}")
    print()


if __name__ == '__main__':
    main()