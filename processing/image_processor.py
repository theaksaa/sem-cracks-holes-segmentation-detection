import cv2
import numpy as np
import os

def process_single_image(image_path, output_path, args, cv_detector=None, yolo_detector=None):
    """Process a single image with selected detection method(s)"""
    
    print(f"\n{'='*80}")
    print(f"Processing: {image_path}")
    print(f"{'='*80}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return False
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize masks
    cv_mask = None
    yolo_mask = None
    combined_mask = None
    
    # Computer Vision detection
    if args.method in ['cv', 'both']:
        print("\n--- Computer Vision Detection ---")
        cv_mask = cv_detector.detect(image)
        cv_count = len(cv2.findContours(cv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        print(f"CV detected {cv_count} regions")
    
    # YOLO detection
    if args.method in ['yolo', 'both']:
        print("\n--- YOLO Detection ---")
        yolo_image, yolo_mask, yolo_count = yolo_detector.detect(image_path)
        print(f"YOLO detected {yolo_count} cracks")
    
    # Combine masks if using both methods
    if args.method == 'both':
        print("\n--- Combining Detections ---")
        combined_mask = cv2.bitwise_or(cv_mask, yolo_mask)
        total_count = len(cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        print(f"Total combined regions: {total_count}")
    elif args.method == 'cv':
        combined_mask = cv_mask
    elif args.method == 'yolo':
        combined_mask = yolo_mask
    
    # Create output visualization
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if args.method == 'cv':
        # Red for CV detections
        output_image[cv_mask > 0] = (0, 0, 255)
        if args.show_borders:
            output_image = cv_detector.draw_detected_box(output_image)
    
    elif args.method == 'yolo':
        # Use YOLO's annotated image
        output_image = yolo_image
    
    elif args.method == 'both':
        # Green for CV (holes), Magenta for YOLO (cracks)
        # If overlap, show as yellow
        cv_only = cv2.bitwise_and(cv_mask, cv2.bitwise_not(yolo_mask))
        yolo_only = cv2.bitwise_and(yolo_mask, cv2.bitwise_not(cv_mask))
        overlap = cv2.bitwise_and(cv_mask, yolo_mask)
        
        output_image[cv_only > 0] = (0, 255, 0)      # Green for CV only (holes)
        output_image[yolo_only > 0] = (255, 0, 255)  # Magenta for YOLO only (cracks)
        output_image[overlap > 0] = (0, 255, 255)    # Yellow for overlap
        
        if args.show_borders:
            output_image = cv_detector.draw_detected_box(output_image)
    
    # Create comparison view
    original_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack([original_color, output_image])
    
    # Save output
    if not args.nosave:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(output_path, output_image)
        print(f"\nSaved output to: {output_path}")
        
        # Save comparison view
        comparison_path = output_path.rsplit('.', 1)[0] + '_comparison.jpg'
        cv2.imwrite(comparison_path, comparison)
        print(f"Saved comparison to: {comparison_path}")
    else:
        # Preview mode - show the image
        print(f"\n--- Preview Mode (not saving) ---")
        
        # Resize for display if image is too large
        display = comparison.copy()
        max_width = 1920
        if display.shape[1] > max_width:
            scale = max_width / display.shape[1]
            new_width = int(display.shape[1] * scale)
            new_height = int(display.shape[0] * scale)
            display = cv2.resize(display, (new_width, new_height))
        
        # Create window title based on detection method
        if args.method == 'cv':
            window_title = 'Crack Detection [CV] - Original | Result (Red: detections)'
        elif args.method == 'yolo':
            window_title = 'Crack Detection [YOLO] - Original | Result (Magenta: cracks)'
        elif args.method == 'both':
            window_title = 'Crack Detection [Combined] - Original | Result (Green: CV, Magenta: YOLO, Yellow: Both)'
        
        cv2.imshow(window_title, display)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True