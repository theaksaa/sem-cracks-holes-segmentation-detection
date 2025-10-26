import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO mode will not be available.")
    print("Install with: pip install ultralytics")

class YOLOCrackDetector:
    """YOLO-based crack detector"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.7):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold (default: 0.25)
            iou_threshold: IOU threshold for NMS (default: 0.7)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Color for crack visualization (magenta in BGR)
        self.color_map = {
            0: (0, 0, 0),        # Background (not used)
            1: (255, 0, 255),    # Cracks (magenta)
        }
    
    def draw_segmentation(self, image, mask, alpha=0.5):
        """
        Draw segmentation overlay on image
        
        Args:
            image: Input image (RGB or BGR)
            mask: Segmentation mask
            alpha: Transparency level (0-1)
        
        Returns:
            Blended image with segmentation overlay
        """
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        overlay = np.zeros_like(image, dtype=np.uint8)
        
        for class_id in range(len(self.color_map)):
            if class_id == 0:
                continue  # Skip background
            
            class_mask = (mask == class_id)
            color = self.color_map[class_id]
            
            for c in range(3):
                overlay[..., c][class_mask] = color[c]
        
        blended_image = image.copy()
        
        for class_id in range(len(self.color_map)):
            if class_id == 0:
                continue
            
            class_mask = (mask == class_id)
            for c in range(3):
                blended_image[..., c][class_mask] = (
                    image[..., c][class_mask] * (1 - alpha) +
                    overlay[..., c][class_mask] * alpha
                ).astype(np.uint8)
        
        return blended_image
    
    def detect(self, image_path):
        """
        Detect cracks using YOLO model
        
        Args:
            image_path: Path to input image
        
        Returns:
            tuple: (annotated_image, binary_mask, num_detections)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        
        # Initialize binary mask for combining with CV method
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        num_detections = 0
        
        if results.masks:
            for seg in results.masks.data:
                mask = seg.cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h))
                
                # Draw segmentation on image
                image_rgb = self.draw_segmentation(image_rgb, mask_resized, alpha=0.5)
                
                # Add to binary mask
                combined_mask[mask_resized > 0.5] = 255
                num_detections += 1
        
        # Convert back to BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr, combined_mask, num_detections
