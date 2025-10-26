import cv2
import numpy as np

class CrackDetector:
    def __init__(self, sensitivity='medium', border_left=50, border_right=50, 
                 border_top=50, border_bottom=50, auto_detect_borders=False, 
                 border_offset=0):
        """
        Initialize crack detector with configurable sensitivity
        
        Args:
            sensitivity: 'low', 'medium', 'high', 'auto' - affects detection threshold
            border_left: pixels to ignore from left border (default: 50)
            border_right: pixels to ignore from right border (default: 50)
            border_top: pixels to ignore from top border (default: 50)
            border_bottom: pixels to ignore from bottom border (default: 50)
            auto_detect_borders: If True, automatically detect rectangular box boundary
            border_offset: Pixels to move border inward from detected edge (default: 0)
        """
        self.sensitivity_map = {
            'low': {'threshold': 10, 'morph_size': 3, 'min_area': 15},
            'medium': {'threshold': 20, 'morph_size': 4, 'min_area': 10},
            'high': {'threshold': 35, 'morph_size': 5, 'min_area': 5},
            'custom': {'threshold': 47, 'morph_size': 3, 'min_area': 2},
            'auto': {'threshold': None, 'morph_size': 1, 'min_area': 2},
        }
        self.sensitivity = sensitivity
        self.params = self.sensitivity_map.get(sensitivity, self.sensitivity_map['medium'])
        self.auto_detect_borders = auto_detect_borders
        self.border_offset = border_offset
        self.border_left = border_left
        self.border_right = border_right
        self.border_top = border_top
        self.border_bottom = border_bottom
        self.detected_box = None
        self.detected_box_contour = None
    
    def detect_rectangular_box(self, image):
        """
        Automatically detect the rectangular box boundary using edge detection.
        """
        print("\n=== Auto-detecting rectangular box boundary ===")
        
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Warning: No contours found in image")
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        self.detected_box_contour = largest_contour
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect = cv2.minAreaRect(largest_contour)
        
        img_area = image.shape[0] * image.shape[1]
        box_ratio = area / img_area
        
        print(f"Detected box: x={x}, y={y}, w={w}, h={h}")
        print(f"  Area: {area} pixels ({box_ratio*100:.1f}% of image)")
        
        if box_ratio < 0.3 or box_ratio > 0.95:
            print(f"Warning: Detected box ratio {box_ratio*100:.1f}% seems invalid")
            return None
        
        return (x, y, w, h)
    
    def draw_detected_box(self, image, show_offset=True):
        """Draw the detected box boundary with cyan color"""
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
        
        if self.detected_box_contour is not None:
            cyan = (255, 255, 0)
            thickness = 3
            cv2.drawContours(output, [self.detected_box_contour], -1, cyan, thickness)
            
            if show_offset and self.border_offset > 0:
                h, w = image.shape[:2]
                temp_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(temp_mask, [self.detected_box_contour], -1, 255, -1)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                   (self.border_offset*2+1, self.border_offset*2+1))
                eroded_mask = cv2.erode(temp_mask, kernel, iterations=1)
                
                offset_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                if offset_contours:
                    green = (0, 255, 0)
                    cv2.drawContours(output, offset_contours, -1, green, thickness)
        
        return output
    
    def set_borders_from_box(self, image, box):
        """Set border values based on detected box coordinates"""
        if box is None:
            print("No box detected, using manual border settings")
            return
        
        x, y, w, h = box
        img_h, img_w = image.shape[:2]
        
        self.border_left = x + self.border_offset
        self.border_right = img_w - (x + w) + self.border_offset
        self.border_top = y + self.border_offset
        self.border_bottom = img_h - (y + h) + self.border_offset
        
        print(f"Set borders from detected box (with {self.border_offset}px offset):")
        print(f"  Left: {self.border_left}, Right: {self.border_right}")
        print(f"  Top: {self.border_top}, Bottom: {self.border_bottom}")
        
        self.detected_box = box
    
    def create_border_mask(self, image):
        """Create mask for the active detection region"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.auto_detect_borders and self.detected_box_contour is not None:
            cv2.drawContours(mask, [self.detected_box_contour], -1, 255, -1)
            
            if self.border_offset > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                   (self.border_offset*2+1, self.border_offset*2+1))
                mask = cv2.erode(mask, kernel, iterations=1)
        else:
            x1 = self.border_left
            x2 = w - self.border_right
            y1 = self.border_top
            y2 = h - self.border_bottom
            
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def detect(self, image):
        """
        Main detection method using computer vision
        Returns: Binary mask of detected regions
        """
        if self.auto_detect_borders:
            box = self.detect_rectangular_box(image)
            self.set_borders_from_box(image, box)
        
        border_mask = self.create_border_mask(image)
        
        if self.params['threshold'] is None:
            mean_val = np.mean(image)
            std_val = np.std(image)
            threshold = max(5, int(mean_val - 2 * std_val) * 1.05)
            print(f"Auto threshold: {threshold} (mean={mean_val:.1f}, std={std_val:.1f})")
        else:
            threshold = self.params['threshold']
        
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        morph_size = self.params['morph_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        binary = cv2.bitwise_and(binary, border_mask)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_mask = np.zeros_like(binary)
        min_area = self.params['min_area']
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
