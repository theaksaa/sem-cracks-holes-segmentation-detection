#!/usr/bin/env python3
"""
Preprocessing script for dark microscopic TIF images.
Simple histogram stretching for clean, noise-free results.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def histogram_stretch(img_array, lower_percentile=0, upper_percentile=100):
    """
    Simple histogram stretching - clean and effective.
    This is what those online tools do - no noise added!
    
    Args:
        img_array: Input image array (16-bit or 8-bit)
        lower_percentile: Lower percentile for stretching (0-100)
        upper_percentile: Upper percentile for stretching (0-100)
        
    Returns:
        Stretched 8-bit image array
    """
    print(f"\nApplying histogram stretch ({lower_percentile}% - {upper_percentile}%)...")
    
    # Calculate percentile values
    p_low = np.percentile(img_array, lower_percentile)
    p_high = np.percentile(img_array, upper_percentile)
    
    print(f"Original range: {img_array.min()} - {img_array.max()}")
    print(f"Stretch range: {p_low:.1f} - {p_high:.1f}")
    
    # Clip values to percentile range
    img_clipped = np.clip(img_array, p_low, p_high)
    
    # Stretch to full 0-255 range
    if p_high > p_low:
        img_stretched = ((img_clipped - p_low) / (p_high - p_low) * 255.0)
    else:
        print("Warning: No dynamic range in image")
        img_stretched = np.zeros_like(img_array, dtype=np.float64)
    
    # Convert to 8-bit
    img_8bit = np.clip(img_stretched, 0, 255).astype(np.uint8)
    
    print(f"Output range: {img_8bit.min()} - {img_8bit.max()}")
    
    return img_8bit


def apply_clahe(img_array, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Optional - only use if histogram stretch alone isn't enough.
    
    Args:
        img_array: Input 8-bit image array
        clip_limit: Contrast limit
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image array
    """
    print(f"\nApplying CLAHE (clip={clip_limit}, tile={tile_size})...")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img_array)
    return enhanced


def denoise_image(img_array, strength=10):
    """
    Apply gentle denoising. Only use if image is noisy.
    
    Args:
        img_array: Input image array
        strength: Denoising strength (higher = more smoothing)
        
    Returns:
        Denoised image
    """
    print(f"\nApplying denoising (strength={strength})...")
    denoised = cv2.fastNlMeansDenoising(img_array, h=strength, templateWindowSize=7, searchWindowSize=21)
    return denoised


def sharpen_image(img_array, amount=1.5):
    """
    Apply sharpening to enhance edges.
    
    Args:
        img_array: Input image array
        amount: Sharpening amount (1.0 = no change, higher = more sharp)
        
    Returns:
        Sharpened image
    """
    print(f"\nApplying sharpening (amount={amount})...")
    gaussian = cv2.GaussianBlur(img_array, (5, 5), 1.0)
    sharpened = cv2.addWeighted(img_array, amount, gaussian, -(amount-1), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_microscopic_image(img_array, 
                                 lower_percentile=0, 
                                 upper_percentile=100,
                                 use_clahe=False,
                                 clahe_clip=2.0,
                                 clahe_tile=8,
                                 use_denoise=False,
                                 denoise_strength=10,
                                 use_sharpen=False,
                                 sharpen_amount=1.5):
    """
    Preprocess microscopic images with clean histogram stretching.
    
    Default: Simple histogram stretch (like online tools - no noise!)
    Optional: Add CLAHE, denoising, or sharpening if needed.
    
    Args:
        img_array: Input image array (16-bit or 8-bit)
        lower_percentile: Lower percentile for histogram stretch (0-100)
        upper_percentile: Upper percentile for histogram stretch (0-100)
        use_clahe: Apply CLAHE contrast enhancement
        clahe_clip: CLAHE clip limit
        clahe_tile: CLAHE tile size
        use_denoise: Apply denoising
        denoise_strength: Denoising strength
        use_sharpen: Apply sharpening
        sharpen_amount: Sharpening amount
        
    Returns:
        Preprocessed 8-bit image array
    """
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Histogram stretching (the main operation)
    img_processed = histogram_stretch(img_array, lower_percentile, upper_percentile)
    
    # Optional: CLAHE
    if use_clahe:
        img_processed = apply_clahe(img_processed, clahe_clip, clahe_tile)
    
    # Optional: Denoise
    if use_denoise:
        img_processed = denoise_image(img_processed, denoise_strength)
    
    # Optional: Sharpen
    if use_sharpen:
        img_processed = sharpen_image(img_processed, sharpen_amount)
    
    print("="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    
    return img_processed