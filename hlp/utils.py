"""Utility functions for HLP (High-Level Policy) module."""

import numpy as np
import random
import cv2


def random_crop(image: np.ndarray, crop_percentage: float = 0.95) -> np.ndarray:
    """
    Randomly crop an image by a percentage and resize back to original size.
    
    Args:
        image: H x W x C (numpy array, uint8 or float)
        crop_percentage: Percentage of original size to crop (default: 0.95)
    
    Returns:
        Cropped and resized image with same shape as input
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * crop_percentage), int(w * crop_percentage)
    if new_h <= 0 or new_w <= 0:
        return image
    top = random.randint(0, max(0, h - new_h))
    left = random.randint(0, max(0, w - new_w))
    cropped = image[top : top + new_h, left : left + new_w, ...]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

