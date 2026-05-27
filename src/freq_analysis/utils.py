# =============================================================================
# src/freq_analysis/utils.py
# =============================================================================
# Low-level image helpers used by every other file in this module.
# Each function is kept under 30 lines and independently testable.
# =============================================================================

import cv2
import numpy as np


def resize_to_square(img, size=224):
    """
    Resize any image to a fixed square (size x size).

    WHY THIS MATTERS FOR FFT
    ------------------------
    The FFT converts spatial pixels into frequency components.  If two images
    have different dimensions, their frequency grids are different scales, so
    a "high frequency" pixel in a 400x400 image represents a different spatial
    frequency than the same pixel in a 1024x1024 image.
    By resizing everything to the same square first, all FFT outputs live on
    the same frequency grid and scores are directly comparable.

    Args:
        img  : numpy array  (H x W x C  or  H x W for grayscale)
        size : target side length in pixels (default 224, matches CNN convention)

    Returns:
        numpy array of shape (size, size, C)  or  (size, size)
    """
    # cv2.INTER_AREA downsamples with averaging — best quality for shrinking.
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_face_image(path, target_size=224):
    """
    Load an image from disk, resize it to a fixed square, return BGR array.

    Returns None if the file cannot be read (wrong path, corrupt file, etc.)
    so callers can skip bad files gracefully without crashing.

    Args:
        path        : string — absolute or relative file path
        target_size : side length to resize to (default 224)

    Returns:
        numpy array (target_size x target_size x 3, dtype=uint8)  or  None
    """
    # cv2.imread returns None if the file does not exist or cannot be decoded.
    img = cv2.imread(path)
    if img is None:
        return None
    return resize_to_square(img, target_size)


def normalize_to_uint8(arr):
    """
    Scale a float array to the range 0-255 and cast to uint8.

    Used when we want to display or save an intermediate result (like a
    frequency spectrum) as a standard image file.

    Args:
        arr : numpy float array (any range)

    Returns:
        numpy uint8 array scaled to 0-255
    """
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    # Avoid division by zero for a constant array.
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn) * 255.0
    return scaled.astype(np.uint8)
