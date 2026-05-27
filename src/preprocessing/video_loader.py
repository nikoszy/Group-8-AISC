# =============================================================================
# src/preprocessing/video_loader.py
# =============================================================================
# Video file loader for Module 1.  Thin wrapper around cv2.VideoCapture that
# raises a clear error instead of silently returning a broken object.
#
# cv2.VideoCapture("nonexistent.mp4") returns an object where isOpened()=False.
# Without this check, the failure would only surface later when cap.read()
# returns (False, None) — confusing to debug.  Raising IOError here makes the
# error message point directly at the bad path.
# =============================================================================

import cv2


def load_video(path):
    """
    Open a video file and return a cv2.VideoCapture ready for reading.

    Args:
        path : str — path to a video file (e.g. "data/FaceForensics++_C23/original/000.mp4").
               Supports any format OpenCV can decode: mp4, avi, mkv, etc.

    Returns:
        cv2.VideoCapture — open capture object.
        Pass to frame_extracter.get_frames() to iterate over frames.
        Call cap.release() when done.

    Raises:
        IOError — if the file does not exist or cannot be decoded by OpenCV.

    Example:
        cap = load_video("video.mp4")
        for frame in get_frames(cap):
            process(frame)
        cap.release()
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    return cap
