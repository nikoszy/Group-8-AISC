# =============================================================================
# src/preprocessing/frame_extracter.py
# =============================================================================
# Frame generator for Module 1.  Yields frames one at a time from an open
# cv2.VideoCapture object.
#
# WHY A GENERATOR?
# ----------------
# Loading all frames into memory at once for a long video would use gigabytes
# of RAM.  A generator yields one frame at a time — the caller processes each
# frame and then it can be garbage-collected.  This keeps memory usage flat
# regardless of video length.
# =============================================================================


def get_frames(cap):
    """
    Yield every frame from an open cv2.VideoCapture, then stop.

    Usage:
        cap = cv2.VideoCapture("video.mp4")
        for frame in get_frames(cap):
            # frame is a (H × W × 3) BGR numpy array
            process(frame)
        cap.release()

    Args:
        cap : cv2.VideoCapture — an already-opened video capture object.
              Use video_loader.load_video() to create one.

    Yields:
        numpy array (H × W × 3, BGR) — one frame per iteration.

    Notes:
        - Does NOT call cap.release() — the caller is responsible for cleanup.
        - Stops automatically when the video ends (cap.read() returns ret=False).
        - Raises no exceptions on a depleted or closed capture; it simply stops.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
