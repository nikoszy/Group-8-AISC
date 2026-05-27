import cv2

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Minimum Haar cascade level-weight to accept a detection.
# Below this, detections are typically weak edge hits (ears, hair, shoulders).
CONF_THRESHOLD = 0.7

# Bounding-box geometry limits
ASPECT_MIN   = 0.7   # width / height  — rejects very tall/narrow boxes (hair)
ASPECT_MAX   = 1.4   # width / height  — rejects very wide/short boxes (ear strips)
CENTER_Y_MAX = 0.60  # box centre must sit in the top 60 % of the frame height


def _pick_best_face(rects, weights, frame_h):
    """
    From a set of Haar detections return the (x, y, w, h) of the best
    candidate, or None if no detection passes all three checks.

    Selection order:
      1. Drop detections whose level_weight < CONF_THRESHOLD.
      2. Among survivors, pick the one with the largest bounding-box area.
      3. Reject if the box centre falls below the top 60 % of the frame.
      4. Reject if aspect ratio (w/h) is outside [ASPECT_MIN, ASPECT_MAX].
    """
    if len(rects) == 0:
        return None

    # 1. Confidence gate
    confident = [
        (rect, float(wt))
        for rect, wt in zip(rects, weights)
        if float(wt) >= CONF_THRESHOLD
    ]
    if not confident:
        return None

    # 2. Largest area among confident detections
    (x, y, bw, bh), _ = max(confident, key=lambda rw: rw[0][2] * rw[0][3])

    # 3. Centre-Y check — face should be in the upper 60 % of the frame
    if (y + bh / 2) / frame_h > CENTER_Y_MAX:
        return None

    # 4. Aspect-ratio check — box should be roughly square
    if not (ASPECT_MIN <= bw / bh <= ASPECT_MAX):
        return None

    return (x, y, bw, bh)


def detect_faces(frame):
    """
    Detect the most confident frontal face in *frame* and return a crop of it.

    Returns None when:
      - No face is detected at all.
      - All detections fall below CONF_THRESHOLD (confidence gate).
      - The best detection's bounding box fails the geometry checks:
          * centre-Y outside the top 60 % of the frame, or
          * aspect ratio (w/h) outside 0.7 – 1.4.
    """
    frame_h = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects, _, weights = _face_cascade.detectMultiScale3(
        gray, scaleFactor=1.3, minNeighbors=5, outputRejectLevels=True
    )

    best = _pick_best_face(rects, weights, frame_h)
    if best is None:
        return None

    x, y, w, h = best
    return frame[y:y + h, x:x + w]
