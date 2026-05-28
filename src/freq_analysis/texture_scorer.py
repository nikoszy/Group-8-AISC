# =============================================================================
# src/freq_analysis/texture_scorer.py
# =============================================================================
# Laplacian-variance sharpness score for deepfake detection.
#
# WHY THIS HELPS ON FF++ C23
# --------------------------
# FF++ Deepfakes are produced by an autoencoder: an encoder compresses the
# source face, a decoder reconstructs the target identity.  Every decoder
# layer performs bilinear upsampling followed by convolution, which acts as
# an implicit low-pass filter.  The output face is therefore slightly
# *smoother* than a genuine photograph taken by a camera.
#
# The Laplacian operator ( d²I/dx² + d²I/dy² ) measures second-order
# spatial derivatives — it is large wherever pixel values change quickly
# (edges, fine texture) and near-zero in smooth regions.  Var(Laplacian)
# captures the overall "texture richness" of the image.
#
# Real faces: high Laplacian variance (lots of fine skin texture)
# Deepfakes : lower Laplacian variance (decoder smoothing removes detail)
#
# The score is kept in [0, 1]; the logistic regression in ensemble.py
# will learn the correct sign and magnitude from the data.
# =============================================================================

import cv2
import numpy as np

from src.freq_analysis.utils import resize_to_square


# Empirical calibration constants measured on 224×224 FF++ face crops.
# Real faces: Var(Laplacian) ≈ 300–4000
# Deepfakes : Var(Laplacian) ≈ 150–2500
# Clipping at 3 000 keeps 95 % of samples in [0, 1].
_LAP_CLIP = 3000.0


def laplacian_score(face_img, target_size=224):
    """
    Compute Laplacian-variance sharpness score for a single face image.

    Lower score  =  smoother  =  more suspicious (possible deepfake).
    Higher score =  sharper   =  looks like a genuine photograph.

    Args:
        face_img    : numpy array  (H × W × 3, BGR)
        target_size : resize to this square before scoring (default 224)

    Returns:
        float in [0.0, 1.0], rounded to 4 decimal places.
    """
    img  = resize_to_square(face_img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    var = float(np.var(lap))

    score = float(np.clip(var / _LAP_CLIP, 0.0, 1.0))
    return round(score, 4)
