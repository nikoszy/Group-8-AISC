# ==============================================================================
# artifact_module.py
# ==============================================================================
# This module detects "compression artifacts" in face images.
#
# CORE IDEA:
#   When an AI generates a deepfake image, the pixel values it produces are
#   slightly "unnatural" compared to pixels from a real camera.
#   If we compress those pixels as a JPEG (which throws away small details)
#   and then decompress them, the fake regions show BIGGER unexpected changes
#   than real regions.  We measure those changes to produce a suspicion score.
#
# SCORE MEANING:
#   0.0 = clean / probably real
#   1.0 = heavily artifacted / probably fake
#
# FUNCTIONS IN THIS FILE:
#   1. recompress_frame          — compress and re-expand an image
#   2. get_difference_map        — measure pixel-by-pixel change
#   3. get_artifact_score_for_frame — turn a single frame into a 0-1 score
#   4. compute_artifact_score    — average score across many frames
#   5. visualize_artifacts       — create a side-by-side visual for demos
#   6. batch_score_folder        — score every image in a folder
# ==============================================================================

# --- IMPORTS ---
# These lines load pre-built tools (libraries) that our code will use.

# 'cv2' is OpenCV — a library for working with images and video
import cv2

# 'numpy' (imported as 'np') lets us do fast maths on arrays of numbers
import numpy as np

# 'os' lets us work with files and folders (create paths, list files, etc.)
import os

# 'datetime' lets us record what time things happened
from datetime import datetime


# ==============================================================================
# FUNCTION 1: recompress_frame
# ==============================================================================

def recompress_frame(frame, quality=75):
    """
    Takes an image, squeezes it into a JPEG in memory, then expands it back.

    WHY DO THIS?
    JPEG compression works by throwing away tiny pixel details.
    Real photos have natural-looking details that compress predictably.
    Deepfake images often have unnatural patterns that JPEG handles differently,
    leaving visible "damage" when compared to the original.

    Inputs:
        frame   : numpy array — the image (height x width x 3 colours)
        quality : integer 0-100 — how aggressively to compress
                  75 = moderate compression, good at exposing artifacts
                  (too high = no damage; too low = everything looks bad)

    Returns:
        numpy array — the recompressed image, same shape as input
    """

    # imencode converts the numpy array image into a JPEG file stored in memory.
    # It does NOT write to disk — it just produces a compressed byte sequence.
    # The first return value is True/False (success flag), second is the bytes.
    # [cv2.IMWRITE_JPEG_QUALITY, quality] tells OpenCV what JPEG quality to use.
    success, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # If the compression failed for any reason, return the original unchanged.
    if not success:
        return frame

    # imdecode turns the JPEG bytes back into a numpy array (decompresses it).
    # cv2.IMREAD_COLOR means load as a full colour (BGR) image.
    recompressed = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)

    # If decoding somehow failed, return the original unchanged.
    if recompressed is None:
        return frame

    # Return the image that has been through one round of JPEG compression.
    return recompressed


# ==============================================================================
# FUNCTION 2: get_difference_map
# ==============================================================================

def get_difference_map(original, recompressed):
    """
    Computes how much each pixel changed after recompression.

    MATHEMATICS:
        For every pixel position (x, y):
            difference = | original_pixel - recompressed_pixel |
        The vertical bars mean "absolute value" — we only care about
        the SIZE of the difference, not the direction (positive or negative).

    WHY GRAYSCALE?
        A colour image has 3 values per pixel (Red, Green, Blue).
        Converting to grayscale gives us 1 value per pixel.
        One number is easier to analyse and display.

    HIGH VALUES = big change = suspicious region (likely fake)
    LOW  VALUES = small change = clean region    (likely real)

    Inputs:
        original     : numpy array — the original image
        recompressed : numpy array — the image after JPEG round-trip

    Returns:
        numpy array — grayscale difference map, same height/width as input
    """

    # cv2.absdiff computes abs(original - recompressed) for every pixel.
    # This gives us a colour difference image — 3 values per pixel.
    diff_colour = cv2.absdiff(original, recompressed)

    # cv2.cvtColor converts the colour difference to grayscale.
    # Grayscale collapses 3 colour channels into 1 brightness value per pixel.
    # COLOR_BGR2GRAY uses the standard brightness formula:
    #   gray = 0.114*Blue + 0.587*Green + 0.299*Red
    diff_gray = cv2.cvtColor(diff_colour, cv2.COLOR_BGR2GRAY)

    # Return the single-channel grayscale difference map.
    return diff_gray


# ==============================================================================
# FUNCTION 3: get_artifact_score_for_frame
# ==============================================================================

def get_artifact_score_for_frame(frame):
    """
    Scores a single image frame on a scale from 0.0 to 1.0.

    0.0 means the image survived JPEG recompression with tiny changes -> clean
    1.0 means huge changes after recompression -> likely deepfake artifacts

    Input:
        frame : numpy array — the image to score

    Returns:
        float — score between 0.0 and 1.0 (rounded to 3 decimal places)
    """

    # Step 1: Recompress the frame using JPEG quality 75.
    recompressed = recompress_frame(frame)

    # Step 2: Compute the per-pixel difference between original and recompressed.
    difference_map = get_difference_map(frame, recompressed)

    # Step 3: Compute the MEAN (average) of all pixel differences.
    # np.mean adds up all values and divides by the total number of pixels.
    # Result is a single number representing the "average damage" level.
    mean_diff = np.mean(difference_map)

    # Step 4: Normalise to 0.0 – 1.0.
    # In practice, JPEG difference values range roughly 0 to 30.
    # Dividing by 30 maps that range to 0.0 – 1.0.
    score = mean_diff / 30.0

    # Step 5: Clip the score so it never goes above 1.0.
    # min(score, 1.0) returns whichever is smaller.
    score = min(score, 1.0)

    # Step 6: Round to 3 decimal places so the number is readable.
    score = round(score, 3)

    # Return the final suspicion score for this frame.
    return score


# ==============================================================================
# FUNCTION 4: compute_artifact_score
# ==============================================================================

def compute_artifact_score(frame_paths, sample_n=20):
    """
    This is the MAIN function that main.py will call.

    Takes a list of image file paths, analyses a sample of them,
    and returns one number representing how artifacted the overall video is.

    WHY SAMPLE INSTEAD OF USING ALL FRAMES?
        A video can have thousands of frames. Analysing all of them would
        take too long. Sampling evenly spreads our analysis across the video,
        which gives a fair estimate without the wait.

    Inputs:
        frame_paths : list of strings — file paths to the frames
        sample_n    : integer — how many frames to actually analyse (default 20)

    Returns:
        float — average artifact score across all sampled frames (0.0 to 1.0)
    """

    # Tell the user what we are about to do.
    print(f"[compute_artifact_score] Received {len(frame_paths)} frame paths")

    # --- Decide which frames to sample ---

    # If we have more frames than sample_n, we pick frames evenly spaced.
    if len(frame_paths) > sample_n:

        # Calculate how many frames to skip between each sample.
        # For example, if we have 100 frames and want 20:  step = 100/20 = 5
        step = len(frame_paths) / sample_n

        # Build a list of indices using a step — [0, 5, 10, 15, ...]
        # int(i * step) converts the floating-point step to a whole number index.
        sampled_paths = [frame_paths[int(i * step)] for i in range(sample_n)]

    else:
        # If we have fewer frames than sample_n, just use all of them.
        sampled_paths = frame_paths

    # Tell the user how many frames we will actually analyse.
    print(f"[compute_artifact_score] Sampling {len(sampled_paths)} frames")

    # --- Score each sampled frame ---

    # This list will collect the score for each frame.
    scores = []

    # Loop through each sampled frame path.
    for path in sampled_paths:

        # Try loading and scoring each frame.
        try:

            # cv2.imread loads the image from disk into a numpy array.
            frame = cv2.imread(path)

            # If the file could not be read (wrong path, corrupt file), skip it.
            if frame is None:
                print(f"  [WARN] Could not load: {path} — skipping")
                continue

            # Compute the artifact score for this single frame.
            score = get_artifact_score_for_frame(frame)

            # Add this score to our collection.
            scores.append(score)

        # If something unexpected goes wrong, warn and skip.
        except Exception as e:
            print(f"  [WARN] Error on {path}: {e} — skipping")

    # --- Summarise ---

    # If we could not score any frames at all, return 0.0 as a safe default.
    if len(scores) == 0:
        print("[compute_artifact_score] WARNING: No frames were scored. Returning 0.0")
        return 0.0

    # Compute the average score across all frames we analysed.
    average_score = float(np.mean(scores))

    # Round to 3 decimal places.
    average_score = round(average_score, 3)

    # Decide whether this looks suspicious.
    # 0.5 is our threshold — above it we flag as suspicious.
    if average_score >= 0.5:
        verdict = "SUSPICIOUS (possible deepfake)"
    else:
        verdict = "CLEAN (probably real)"

    # Print a human-readable summary.
    print(f"[compute_artifact_score] Frames analysed : {len(scores)}")
    print(f"[compute_artifact_score] Average score   : {average_score}")
    print(f"[compute_artifact_score] Verdict         : {verdict}")

    # Return the single average score.
    return average_score


# ==============================================================================
# FUNCTION 5: visualize_artifacts
# ==============================================================================

def visualize_artifacts(frame_path, save_path=None):
    """
    Creates a side-by-side image showing:
        Panel 1: ORIGINAL   — the untouched input frame
        Panel 2: RECOMPRESSED — the frame after JPEG round-trip
        Panel 3: DIFFERENCE x10 — the pixel differences, magnified 10x

    This is useful for DEMOS: you can SEE where the deepfake artifacts are.
    Bright spots in the third panel = where the AI left "fingerprints".

    Inputs:
        frame_path : string — file path to the image
        save_path  : string or None — where to save the output image
                     If None, the image is shown on screen instead.

    Returns:
        Nothing (saves or displays the image)
    """

    # Load the image from disk.
    frame = cv2.imread(frame_path)

    # If loading failed, print an error and return early.
    if frame is None:
        print(f"[visualize_artifacts] ERROR: Could not load image: {frame_path}")
        return

    # Get the recompressed version using our recompress_frame function.
    recompressed = recompress_frame(frame)

    # Get the grayscale difference map.
    diff_gray = get_difference_map(frame, recompressed)

    # --- Prepare the three panels ---

    # Panel 1 is the original, which is already a colour (BGR) image.
    panel_original = frame

    # Panel 2 is the recompressed version, also already colour.
    panel_recompressed = recompressed

    # Panel 3 is the difference map, but we need it in colour (3 channels)
    # so it can sit next to the colour panels.
    # First, multiply by 10 so the differences are big enough to see.
    # Without this multiplication, most differences would look like black.
    diff_visible = diff_gray * 10

    # Clip values to 255 (the maximum brightness for an 8-bit image).
    # Without this, values above 255 would "wrap around" and look wrong.
    diff_visible = np.clip(diff_visible, 0, 255).astype(np.uint8)

    # Convert the grayscale difference to a 3-channel colour image so we can
    # stack it with the other two colour panels.
    # GRAY2BGR just copies the single channel into all three (R=G=B -> grey).
    diff_colour = cv2.cvtColor(diff_visible, cv2.COLOR_GRAY2BGR)

    # --- Make all three panels the same height ---
    # We take the smallest height among the three so nothing gets cut off.

    # Get the height of each panel (index 0 = rows = height).
    h1 = panel_original.shape[0]
    h2 = panel_recompressed.shape[0]
    h3 = diff_colour.shape[0]

    # Use the smallest height as the target.
    target_h = min(h1, h2, h3)

    # Get the width of each panel (index 1 = columns = width).
    w1 = panel_original.shape[1]
    w2 = panel_recompressed.shape[1]
    w3 = diff_colour.shape[1]

    # Resize each panel to (target_h x original_width) if needed.
    # cv2.resize expects (width, height) not (height, width).
    if panel_original.shape[0] != target_h:
        panel_original = cv2.resize(panel_original, (w1, target_h))

    if panel_recompressed.shape[0] != target_h:
        panel_recompressed = cv2.resize(panel_recompressed, (w2, target_h))

    if diff_colour.shape[0] != target_h:
        diff_colour = cv2.resize(diff_colour, (w3, target_h))

    # --- Add text labels to each panel ---

    # Font settings for the text labels.
    # cv2.FONT_HERSHEY_SIMPLEX is a clean, simple font.
    font       = cv2.FONT_HERSHEY_SIMPLEX

    # Font scale 0.7 means roughly 70% of the default font size.
    font_scale = 0.7

    # Thickness of the text stroke in pixels.
    thickness  = 2

    # White colour in BGR format (B=255, G=255, R=255).
    colour     = (255, 255, 255)

    # Position for the text: 10 pixels from the left, 30 from the top.
    text_pos   = (10, 30)

    # Write "ORIGINAL" on the first panel.
    cv2.putText(panel_original,    "ORIGINAL",        text_pos, font, font_scale, colour, thickness)

    # Write "RECOMPRESSED" on the second panel.
    cv2.putText(panel_recompressed,"RECOMPRESSED",     text_pos, font, font_scale, colour, thickness)

    # Write "DIFFERENCE x10" on the third panel.
    cv2.putText(diff_colour,       "DIFFERENCE x10",  text_pos, font, font_scale, colour, thickness)

    # --- Combine the three panels side by side ---

    # np.hstack stacks arrays horizontally (left to right).
    # All three must have the same height for this to work.
    combined = np.hstack([panel_original, panel_recompressed, diff_colour])

    # --- Save or display ---

    # If save_path was provided, write the combined image to disk.
    if save_path is not None:

        # Make sure the output folder exists.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # cv2.imwrite saves the image as a file.
        cv2.imwrite(save_path, combined)

        # Confirm to the user.
        print(f"[visualize_artifacts] Saved to: {save_path}")

    else:

        # cv2.imshow opens a window and displays the image.
        cv2.imshow("Artifact Visualisation", combined)

        # cv2.waitKey(0) pauses until the user presses any key.
        cv2.waitKey(0)

        # Closes all OpenCV windows after the key press.
        cv2.destroyAllWindows()


# ==============================================================================
# FUNCTION 6: batch_score_folder
# ==============================================================================

def batch_score_folder(folder_path, label):
    """
    Scores every JPEG image in a folder and returns a list of results.

    This is used to build training data for the machine learning model.
    After running this on both the real folder and fake folder, you have
    a dataset of (image, score, label) rows that the model can learn from.

    Inputs:
        folder_path : string — path to the folder containing JPEG images
        label       : string — "real" or "fake" (used to label each result)

    Returns:
        list of dicts — each dict looks like:
            {"file": "path/to/image.jpg", "score": 0.34, "label": "real"}
    """

    # Tell the user which folder we are processing.
    print(f"\n[batch_score_folder] Folder : {folder_path}")
    print(f"[batch_score_folder] Label  : {label}")

    # This list will hold all the result dicts.
    results = []

    # --- Get the list of JPEG files in the folder ---

    # Try to list files; if the folder doesn't exist, print an error.
    try:

        # os.listdir returns all file names in the folder.
        all_files = os.listdir(folder_path)

    # If the folder is missing, catch the error.
    except FileNotFoundError:
        print(f"[batch_score_folder] ERROR: Folder not found: {folder_path}")
        return results

    # Filter to only .jpg files (ignore any other file types in the folder).
    jpg_files = [f for f in all_files if f.lower().endswith(".jpg")]

    # Sort alphabetically so we process them in a consistent order.
    jpg_files = sorted(jpg_files)

    # Tell the user how many images we found.
    print(f"[batch_score_folder] Found {len(jpg_files)} JPEG files")

    # If there are no images, return an empty list.
    if len(jpg_files) == 0:
        print(f"[batch_score_folder] WARNING: No .jpg files found in {folder_path}")
        return results

    # --- Score each image ---

    # Loop through every file, with i as the position number (0, 1, 2, ...)
    for i, filename in enumerate(jpg_files):

        # Build the full file path by joining folder path and file name.
        full_path = os.path.join(folder_path, filename)

        # Try to load and score the image.
        try:

            # cv2.imread loads the image into a numpy array.
            frame = cv2.imread(full_path)

            # If loading failed, skip this file.
            if frame is None:
                print(f"  [SKIP] {filename} — could not load")
                continue

            # Score this single frame.
            score = get_artifact_score_for_frame(frame)

            # Add a result dict to our list.
            results.append({
                "file"  : full_path,
                "score" : score,
                "label" : label
            })

            # Print progress every 10 files (so the terminal is not too busy).
            # The '%' symbol is the modulo operator — i % 10 == 0 means
            # "every time i is exactly divisible by 10".
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1} / {len(jpg_files)} files processed")

        # Catch any unexpected error.
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    # --- Print summary ---

    # If we scored at least one image, compute the average.
    if results:

        # Extract just the score values from each result dict.
        all_scores = [r["score"] for r in results]

        # np.mean computes the average.
        avg = round(float(np.mean(all_scores)), 3)

        # Print the summary line.
        print(f"[batch_score_folder] Folder: {folder_path} | Label: {label} | Avg score: {avg}")

    # Return the full list of result dicts.
    return results


# ==============================================================================
# PART 3 — TEST BLOCK
# Run this file directly to test all functions and verify everything works.
# ==============================================================================

if __name__ == '__main__':

    # Print a header banner.
    print()
    print("=" * 50)
    print("ARTIFACT MODULE — RUNNING SELF-TEST")
    print("=" * 50)
    print()

    # -----------------------------------------------------------------------
    # Step 1: Score all REAL frames
    # -----------------------------------------------------------------------

    # Tell the user what Step 1 is doing.
    print("STEP 1: Scoring real frames...")

    # Call batch_score_folder on the real frames folder.
    # 'real' is the label we assign to every result from this folder.
    real_results = batch_score_folder('data/real/frames/', 'real')

    # Tell the user how many real frames were scored.
    print(f"  Real frames scored: {len(real_results)}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Score all FAKE frames
    # -----------------------------------------------------------------------

    # Tell the user what Step 2 is doing.
    print("STEP 2: Scoring fake frames...")

    # Call batch_score_folder on the fake frames folder.
    fake_results = batch_score_folder('data/fake/frames/', 'fake')

    # Tell the user how many fake frames were scored.
    print(f"  Fake frames scored: {len(fake_results)}")
    print()

    # -----------------------------------------------------------------------
    # Step 3: Print comparison
    # -----------------------------------------------------------------------

    # Tell the user what Step 3 is doing.
    print("STEP 3: Comparing real vs fake scores...")
    print()

    # Compute the average real score.
    # If there are no real results, use 0.0 as a safe fallback.
    if real_results:
        avg_real = round(float(np.mean([r["score"] for r in real_results])), 3)
    else:
        avg_real = 0.0

    # Compute the average fake score.
    if fake_results:
        avg_fake = round(float(np.mean([r["score"] for r in fake_results])), 3)
    else:
        avg_fake = 0.0

    # Print both averages so the user can compare them.
    print(f"  Average REAL score : {avg_real}")
    print(f"  Average FAKE score : {avg_fake}")
    print()

    # Check whether fake average is higher than real average (as expected).
    # A well-working artifact detector should give higher scores to fake images.
    fake_higher = avg_fake > avg_real

    # Print whether the relationship holds.
    print(f"  Fake average > Real average? : {'YES' if fake_higher else 'NO'}")
    print()

    # Determine if the module is working correctly.
    # "Working correctly" means fakes score higher than reals.
    module_ok = "YES" if fake_higher else "NO (scores may still be useful — check visualizations)"

    # Print the verdict.
    print(f"  Module working correctly: {module_ok}")
    print()

    # -----------------------------------------------------------------------
    # Step 4: Create visualizations
    # -----------------------------------------------------------------------

    # Tell the user what Step 4 is doing.
    print("STEP 4: Creating visualizations...")

    # Define the folder where we will save the visualizations.
    VIZ_DIR = "data/visualizations"

    # Create the folder if it does not already exist.
    # exist_ok=True means no error if it already exists.
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Try to create a visualization from the first real frame.
    try:

        # Get the full path to the first real frame.
        # sorted() ensures we always pick the same file (alphabetically first).
        first_real_files = sorted([
            f for f in os.listdir('data/real/frames/') if f.endswith('.jpg')
        ])

        # Only proceed if there are real files.
        if first_real_files:

            # Build the full path to the first real file.
            first_real_path = os.path.join('data/real/frames/', first_real_files[0])

            # Build the output save path.
            real_viz_path = os.path.join(VIZ_DIR, 'real_example.jpg')

            # Call visualize_artifacts to create and save the side-by-side image.
            visualize_artifacts(first_real_path, save_path=real_viz_path)

        else:
            print("  [WARN] No real frames found to visualize")

    # Catch any error during visualization.
    except Exception as e:
        print(f"  [ERROR] Could not visualize real frame: {e}")

    # Try to create a visualization from the first fake frame.
    try:

        # Get the list of fake frame files.
        first_fake_files = sorted([
            f for f in os.listdir('data/fake/frames/') if f.endswith('.jpg')
        ])

        # Only proceed if there are fake files.
        if first_fake_files:

            # Build the full path to the first fake file.
            first_fake_path = os.path.join('data/fake/frames/', first_fake_files[0])

            # Build the output save path.
            fake_viz_path = os.path.join(VIZ_DIR, 'fake_example.jpg')

            # Call visualize_artifacts to create and save the side-by-side image.
            visualize_artifacts(first_fake_path, save_path=fake_viz_path)

        else:
            print("  [WARN] No fake frames found to visualize")

    # Catch any error during visualization.
    except Exception as e:
        print(f"  [ERROR] Could not visualize fake frame: {e}")

    # Tell the user where the visualizations were saved.
    print(f"  Visualizations saved to: {VIZ_DIR}/")
    print()

    # -----------------------------------------------------------------------
    # Step 5: Print final summary
    # -----------------------------------------------------------------------

    # Print the complete summary block.
    print("=" * 50)
    print("ARTIFACT MODULE TEST COMPLETE")
    print("=" * 50)
    print(f"Real frames scored    : {len(real_results)}")
    print(f"Fake frames scored    : {len(fake_results)}")
    print(f"Average real score    : {avg_real}")
    print(f"Average fake score    : {avg_fake}")
    print(f"Fake > Real?          : {'YES' if fake_higher else 'NO'}")
    print(f"Visualizations saved  : {VIZ_DIR}/")
    print("=" * 50)
    print()

    # -----------------------------------------------------------------------
    # NEXT STEPS
    # -----------------------------------------------------------------------

    # Tell the user exactly what to do next and in what order.
    print("NEXT STEPS:")
    print("  1. python artifact_module.py        — tests the artifact module  (you are here)")
    print("  2. python generate_training_data.py — generates the ML training CSV")
    print("  3. Check data/visualizations/       — see what artifacts look like visually")
