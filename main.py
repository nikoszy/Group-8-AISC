from src.preprocessing.video_loader import load_video
from src.preprocessing.frame_extracter import get_frames
from src.preprocessing.face_detector import detect_faces
from download_data import download_data
from src.freq_analysis import frequency_analyzer
import cv2
import os


download_data(
    name="xdxd003/ff-c23",
    out_dir="data",
    unzip=True
)

video_path = None

for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            break
    if video_path:
        break

if video_path is None:
    raise FileNotFoundError("No .mp4 video found in data folder")

print("Using video:", video_path)

cap = load_video(video_path)

for frame in get_frames(cap):
    face = detect_faces(frame)

    if face is not None:
        cv2.imshow("Face", face)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()