from src.preprocessing.video_loader import load_video
from src.preprocessing.frame_extracter import get_frames
from src.preprocessing.face_detector import detect_faces

import cv2

cap=load_video("data/Deepfakes/000_003.mp4")
for frame in get_frames(cap):
    face=detect_faces(frame)

    if face is not None:
        cv2.imshow("Face",face)
        if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()
    