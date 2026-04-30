import cv2

def load_video(path):
    cap=cv2.VideoCapture(path)
    if not cap:
        raise Error("Could not open video")
    return cap    