def get_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
