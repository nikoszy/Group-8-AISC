<<<<<<< HEAD
# Video Frame and Audio Extraction Tool

This tool extracts frames and audio from video files in the DeepFake Detection (DFD) dataset for further analysis.

## Features

- Extract frames from videos at a specified frame rate
- Extract audio tracks from videos
- Process entire directories of videos
- Organized output structure for easy analysis

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to process all videos in the dataset:

```bash
python main.py
```

This will:
- Process all MP4 files in `data/DFD_manipulated_sequences/DFD_manipulated_sequences/`
- Extract 1 frame per second from each video
- Extract audio from each video
- Save results to `data/processed/`

### Custom Usage

You can also use the extraction functions directly in your code:

```python
from preprocessing.extract import extract_frames_from_videos

# Extract frames and audio from videos
results = extract_frames_from_videos(
    video_dir="path/to/videos",
    output_base_dir="path/to/output",
    frame_rate=30  # frames per second to extract
)

# Results is a dict mapping video paths to extracted data
for video_path, data in results.items():
    print(f"Video: {video_path}")
    print(f"Frames: {len(data['frames'])} files")
    print(f"Audio: {data['audio']}")
```

### Individual Video Processing

```python
from preprocessing.extract import extract_frames_from_video, extract_audio_from_video

# Extract frames from a single video
frame_paths = extract_frames_from_video(
    video_path="path/to/video.mp4",
    output_dir="path/to/frames",
    frame_rate=30
)

# Extract audio from a single video
audio_path = extract_audio_from_video(
    video_path="path/to/video.mp4",
    output_dir="path/to/audio"
)
```

## Output Structure

```
data/processed/
├── frames/
│   ├── video_name_1/
│   │   ├── video_name_1_frame_000000.jpg
│   │   ├── video_name_1_frame_000001.jpg
│   │   └── ...
│   └── video_name_2/
│       └── ...
└── audio/
    ├── video_name_1/
    │   └── video_name_1_audio.wav
    └── video_name_2/
        └── ...
```

## Parameters

- `frame_rate`: Number of frames to extract per second (default: 30)
- `video_dir`: Directory containing input video files
- `output_base_dir`: Directory to save extracted frames and audio

## Requirements

- Python 3.7+
- OpenCV for frame extraction
- MoviePy for audio extraction
- FFmpeg (automatically installed with MoviePy)

## Notes

- Videos without audio will still have frames extracted
- Frame extraction uses JPEG format for smaller file sizes
- Audio is extracted as WAV format
- Processing large datasets may take significant time
=======
# Group-8-AISC
>>>>>>> fac26bdd39f775fec1cb50be4fb034701e147c7f
