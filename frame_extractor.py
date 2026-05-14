# frame_extractor.py
# This file extracts individual image frames from a video file.
# We sample frames at regular intervals rather than every single frame
# because video usually runs at 25-30 frames per second - that's too many
# frames for us to analyze efficiently.

import config   # our settings file


def extract_frames(video_path):
    # Opens a video file and returns a list of (frame, timestamp) pairs.
    # Each "frame" is a numpy array (a grid of pixel colours).
    # Each "timestamp" is a number of seconds into the video.
    # Returns an empty list if OpenCV is not installed.

    try:
        import cv2   # OpenCV - the library for reading video files
    except ImportError:
        # If OpenCV is not installed, tell the user and return nothing
        print("Warning: opencv-python not installed. Cannot extract video frames.")
        print("Install it with: pip install opencv-python")
        return []

    print("Opening video: " + str(video_path))

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    # Check that the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file: " + str(video_path))
        return []

    # Get the frame rate of the video (e.g. 25.0 means 25 frames per second)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Protect against bad video files that report 0 fps
    if video_fps <= 0:
        video_fps = 25.0

    # Calculate how many frames to skip between each sample
    # E.g. if video is 25fps and we want 5fps, we take every 5th frame
    step = max(1, round(video_fps / config.FRAME_SAMPLE_RATE))

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video FPS: " + str(round(video_fps, 1)) +
          " | Sampling every " + str(step) + " frames" +
          " | Total frames: " + str(total_frames))

    frames = []      # this list will hold all our (frame, timestamp) pairs
    frame_idx = 0    # counts which frame we are currently on

    # Loop through the video reading one frame at a time
    while cap.isOpened():

        # Stop if we have enough frames already
        if len(frames) >= config.MAX_FRAMES:
            break

        # Read the next frame from the video
        success, frame = cap.read()

        # If reading failed (end of video), stop the loop
        if not success:
            break

        # Only keep this frame if it's one of our sample frames
        if frame_idx % step == 0:
            # Calculate the time in seconds for this frame
            timestamp = frame_idx / video_fps

            # Add the frame and its timestamp to our list
            frames.append((frame, timestamp))

        frame_idx += 1   # move to the next frame number

    # Always close the video file when done
    cap.release()

    print("Extracted " + str(len(frames)) + " frames from video")
    return frames
