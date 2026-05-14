# webcam_capture.py
# Records a live interview from the webcam, analyzes it in the background,
# and if the result is uncertain shows a targeted prompt overlay - all without
# ever closing the webcam window.
#
# The session has three phases, all using the same open webcam:
#   Phase 1: Record the main interview clip
#   Phase 2: Analyze in the background (webcam stays live on screen)
#   Phase 3: If uncertain, show overlay prompt and record the follow-up
#
# Requirements: pip install opencv-python
#
# Usage:
#   python webcam_capture.py
#   python webcam_capture.py --seconds 60
#   python webcam_capture.py --output my_interview.mp4 --seconds 90

import sys      # for reading command line arguments
import time     # for tracking elapsed time
import os       # for checking file paths
import threading  # for running analysis in the background

import pipeline     # the main analysis pipeline
import closed_loop  # for selecting which prompt to show
import config       # our settings


def record_audio_to_file(audio_path, duration_seconds, stop_event):
    # Records microphone audio to a .wav file in a background thread.
    # Runs alongside OpenCV video recording - the two run at the same time.
    # stop_event is a threading.Event - calling stop_event.set() stops early.
    #
    # Why a separate audio file?
    # OpenCV VideoWriter only saves video frames - it cannot record audio.
    # We use sounddevice to capture the microphone in parallel,
    # then pass the .wav file directly to Whisper for transcription.

    try:
        import sounddevice as sd
        import numpy as np
        import wave

        sample_rate = 16000   # 16000 Hz is what Whisper expects
        channels    = 1       # mono audio

        print("Microphone recording started...")

        # Collect audio chunks in a list as they arrive from the microphone
        audio_chunks = []

        def audio_callback(indata, frames, time_info, status):
            # Called automatically by sounddevice for each small chunk of audio
            if not stop_event.is_set():
                audio_chunks.append(indata.copy())

        # Open the microphone stream and wait until stop_event is set
        with sd.InputStream(samplerate=sample_rate, channels=channels,
                            dtype="int16", callback=audio_callback):
            stop_event.wait(timeout=duration_seconds)

        # Write everything collected into a .wav file
        if audio_chunks:
            all_audio = np.concatenate(audio_chunks, axis=0)
            with wave.open(audio_path, "w") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)           # 2 bytes = 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(all_audio.tobytes())
            print("Audio saved: " + audio_path +
                  " (" + str(round(len(all_audio) / sample_rate, 1)) + " seconds)")
        else:
            print("Warning: no audio was captured")

    except ImportError:
        print("Warning: sounddevice not installed - no audio will be recorded.")
        print("Fix with: pip install sounddevice")


def run_analysis_in_background(video_path, audio_path, result_holder):
    # Runs pipeline.analyze() in a background thread so the webcam
    # preview stays live while we wait for the result.
    # audio_path is the separate .wav file recorded by sounddevice.

    print("Background analysis started...")
    result = pipeline.analyze(video_path=video_path, audio_path=audio_path)
    result_holder[0] = result
    print("Background analysis complete.")


def draw_recording_indicator(frame, seconds_left):
    # Draws a small red dot and countdown in the top-left corner.
    # This is the only thing shown during normal recording.
    # Returns the modified frame (preview only - not saved to disk).

    import cv2

    preview = frame.copy()

    # Red filled circle = recording indicator
    cv2.circle(preview, (30, 30), 10, (0, 0, 255), -1)

    # Countdown text
    cv2.putText(
        preview,
        "Recording: " + str(seconds_left) + "s left",
        (50, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return preview


def draw_analyzing_indicator(frame):
    # Draws a small blue dot while background analysis runs.
    # The interviewee sees a normal feed - nothing alarming.
    # Returns the modified frame (preview only - not saved to disk).

    import cv2

    preview = frame.copy()

    # Small blue dot in top-left corner
    cv2.circle(preview, (30, 30), 10, (255, 150, 0), -1)

    cv2.putText(
        preview,
        "Analyzing...",
        (50, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return preview


def draw_gaze_overlay(frame, gaze_model, max_eye_dist_holder):
    # Runs YOLO on a single live frame and draws the iris-ratio gaze overlay.
    # Uses the same logic as measure_frame() so the live display matches analysis.
    #
    # Shows:
    #   - Green boxes around each eye (matching iris.py style)
    #   - Blue dot on each eye centre
    #   - Blue dot on nose
    #   - A horizontal bar showing iris_ratio (needle at centre = focused)
    #   - GREEN status if 0.42 < ratio < 0.58, RED otherwise
    #   - Raw ratio number and confidence on screen
    #
    # max_eye_dist_holder is unused here (kept for signature compatibility).

    import cv2

    display = frame.copy()
    height, width = display.shape[:2]

    results = gaze_model(frame, verbose=False)

    keypoints = None
    for result in results:
        if result.keypoints is None:
            continue
        kp_data = result.keypoints.data
        if kp_data is None or len(kp_data) == 0:
            continue
        best_idx = int(result.boxes.conf.argmax()) if (result.boxes is not None and len(result.boxes.conf) > 0) else 0
        keypoints = kp_data[best_idx].cpu().numpy()
        break

    if keypoints is None:
        cv2.putText(display, "No face detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        return display

    nx, ny, n_conf = float(keypoints[0][0]), float(keypoints[0][1]), float(keypoints[0][2])
    lx, ly, l_conf = float(keypoints[1][0]), float(keypoints[1][1]), float(keypoints[1][2])
    rx, ry, r_conf = float(keypoints[2][0]), float(keypoints[2][1]), float(keypoints[2][2])

    avg_conf = (n_conf + l_conf + r_conf) / 3

    if not all(c > 0.5 for c in [n_conf, l_conf, r_conf]):
        cv2.putText(display, "Low confidence: " + str(round(avg_conf, 2)),
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        return display

    eye_span = abs(lx - rx)
    if eye_span < 1.0:
        return display

    iris_ratio = (nx - min(lx, rx)) / eye_span
    is_focused = 0.42 < iris_ratio < 0.58

    # Draw eye boxes and dots (matching iris.py style)
    for ex, ey in [(lx, ly), (rx, ry)]:
        cv2.rectangle(display, (int(ex) - 15, int(ey) - 10),
                      (int(ex) + 15, int(ey) + 10), (0, 255, 0), 1)   # green box
        cv2.circle(display, (int(ex), int(ey)), 3, (255, 0, 0), -1)   # blue dot

    # Nose dot
    cv2.circle(display, (int(nx), int(ny)), 4, (0, 200, 255), -1)

    # Status text above nose (matching iris.py style)
    status = "Focused" if is_focused else "Looking Away"
    colour = (0, 255, 0) if is_focused else (0, 0, 255)
    cv2.putText(display, status, (int(nx), int(ny) - 30), 1, 1.2, colour, 2)

    # Iris ratio bar at bottom of frame
    # The bar shows the 0-1 range; a needle marks the current ratio
    # The green zone in the centre (0.42-0.58) shows the "focused" band
    bar_x      = 10
    bar_y      = height - 50
    bar_w      = 220
    bar_h      = 16

    # Background bar
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    # Green focused zone
    zone_start = int(bar_x + 0.42 * bar_w)
    zone_end   = int(bar_x + 0.58 * bar_w)
    cv2.rectangle(display, (zone_start, bar_y), (zone_end, bar_y + bar_h), (0, 180, 0), -1)

    # Border
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)

    # Needle showing current iris_ratio
    needle_x = int(bar_x + iris_ratio * bar_w)
    needle_x  = max(bar_x + 1, min(bar_x + bar_w - 1, needle_x))
    cv2.line(display, (needle_x, bar_y - 3), (needle_x, bar_y + bar_h + 3), (255, 255, 255), 2)

    # Labels
    cv2.putText(display, "Iris ratio: " + str(round(iris_ratio, 3)),
                (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "KP conf: " + str(round(avg_conf, 2)),
                (bar_x + bar_w + 8, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return display


def draw_prompt_overlay(frame, message, seconds_left):
    # Draws a semi-transparent prompt banner at the bottom of the frame.
    # Only shown on the INTERVIEWER's screen - not saved to the video file.
    # Returns the modified frame.

    import cv2

    display = frame.copy()
    height, width = display.shape[:2]

    # Draw a dark banner across the bottom
    banner = display.copy()
    cv2.rectangle(banner, (0, height - 90), (width, height), (20, 20, 20), -1)

    # Blend at 70% so the video is still visible behind the banner
    cv2.addWeighted(banner, 0.70, display, 0.30, 0, display)

    # Amber left-edge indicator bar
    cv2.rectangle(display, (0, height - 90), (6, height), (30, 165, 255), -1)

    # The prompt message in white
    cv2.putText(
        display,
        message,
        (18, height - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Countdown in smaller grey text below
    cv2.putText(
        display,
        "(" + str(seconds_left) + "s)",
        (18, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )

    return display


def run_interview_session(output_file, duration_seconds):
    # Runs the full interview session using a single open webcam.
    #
    # Phase 1 - Record: captures frames for duration_seconds, saves to output_file
    # Phase 2 - Analyze: runs analysis in background, shows live feed with indicator
    # Phase 3 - Prompt: if uncertain, shows overlay and records follow-up clip
    # Phase 4 - Merge: re-analyzes follow-up and merges with original result
    #
    # The webcam is opened once at the start and closed once at the very end.
    # Returns the final result dictionary, or None if the webcam failed.

    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is not installed.")
        print("Install it with: pip install opencv-python")
        return None

    # ── Open the webcam once - stays open the whole session ──────────────────

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)   # 0 = default webcam

    if not cap.isOpened():
        print("Error: could not open webcam.")
        print("Make sure your webcam is connected and not used by another app.")
        return None

    # Read the webcam properties
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0   # safe fallback if webcam reports 0

    print("Webcam ready: " + str(frame_width) + "x" + str(frame_height) +
          " at " + str(round(fps, 1)) + " fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # video compression format

    # max_eye_dist_holder[0] tracks the largest eye distance seen so far.
    # It starts at 1.0 and grows each frame - used as the front-facing baseline.
    max_eye_dist_holder = [1.0]
    gaze_model = None
    try:
        from ultralytics import YOLO
        import config as _cfg
        print("Loading gaze model for live overlay...")
        gaze_model = YOLO(_cfg.YOLO_MODEL)
        print("Gaze model ready.")
    except Exception:
        print("Note: YOLO not available, gaze overlay disabled. Install ultralytics to enable.")

    # ── Phase 1: Record the main interview clip ───────────────────────────────

    print("")
    print("Phase 1: Recording interview (" + str(duration_seconds) + " seconds)...")
    print("Press Q to stop early.")
    print("")

    writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        print("Error: could not create video file: " + output_file)
        cap.release()
        return None

    # Audio is saved to a separate .wav file alongside the video.
    # OpenCV cannot record audio, so we use sounddevice in a background thread.
    audio_path  = output_file.replace(".mp4", ".wav")
    audio_stop  = threading.Event()   # setting this tells the audio thread to stop
    audio_thread = threading.Thread(
        target=record_audio_to_file,
        args=(audio_path, duration_seconds + 5, audio_stop)
    )
    audio_thread.start()

    start_time     = time.time()
    last_countdown = -1

    while True:
        elapsed   = time.time() - start_time
        remaining = duration_seconds - elapsed

        if remaining <= 0:
            break

        success, frame = cap.read()
        if not success:
            print("Warning: webcam stopped sending frames.")
            break

        # Save the clean frame to disk (no overlays ever saved to the file)
        writer.write(frame)

        # Print countdown in terminal each second
        seconds_left = int(remaining)
        if seconds_left != last_countdown:
            print("  " + str(seconds_left) + " seconds remaining...")
            last_countdown = seconds_left

        # Show the recording indicator on the preview only.
        # If YOLO is available, also show the live gaze overlay.
        if gaze_model is not None:
            preview = draw_gaze_overlay(frame, gaze_model, max_eye_dist_holder)
        else:
            preview = draw_recording_indicator(frame, seconds_left)

        # Always draw the recording dot and countdown on top
        cv2.circle(preview, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(preview, "REC " + str(seconds_left) + "s",
                    (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Interview - Live", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("Recording stopped early.")
            break

    # Stop the audio thread and wait for the .wav file to finish writing
    audio_stop.set()
    audio_thread.join()

    # Finish writing the main clip - webcam stays open
    writer.release()
    print("")
    print("Main clip saved: " + output_file)
    if os.path.exists(audio_path):
        print("Audio saved:     " + audio_path)

    # ── Phase 2: Analyze in the background ────────────────────────────────────

    print("")
    print("Phase 2: Analyzing in background (webcam stays live)...")

    # result_holder[0] will be filled by the background thread when done
    result_holder = [None]

    # Only pass audio_path to the pipeline if the file was actually written.
    # If sounddevice wasn't installed or recording failed, passing a missing
    # path causes librosa to error silently and return no transcript.
    audio_path_to_use = audio_path if os.path.exists(audio_path) else None
    if audio_path_to_use is None:
        print("Warning: audio file not found (" + audio_path + ") - transcript will be empty")

    # Start analysis in a separate thread so the webcam loop keeps running
    analysis_thread = threading.Thread(
        target=run_analysis_in_background,
        args=(output_file, audio_path_to_use, result_holder)
    )
    analysis_thread.start()

    # Keep showing the live webcam feed while we wait for analysis to finish
    while analysis_thread.is_alive():
        success, frame = cap.read()
        if not success:
            break

        # Show "Analyzing..." indicator on the preview only
        preview = draw_analyzing_indicator(frame)
        cv2.imshow("Interview - Live", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    # Wait for the thread to fully finish before reading the result
    analysis_thread.join()

    result = result_holder[0]

    if result is None:
        print("Analysis failed.")
        cap.release()
        cv2.destroyAllWindows()
        return None

    # Print the result to the terminal
    pipeline.print_result(result)

    # ── Phase 3: Decide whether to prompt ────────────────────────────────────
    # Use should_request_more_evidence() which checks gaze and NLP anomalies
    # BEFORE looking at the decision label.
    # This means 5% eye contact stops here even if decision was 'fake'.

    if not closed_loop.should_request_more_evidence(result, 0):
        # Either confident, or a clear anomaly was detected - no prompt needed.
        # Any clear anomaly findings have already been printed by should_request_more_evidence.
        print("")
        print("No follow-up prompt triggered.")
        cap.release()
        cv2.destroyAllWindows()
        return result

    print("")
    print("Phase 3: Result is uncertain - showing targeted prompt(s)...")

    # Decide which prompt(s) to show based on which signal was weak
    gaze_score = result.get("gaze_score", 0.0)
    nlp_score  = result.get("nlp_score",  0.0)
    prompts    = closed_loop.select_live_prompt(gaze_score, nlp_score)

    # Build filename for the follow-up clip and audio
    base_name          = output_file.replace(".mp4", "")
    followup_file      = base_name + "_followup.mp4"
    followup_audio     = base_name + "_followup.wav"

    # Open a new video writer for the follow-up clip (same open webcam)
    followup_writer = cv2.VideoWriter(
        followup_file, fourcc, fps, (frame_width, frame_height)
    )

    # Start audio recording for the follow-up session
    followup_audio_stop   = threading.Event()
    followup_audio_thread = threading.Thread(
        target=record_audio_to_file,
        args=(followup_audio,
              config.PROMPT_DISPLAY_SECONDS * len(prompts) + 5,
              followup_audio_stop)
    )
    followup_audio_thread.start()

    # Show each prompt in turn, recording the clean feed the whole time
    for prompt in prompts:
        print("Showing prompt [" + prompt["type"] + "]: " + prompt["message"])

        prompt_start = time.time()

        while True:
            elapsed = time.time() - prompt_start
            if elapsed >= config.PROMPT_DISPLAY_SECONDS:
                break

            success, frame = cap.read()
            if not success:
                break

            # Save the clean frame to the follow-up file (no overlay saved)
            followup_writer.write(frame)

            # Draw the prompt overlay on the preview only
            seconds_left = int(config.PROMPT_DISPLAY_SECONDS - elapsed)
            preview = draw_prompt_overlay(frame, prompt["message"], seconds_left)
            cv2.imshow("Interview - Live", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    # Stop audio and wait for the .wav file to finish writing
    followup_audio_stop.set()
    followup_audio_thread.join()

    followup_writer.release()
    print("Follow-up clip saved:  " + followup_file)
    if os.path.exists(followup_audio):
        print("Follow-up audio saved: " + followup_audio)
    else:
        print("Warning: follow-up audio not saved - transcript will be empty")

    # ── Phase 4: Re-analyze follow-up and merge ───────────────────────────────

    print("")
    print("Phase 4: Re-analysing follow-up response...")

    # Only pass the audio path if the file was actually written
    followup_audio_path = followup_audio if os.path.exists(followup_audio) else None

    followup_result = pipeline.analyze(
        video_path = followup_file,
        audio_path = followup_audio_path
    )

    # Print the follow-up result on its own before merging
    print("")
    print("=== Follow-up Response Analysis ===")
    pipeline.print_result(followup_result)

    # Merge original result (40%) with follow-up result (60%)
    merged = closed_loop.merge_results(result, followup_result, iteration=1)

    print("")
    print("=== Merged Final Result (original 40% + follow-up 60%) ===")
    pipeline.print_result(merged)

    # ── Close webcam once at the very end ────────────────────────────────────

    cap.release()
    cv2.destroyAllWindows()
    print("Session complete. Webcam released.")

    return merged


def parse_arguments():
    # Reads command line arguments.
    # Returns a dictionary with "output" and "seconds" keys.

    args = {
        "output":  "interview.mp4",   # default output filename
        "seconds": 30,                # default recording duration
    }

    all_args = sys.argv[1:]   # everything after "webcam_capture.py"

    i = 0
    while i < len(all_args):
        arg = all_args[i]

        if arg == "--output" or arg == "-o":
            if i + 1 < len(all_args):
                args["output"] = all_args[i + 1]
                i += 2
            else:
                print("Error: --output requires a filename")
                sys.exit(1)

        elif arg == "--seconds" or arg == "-s":
            if i + 1 < len(all_args):
                try:
                    args["seconds"] = int(all_args[i + 1])
                except ValueError:
                    print("Error: --seconds must be a whole number")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --seconds requires a number")
                sys.exit(1)

        elif arg == "--help" or arg == "-h":
            print("")
            print("Webcam Interview Recorder")
            print("=========================")
            print("Records, analyzes, and prompts - all in one continuous session.")
            print("")
            print("Usage:")
            print("  python webcam_capture.py")
            print("  python webcam_capture.py --seconds 60")
            print("  python webcam_capture.py --output my_interview.mp4 --seconds 90")
            print("")
            print("Options:")
            print("  --output   filename to save to (default: interview.mp4)")
            print("  --seconds  how long to record in seconds (default: 30)")
            print("  --help     show this message")
            print("")
            sys.exit(0)

        else:
            print("Unknown argument: " + arg)
            print("Run 'python webcam_capture.py --help' for usage")
            sys.exit(1)

    return args


def main():
    print("Webcam Interview Recorder")
    print("=========================")

    args = parse_arguments()

    # Run the full session (record -> analyze -> prompt) in one continuous flow
    result = run_interview_session(
        output_file      = args["output"],
        duration_seconds = args["seconds"]
    )

    if result is None:
        print("Session ended with an error.")
        sys.exit(1)

    # Save a text report alongside the video
    base_name   = args["output"].replace(".mp4", "")
    report_file = base_name + "_report.txt"

    from run import save_report
    save_report(result, report_file)

    # Update the Results Table spreadsheet if it exists
    import save_results_table
    xlsx_path = "Results_Table.xlsx"
    if os.path.exists(xlsx_path):
        save_results_table.save_results_table(result, xlsx_path=xlsx_path)
    else:
        print("Note: Results_Table.xlsx not found in project folder - skipping table update")


# Only run main() if this file is run directly
if __name__ == "__main__":
    main()
