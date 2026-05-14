# gaze_analyzer.py
# This file analyzes eye gaze from video frames.
# It uses YOLO Pose to detect face landmarks (eyes, nose, ears)
# and then calculates features like eye contact ratio and blink rate.
#
# YOLO Pose returns "keypoints" - specific points on the body.
# We care about these keypoint indices (COCO format):
#   0 = nose
#   1 = left eye
#   2 = right eye
#   3 = left ear
#   4 = right ear

import math     # math functions like sqrt and distance
import config   # our settings


# Names for the keypoint index numbers - makes code easier to read
NOSE      = 0
LEFT_EYE  = 1
RIGHT_EYE = 2


def analyze_gaze(frames):
    # Main function: takes a list of (frame, timestamp) pairs,
    # runs YOLO Pose on each frame, and returns a dictionary of gaze features.
    # Falls back to a simple heuristic if YOLO is not installed.

    if len(frames) == 0:
        # No frames to analyze - return neutral scores
        print("No frames to analyze, returning neutral gaze features")
        return make_neutral_gaze_features(0)

    # Try to use the real YOLO model
    model = load_yolo_model()

    if model is not None:
        # YOLO is available - use it for accurate results
        return analyze_with_yolo(frames, model)
    else:
        # YOLO not available - use a simpler fallback method
        print("Using simple heuristic gaze estimator (install ultralytics for better results)")
        return analyze_with_heuristic(frames)


def load_yolo_model():
    # Try to load the YOLO Pose model.
    # Returns the model if successful, or None if ultralytics is not installed.

    try:
        from ultralytics import YOLO   # the YOLO library
        print("Loading YOLO Pose model...")
        model = YOLO(config.YOLO_MODEL)
        print("YOLO model loaded successfully")
        return model
    except ImportError:
        # ultralytics is not installed
        print("Note: ultralytics not installed. Install with: pip install ultralytics")
        return None


def analyze_with_yolo(frames, model):
    # Runs YOLO Pose on each frame and collects per-frame measurements.
    # Then averages them into a single set of gaze features.

    per_frame_data = []   # list to collect measurements from each frame

    for frame, timestamp in frames:
        # Run YOLO on this frame (verbose=False stops it printing for every frame)
        results = model(frame, verbose=False)

        # Extract the face keypoints from the YOLO results
        keypoints = get_keypoints_from_results(results)

        if keypoints is not None:
            # Calculate gaze measurements for this frame
            measurements = measure_frame(keypoints)

            # measure_frame returns None if keypoint confidence is too low
            if measurements is not None:
                measurements["timestamp"] = timestamp
                per_frame_data.append(measurements)

    # Combine all per-frame data into summary features
    return aggregate_frame_data(per_frame_data, len(frames))


def get_keypoints_from_results(results):
    # Pulls out the face keypoints from YOLO's output.
    # YOLO can detect multiple people - we pick the most prominent one.
    # Returns a list of [x, y, confidence] for each keypoint, or None if no face found.

    for result in results:
        # Check if YOLO found any keypoints
        if result.keypoints is None:
            continue

        keypoint_data = result.keypoints.data

        if keypoint_data is None or len(keypoint_data) == 0:
            continue

        # If multiple people detected, pick the one YOLO is most confident about
        if result.boxes is not None:
            confidences = result.boxes.conf
            best_person_idx = int(confidences.argmax())   # index of highest confidence
        else:
            best_person_idx = 0   # just use the first person if no confidence info

        # Convert to a regular Python list (from a PyTorch tensor)
        kps = keypoint_data[best_person_idx].cpu().numpy()
        return kps   # shape: (17 keypoints, 3 values each: x, y, confidence)

    return None   # no person found in this frame


def measure_frame(keypoints):
    # Given keypoints for one frame, calculate gaze measurements.
    # Returns a dictionary of numbers, or None if confidence is too low.
    #
    # HOW GAZE IS MEASURED (based on iris.py approach)
    # ──────────────────────────────────────────────────
    # YOLO gives us the nose (0), left eye (1), and right eye (2) positions.
    # We cannot track the eyeball itself, but we can measure where the NOSE
    # sits relative to the two eyes horizontally.
    #
    # When someone faces the camera:
    #   The nose is centred between the two eyes → iris_ratio ≈ 0.5
    #
    # When someone turns their head to the side:
    #   The nose shifts toward one eye and away from the other → ratio moves toward 0 or 1
    #
    # Formula (from iris.py):
    #   eye_span    = horizontal distance between left and right eye
    #   iris_ratio  = (nose_x - leftmost_eye_x) / eye_span
    #
    # A ratio between 0.42 and 0.58 means the nose is centred → focused on camera.
    # Outside that range means the head is turned away.
    #
    # This correctly detects a side profile because when fully turned,
    # the nose shifts far outside the 0.42–0.58 band.

    left_eye  = keypoints[LEFT_EYE]    # [x, y, confidence]
    right_eye = keypoints[RIGHT_EYE]
    nose      = keypoints[NOSE]

    nx, ny, n_conf = float(nose[0]),      float(nose[1]),      float(nose[2])
    lx, ly, l_conf = float(left_eye[0]),  float(left_eye[1]),  float(left_eye[2])
    rx, ry, r_conf = float(right_eye[0]), float(right_eye[1]), float(right_eye[2])

    # Skip this frame if any of the three keypoints have low confidence
    # (matches the iris.py threshold of > 0.5)
    if not all(c > 0.5 for c in [n_conf, l_conf, r_conf]):
        return None

    # Horizontal span between the two eyes (always positive)
    eye_span = abs(lx - rx)
    if eye_span < 1.0:
        return None   # eyes are on top of each other - bad detection

    # Where does the nose sit between the two eyes? (0.0 = left edge, 1.0 = right edge)
    iris_ratio = (nx - min(lx, rx)) / eye_span

    # Is the person looking at the camera?
    # Focused range: 0.42 to 0.58 (nose is near centre between the eyes)
    is_focused = 0.42 < iris_ratio < 0.58

    # Eye openness - used for blink detection
    eye_distance = math.dist((lx, ly), (rx, ry))
    eye_distance = max(eye_distance, 1.0)
    left_openness  = abs(ly - ny) / eye_distance
    right_openness = abs(ry - ny) / eye_distance
    avg_openness   = (left_openness + right_openness) / 2

    return {
        "iris_ratio":    iris_ratio,          # 0.5 = centred, moves toward 0 or 1 when turned
        "is_focused":    is_focused,          # True if nose is centred between eyes
        "eye_span":      eye_span,            # horizontal pixel distance between eyes
        "eye_openness":  avg_openness,        # for blink detection
        "kp_confidence": (n_conf + l_conf + r_conf) / 3,
        "left_eye_x":    lx,                  # for saccade calculation
        "eye_mid_x":     (lx + rx) / 2,      # for display
        "eye_mid_y":     (ly + ry) / 2,      # for display
        "nose_x":        nx,                  # for display
        "nose_y":        ny,                  # for display
        "left_eye_pos":  (lx, ly),           # for display
        "right_eye_pos": (rx, ry),           # for display
    }


def aggregate_frame_data(per_frame_data, total_frames):
    # Takes the list of per-frame measurements and computes summary gaze features.

    if len(per_frame_data) == 0:
        print("Warning: no faces detected in any frame")
        return make_neutral_gaze_features(0)

    all_iris_ratios = [f["iris_ratio"]    for f in per_frame_data]
    all_focused     = [f["is_focused"]    for f in per_frame_data]
    all_openness    = [f["eye_openness"]  for f in per_frame_data]
    all_left_eye_x  = [f["left_eye_x"]   for f in per_frame_data]
    all_confidences = [f["kp_confidence"] for f in per_frame_data]

    # ── Eye contact ratio ─────────────────────────────────────────────────────
    # Fraction of frames where the nose was centred between the eyes (0.42–0.58)
    # A side profile will have a ratio outside this band and will NOT be counted.
    frames_focused   = sum(1 for f in all_focused if f)
    eye_contact_ratio = frames_focused / len(all_focused)

    # ── Average gaze deviation ────────────────────────────────────────────────
    # How far from centre (0.5) is the iris_ratio on average?
    # 0.0 = always centred, 0.5 = always at the edge
    avg_deviation = sum(abs(r - 0.5) for r in all_iris_ratios) / len(all_iris_ratios)

    # ── Gaze consistency ──────────────────────────────────────────────────────
    # How much does the iris_ratio vary? Low variance = steady gaze.
    avg_ratio = sum(all_iris_ratios) / len(all_iris_ratios)
    variance  = sum((r - avg_ratio) ** 2 for r in all_iris_ratios) / len(all_iris_ratios)
    std_dev   = math.sqrt(variance)
    gaze_consistency = max(0.0, 1.0 - min(std_dev * 4, 1.0))

    # ── Blink detection ───────────────────────────────────────────────────────
    blink_count = 0
    for i in range(1, len(all_openness)):
        if all_openness[i] - all_openness[i - 1] < -0.3:
            blink_count += 1
    duration_seconds = max(len(per_frame_data), 1)
    blink_rate = (blink_count / duration_seconds) * 60

    # ── Saccade velocity ──────────────────────────────────────────────────────
    if len(all_left_eye_x) > 1:
        movements = [abs(all_left_eye_x[i] - all_left_eye_x[i - 1])
                     for i in range(1, len(all_left_eye_x))]
        saccade_velocity = sum(movements) / len(movements)
    else:
        saccade_velocity = 0.0

    # ── Pupil variability ─────────────────────────────────────────────────────
    openness_mean     = sum(all_openness) / len(all_openness)
    openness_variance = sum((o - openness_mean) ** 2 for o in all_openness) / len(all_openness)
    pupil_variability = math.sqrt(openness_variance)

    avg_kp_confidence = sum(all_confidences) / len(all_confidences)

    features = {
        "avg_gaze_deviation":   avg_deviation,
        "gaze_consistency":     gaze_consistency,
        "blink_rate_per_min":   blink_rate,
        "eye_contact_ratio":    eye_contact_ratio,
        "saccade_velocity":     saccade_velocity,
        "pupil_variability":    pupil_variability,
        "frame_count":          total_frames,
        "valid_frame_count":    len(per_frame_data),
        "avg_kp_confidence":    avg_kp_confidence,
    }

    print("Gaze analysis done."
          " Eye contact: " + str(round(eye_contact_ratio * 100, 1)) + "%"
          " | Avg iris ratio: " + str(round(avg_ratio, 3)) +
          " | Valid frames: "   + str(len(per_frame_data)) + "/" + str(total_frames))
    return features


def analyze_with_heuristic(frames):
    # Simple fallback when YOLO is not available.
    # Uses skin colour detection as a very rough proxy for face presence.
    # Results are approximate - install ultralytics for real gaze analysis.

    try:
        import cv2          # OpenCV for colour space conversion
        import numpy as np  # numpy for creating colour range arrays

        eye_contact_estimates = []

        for frame, timestamp in frames:
            # Get the upper-middle portion of the frame (where the face likely is)
            height, width = frame.shape[:2]
            face_region = frame[height // 5 : height // 2, width // 4 : 3 * width // 4]

            # Convert to HSV colour space (easier to detect skin tones)
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

            # Define skin colour range in HSV
            lower_skin = np.array([0, 20, 70],   dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # Create a mask: white where skin colour detected, black elsewhere
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Calculate what fraction of the region is skin-coloured
            skin_fraction = skin_mask.mean() / 255.0

            # Use skin fraction as a rough proxy for eye contact
            eye_contact_estimates.append(skin_fraction)

        # Average across all frames
        avg_eye_contact = sum(eye_contact_estimates) / max(len(eye_contact_estimates), 1)

    except ImportError:
        # If even OpenCV is not available, use neutral defaults
        avg_eye_contact = 0.5

    # Return neutral/estimated features
    features = {
        "avg_gaze_deviation":       0.3,               # assume moderate deviation
        "gaze_consistency":         0.6,               # assume moderate consistency
        "blink_rate_per_min":       15.0,              # average human blink rate
        "eye_contact_ratio":        avg_eye_contact,   # our rough estimate
        "saccade_velocity":         150.0,             # average value
        "pupil_variability":        0.1,               # low variability assumed
        "frame_count":              len(frames),
    }

    return features


def make_neutral_gaze_features(frame_count):
    # Returns a dictionary of neutral/default gaze features.
    # Used when there is no video or when no faces were detected.

    features = {
        "avg_gaze_deviation":   0.3,          # moderate - neither good nor bad
        "gaze_consistency":     0.7,          # slightly above average
        "blink_rate_per_min":   15.0,         # normal human average
        "eye_contact_ratio":    0.6,          # moderate eye contact
        "saccade_velocity":     150.0,        # normal eye movement
        "pupil_variability":    0.05,         # low variability
        "frame_count":          frame_count,  # how many frames we had
    }

    return features


def build_gaze_evidence(gaze_features):
    # Looks at the gaze features and builds a list of plain-English findings.
    # Returns a list of dictionaries, each describing something suspicious.

    evidence = []   # start with empty list

    # Check for low eye contact
    if gaze_features["eye_contact_ratio"] < 0.3:
        evidence.append({
            "type":        "gaze",
            "description": "Low eye contact ratio (" + str(round(gaze_features["eye_contact_ratio"] * 100, 1)) + "%) - possible evasiveness",
            "weight":      0.40,
            "value":       gaze_features["eye_contact_ratio"]
        })

    # Check for erratic gaze
    if gaze_features["gaze_consistency"] < 0.4:
        evidence.append({
            "type":        "gaze",
            "description": "Erratic gaze pattern (consistency score: " + str(round(gaze_features["gaze_consistency"], 2)) + ")",
            "weight":      0.35,
            "value":       gaze_features["gaze_consistency"]
        })

    # Check for high blink rate (stress indicator)
    if gaze_features["blink_rate_per_min"] > 30:
        evidence.append({
            "type":        "gaze",
            "description": "Elevated blink rate (" + str(round(gaze_features["blink_rate_per_min"], 0)) + " blinks/min) - possible stress",
            "weight":      0.25,
            "value":       gaze_features["blink_rate_per_min"]
        })

    return evidence
