# config.py
# This file holds all the settings for the interview analyzer.
# You can change any of these values to adjust how the program works.

# ── GAZE / VIDEO ANALYSIS SETTINGS ───────────────────────────────────────────

# Which YOLO Pose model file to use for eye gaze tracking
# The file "yolov8n-pose.pt" is downloaded automatically the first time if not present
YOLO_MODEL = "yolov8n-pose.pt"

# ── VIDEO SETTINGS ────────────────────────────────────────────────────────────

# How many frames to grab from the video every second
# Lower number = faster but less accurate
FRAME_SAMPLE_RATE = 5

# Maximum number of frames to look at (to avoid running out of memory)
MAX_FRAMES = 300

# ── AUDIO SETTINGS ────────────────────────────────────────────────────────────

# Which Whisper model to use for speech-to-text
# Options: "tiny" (fastest), "base", "small", "medium", "large" (most accurate)
WHISPER_MODEL = "base"

# The language spoken in the interview
WHISPER_LANGUAGE = "en"

# Whether to reduce background noise before transcribing
NOISE_REDUCE = True

# Audio sample rate (16000 Hz is what Whisper expects - don't change this)
SAMPLE_RATE = 16000

# ── NLP FEATURE WEIGHTS ───────────────────────────────────────────────────────
# These three weights control how filler words, repetition, and length variation
# contribute to the NLP anomaly score inside nlp.py.
# They must add up to 1.0.
# Matches the NLP sheet weightings (w1=filler, w2=repetition, w3=variation)
NLP_FILLER_WEIGHT      = 0.4   # weight for filler word rate
NLP_REPETITION_WEIGHT  = 0.4   # weight for sentence repetition
NLP_VARIATION_WEIGHT   = 0.2   # weight for sentence length variation

# ── FUSION / DECISION SETTINGS ────────────────────────────────────────────────

# How much weight to give the gaze (eye tracking) score vs the language score
# These two numbers must add up to 1.0
GAZE_WEIGHT = 0.45   # 45% of the final score comes from eye gaze
NLP_WEIGHT  = 0.55   # 55% of the final score comes from language analysis

# ── RULE ENGINE WEIGHT SETS ───────────────────────────────────────────────────
# The rule engine picks a (w1, w2) fusion weight pair based on signal reliability.
# These match the five cases in the Results Table spreadsheet.
# w1 = gaze weight, w2 = NLP weight  (must sum to 1.0 each row)
#
# Case 1: default - gaze trusted more         (0.6, 0.4)
# Case 2: default - gaze trusted more         (0.6, 0.4)
# Case 3: default - gaze trusted more         (0.6, 0.4)
# Case 4: NLP trusted more (low gaze quality) (0.4, 0.6)
# Case 5: equal weight                        (0.5, 0.5)
#
# The rule engine selects a case based on gaze confidence and NLP anomaly level.
RULE_WEIGHT_SETS = {
    "high_gaze_trust":   (0.6, 0.4),   # gaze is confident and stable
    "high_nlp_trust":    (0.4, 0.6),   # gaze is unreliable, trust NLP more
    "equal":             (0.5, 0.5),   # both signals equally reliable
}

# Gaze confidence threshold - below this, switch to high_nlp_trust weights
RULE_GAZE_CONFIDENCE_THRESHOLD = 0.6

# NLP anomaly threshold - above this, switch to high_nlp_trust weights
RULE_NLP_ANOMALY_THRESHOLD = 0.6

# ── DECISION BANDS ────────────────────────────────────────────────────────────
# Both gaze and NLP use a three-band system:
#
#   GAZE (eye_contact_ratio):
#     < GAZE_ANOMALY_THRESHOLD  (< 0.50)  → clear anomaly  → decision = fake
#     < GAZE_REAL_THRESHOLD     (< 0.80)  → uncertain zone → closed-loop prompt
#     >= GAZE_REAL_THRESHOLD    (>= 0.80) → clearly real   → decision = real
#
#   NLP (nlp_anomaly_score):
#     > NLP_ANOMALY_THRESHOLD   (> 0.20)  → clear anomaly  → decision = fake
#     > NLP_REAL_THRESHOLD      (> 0.10)  → uncertain zone → closed-loop prompt
#     <= NLP_REAL_THRESHOLD     (<= 0.10) → clearly real   → decision = real
#
# Bands are checked before weighted fusion so an unambiguous signal
# always overrides regardless of what the other modality shows.

# Gaze: below this eye contact = clearly not on screen → anomaly
GAZE_ANOMALY_THRESHOLD = 0.50

# Gaze: above this eye contact = clearly paying attention → real
GAZE_REAL_THRESHOLD = 0.80

# NLP: above this anomaly score = clearly scripted/anomalous → anomaly
NLP_ANOMALY_THRESHOLD = 0.20

# NLP: below this anomaly score = clearly normal speech → real
NLP_REAL_THRESHOLD = 0.10

# Legacy aliases kept so existing code referencing the old names still works
CLEAR_GAZE_ANOMALY_THRESHOLD = GAZE_ANOMALY_THRESHOLD
CLEAR_NLP_ANOMALY_THRESHOLD  = NLP_ANOMALY_THRESHOLD

# If uncertainty is above this level, ask for more evidence (closed loop)
UNCERTAINTY_THRESHOLD = 0.30

# If eye contact is below this fraction of the time, flag it as suspicious
MIN_EYE_CONTACT = 0.15

# If the deception score is above this level, override and call it fake
MAX_DECEPTION_SCORE = 0.75

# ── CLOSED LOOP SETTINGS ──────────────────────────────────────────────────────

# Maximum number of times to ask for more evidence before giving up
# Set to 1 for live interviews - one prompt round is enough
MAX_LOOP_ITERATIONS = 1

# ── LIVE PROMPTING SETTINGS ───────────────────────────────────────────────────

# If the gaze score alone is above this level, show the nose-touch prompt
# (gaze score is 0 to 1 - higher means more suspicious gaze)
GAZE_UNCERTAIN_THRESHOLD = 0.55

# If the NLP score alone is above this level, show the name/role prompt
# (NLP score is 0 to 1 - higher means more suspicious language)
NLP_UNCERTAIN_THRESHOLD = 0.55

# How many seconds to show the prompt overlay on screen before continuing
PROMPT_DISPLAY_SECONDS = 8

# How many extra seconds of video to request when uncertain
EXTRA_VIDEO_SECONDS = 30

# The follow-up question to ask when we need more information
FOLLOW_UP_QUESTION = "Could you elaborate on that last point with a specific example?"
