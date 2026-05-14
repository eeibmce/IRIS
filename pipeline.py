# pipeline.py
# This is the main file that ties the whole project together.
# It runs each step of the pipeline in order:
#   1. Extract frames from the video
#   2. Analyze eye gaze from the frames
#   3. Load and transcribe the audio
#   4. Analyze the transcript for deception
#   5. Fuse the gaze and language scores into a final decision
#   6. If uncertain, loop back and request more evidence

import frame_extractor      # extracts frames from video
import gaze_analyzer        # analyzes eye gaze in frames
import audio_processor      # loads audio and transcribes speech
import transcript_analyzer  # analyzes transcript for deception
import fusion               # combines all scores into a decision
import closed_loop          # handles uncertainty and follow-up
import config               # settings and thresholds


def analyze(video_path=None, audio_path=None, transcript=None, iteration=0):
    # Main function: runs the full pipeline on an interview.
    # You must provide at least one of: video_path, audio_path, or transcript.
    #
    # video_path:  path to the video file (e.g. "interview.mp4")
    # audio_path:  path to a separate audio file (optional, uses video_path if not given)
    # transcript:  pre-written text (skips Whisper if provided)
    # iteration:   internal counter for closed-loop tracking (leave as 0)
    #
    # Returns a result dictionary with: decision, confidence, uncertainty, evidence

    print("")
    print("=== Interview Analyzer (iteration " + str(iteration) + ") ===")

    # ── Step 1: VIDEO PATH ────────────────────────────────────────────────────

    if video_path is not None:
        # Extract image frames from the video file
        frames = frame_extractor.extract_frames(video_path)
    else:
        # No video provided
        print("No video provided - using neutral gaze scores")
        frames = []

    # Analyze the frames for eye gaze features
    gaze_features = gaze_analyzer.analyze_gaze(frames)

    # Build a list of gaze-based evidence items
    gaze_evidence = gaze_analyzer.build_gaze_evidence(gaze_features)

    # ── Step 2: AUDIO PATH ────────────────────────────────────────────────────

    if transcript is not None:
        # A transcript was already provided - skip audio processing
        print("Using provided transcript")
        transcript_text = transcript

    else:
        # Determine which file to load audio from
        audio_source = audio_path if audio_path is not None else video_path

        if audio_source is not None:
            # Load and clean the audio, then transcribe it
            audio_data = audio_processor.load_and_clean_audio(audio_source)
            transcript_text = audio_processor.transcribe_audio(audio_data)
        else:
            print("No audio source provided - using empty transcript")
            transcript_text = ""

    # ── Step 3: NLP ANALYSIS ──────────────────────────────────────────────────

    # Analyze the transcript text for deception markers
    nlp_features = transcript_analyzer.analyze_transcript(transcript_text)

    # Build a list of language-based evidence items
    nlp_evidence = transcript_analyzer.build_nlp_evidence(nlp_features)

    # ── Step 4: FUSION ────────────────────────────────────────────────────────

    # Combine gaze and NLP scores into a final decision
    result = fusion.fuse(gaze_features, nlp_features, gaze_evidence, nlp_evidence)

    # Add transcript and metadata to the result
    result["transcript"]             = transcript_text
    result["gaze_features"]          = gaze_features
    result["nlp_features"]           = nlp_features
    result["closed_loop_iterations"] = iteration

    print("Decision: " + result["decision"] +
          " | Confidence: " + str(round(result["confidence"] * 100, 1)) + "%" +
          " | Uncertainty: " + str(round(result["uncertainty"] * 100, 1)) + "%")

    # ── Step 5: CLOSED LOOP ───────────────────────────────────────────────────

    if closed_loop.should_request_more_evidence(result, iteration):
        # Uncertain result and no clear anomaly - request more evidence
        request = closed_loop.request_additional_evidence(result, iteration + 1)
        print("Batch mode: no new data available, stopping closed loop")
        result["closed_loop_iterations"] = iteration + 1

    else:
        # Check if the decision was blocked due to a clear anomaly (not just confidence)
        # Add a summary note to the result so the terminal report can show it
        gaze_features = result.get("gaze_features", {})
        nlp_features  = result.get("nlp_features",  {})
        eye_contact   = gaze_features.get("eye_contact_ratio",  1.0)
        nlp_anomaly   = nlp_features.get("nlp_anomaly_score",   0.0)

        notes = []
        if eye_contact < config.GAZE_ANOMALY_THRESHOLD:
            notes.append("Gaze clearly not on screen (" +
                         str(round(eye_contact * 100, 1)) + "% eye contact)")
        if nlp_anomaly > config.NLP_ANOMALY_THRESHOLD:
            notes.append("NLP clearly anomalous (score: " +
                         str(round(nlp_anomaly, 3)) + ")")

        if notes:
            result["clear_anomaly_notes"] = notes
            print("Clear anomaly findings (no prompt triggered):")
            for note in notes:
                print("  - " + note)

    print("Final decision: " + result["decision"])
    return result


def analyze_text_only(transcript):
    # Shortcut function for analyzing just text (no video or audio needed).
    # Useful for quick testing or when you only have a written transcript.

    return analyze(transcript=transcript)


def print_result(result):
    # Prints a full detailed breakdown of the analysis result to the terminal.
    # Delegates to save_results_table.print_terminal_result() which formats
    # all gaze, NLP, fusion and decision fields in a clear boxed layout.

    import save_results_table
    save_results_table.print_terminal_result(result)
