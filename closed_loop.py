# closed_loop.py
# This file handles the "closed loop" part of the pipeline.
# When the fusion engine is not confident enough about its decision,
# it triggers this orchestrator to request more evidence from the interviewer.
# The orchestrator builds a follow-up question, collects new data,
# and feeds it back into the pipeline for another attempt.

import config   # our settings


def should_request_more_evidence(result, iteration):
    # Decides whether to trigger the closed-loop prompt.
    # Returns True ONLY when both signals are in the uncertain middle band.
    #
    # Gaze bands (eye_contact_ratio):
    #   < 0.50  → clear anomaly  → no prompt (already fake)
    #   0.50-0.80 → uncertain    → prompt
    #   >= 0.80 → clearly real   → no prompt (already real)
    #
    # NLP bands (nlp_anomaly_score):
    #   > 0.20  → clear anomaly  → no prompt (already fake)
    #   0.10-0.20 → uncertain    → prompt
    #   <= 0.10 → clearly real   → no prompt (already real)

    if iteration >= config.MAX_LOOP_ITERATIONS:
        print("Reached maximum iterations (" + str(config.MAX_LOOP_ITERATIONS) + "), stopping")
        return False

    gaze_features = result.get("gaze_features", {})
    nlp_features  = result.get("nlp_features",  {})

    eye_contact = gaze_features.get("eye_contact_ratio",  1.0)
    nlp_anomaly = nlp_features.get("nlp_anomaly_score",   0.0)

    # Clear gaze anomaly - fusion already returned fake, do not prompt
    if eye_contact < config.GAZE_ANOMALY_THRESHOLD:
        print("Clear gaze anomaly (" + str(round(eye_contact * 100, 1)) +
              "% < " + str(round(config.GAZE_ANOMALY_THRESHOLD * 100)) +
              "%) - anomaly noted, no prompt")
        return False

    # Clearly real gaze - do not prompt
    if eye_contact >= config.GAZE_REAL_THRESHOLD:
        if nlp_anomaly <= config.NLP_ANOMALY_THRESHOLD:
            print("Clear gaze real (" + str(round(eye_contact * 100, 1)) +
                  "% >= " + str(round(config.GAZE_REAL_THRESHOLD * 100)) +
                  "%) - result is real, no prompt")
            return False

    # Clear NLP anomaly - fusion already returned fake, do not prompt
    if nlp_anomaly > config.NLP_ANOMALY_THRESHOLD:
        print("Clear NLP anomaly (score " + str(round(nlp_anomaly, 3)) +
              " > " + str(config.NLP_ANOMALY_THRESHOLD) + ") - anomaly noted, no prompt")
        return False

    # Clearly real NLP - do not prompt if gaze is also real
    if nlp_anomaly <= config.NLP_REAL_THRESHOLD:
        if eye_contact >= config.GAZE_REAL_THRESHOLD:
            print("Clear NLP real (score " + str(round(nlp_anomaly, 3)) +
                  " <= " + str(config.NLP_REAL_THRESHOLD) + ") - result is real, no prompt")
            return False

    # Decision must be uncertain to proceed
    if result["decision"] != "uncertain":
        print("Decision is confident (" + result["decision"] + ") - no prompt needed")
        return False

    # Both signals in uncertain middle band - prompt for more evidence
    print("Both signals in uncertain band - requesting more evidence "
          "(iteration " + str(iteration + 1) + ")")
    return True


def select_live_prompt(gaze_score, nlp_score):
    # Decides which prompt(s) to show on the interviewer overlay during a live session.
    # Looks at which individual signal (gaze or audio) is weak and returns
    # a list of prompt dictionaries. Each dictionary has:
    #   "type"    - what kind of prompt it is ("gaze", "audio", or "general")
    #   "message" - the actual text to show on screen

    prompts = []   # start with an empty list

    # Check if the gaze (eye tracking) signal is suspicious
    gaze_is_uncertain = gaze_score >= config.GAZE_UNCERTAIN_THRESHOLD

    # Check if the audio/language signal is suspicious
    nlp_is_uncertain = nlp_score >= config.NLP_UNCERTAIN_THRESHOLD

    if gaze_is_uncertain:
        # Eye contact was low or gaze was erratic - ask them to touch their nose.
        # This proves they can follow instructions and resets their gaze to centre.
        prompts.append({
            "type":    "gaze",
            "message": "Please touch your nose with your finger."
        })

    if nlp_is_uncertain:
        # Audio/language was unclear or suspicious - ask them to state their identity.
        # This gives us a clean audio sample and a baseline for their speech pattern.
        prompts.append({
            "type":    "audio",
            "message": "Please state your full name and your current role."
        })

    if not prompts:
        # Neither signal was clearly weak - use the default follow-up question
        prompts.append({
            "type":    "general",
            "message": config.FOLLOW_UP_QUESTION
        })

    return prompts


def build_follow_up_prompt(result):
    # Creates a targeted follow-up question based on what was most suspicious.
    # If we found gaze issues, we ask them to look at the camera.
    # If we found language issues, we ask for more specific detail.
    # Returns a string containing the follow-up question.

    # Find the piece of evidence with the highest weight
    highest_weight = 0.0
    most_important_evidence = None

    for item in result["evidence"]:
        if item["weight"] > highest_weight:
            highest_weight = item["weight"]
            most_important_evidence = item

    # If we found no evidence at all, use the default question
    if most_important_evidence is None:
        return config.FOLLOW_UP_QUESTION

    # Choose a question based on what type of evidence was most suspicious
    evidence_type = most_important_evidence["type"]

    if evidence_type == "gaze":
        # Gaze was suspicious - ask them to look at the camera and give detail
        return "Could you please look directly at the camera and describe a specific situation where you demonstrated that skill?"

    elif evidence_type in ("linguistic", "rule"):
        # Language was suspicious - ask for specific concrete details
        return "Could you give me a concrete example - including where you were, who else was present, and exactly what happened step by step?"

    else:
        # Fallback to default question
        return config.FOLLOW_UP_QUESTION


def request_additional_evidence(result, iteration):
    # Builds a "prompt request" describing what additional data we need.
    # In a live system this would be sent to the interviewer's screen.
    # Returns a dictionary describing the request.

    # Build the follow-up question
    question = build_follow_up_prompt(result)

    # Alternate between requesting video evidence and audio/text evidence
    # Odd iterations (1, 3, 5...) request visual evidence
    # Even iterations (2, 4, 6...) request speech/text evidence
    if iteration % 2 == 1:
        evidence_type = "visual_evidence"
    else:
        evidence_type = "speech_or_text_evidence"

    print("Prompt request: " + question)
    print("Evidence type needed: " + evidence_type)

    request = {
        "prompt":           question,          # the question to ask
        "type":             evidence_type,     # what kind of response we want
        "duration_seconds": config.EXTRA_VIDEO_SECONDS,  # how long extra video to capture
        "iteration":        iteration          # which loop we are on
    }

    return request


def merge_results(original_result, new_result, iteration):
    # Combines two results together after getting more evidence.
    # Gives 40% weight to the original result and 60% to the newer one.
    # All numeric scores are weighted-averaged so the merged result
    # carries meaningful values for gaze_score, nlp_score, weighted_sum etc.
    # Returns a single merged result dictionary.

    weight_old = 0.4   # weight for the original result
    weight_new = 0.6   # weight for the new result

    def wavg(key, default=0.0):
        # Weighted average of a numeric field across both results
        return (weight_old * original_result.get(key, default) +
                weight_new * new_result.get(key, default))

    # Average the confidence and uncertainty
    merged_confidence  = wavg("confidence")
    merged_uncertainty = wavg("uncertainty")

    # Average the signal scores
    merged_gaze_score   = wavg("gaze_score")
    merged_nlp_score    = wavg("nlp_score")
    merged_weighted_sum = wavg("weighted_sum")
    merged_raw_score    = wavg("raw_score")

    # Average the fusion weights (show what was used across both rounds)
    merged_gaze_weight = wavg("fusion_gaze_weight", default=config.GAZE_WEIGHT)
    merged_nlp_weight  = wavg("fusion_nlp_weight",  default=config.NLP_WEIGHT)

    # Perceptron output based on the merged weighted sum
    merged_perceptron = 1 if merged_weighted_sum >= 0.5 else 0

    # Rule engine - report whichever round drove the final decision
    merged_rule = new_result.get("rule_engine_rule",
                  original_result.get("rule_engine_rule", "merged"))

    # Decide on the final decision from the merged result
    if merged_uncertainty >= config.UNCERTAINTY_THRESHOLD:
        merged_decision = "uncertain"
    elif new_result["decision"] == original_result["decision"]:
        merged_decision = new_result["decision"]
    else:
        # Results disagree - trust the one with higher confidence
        if new_result["confidence"] >= original_result["confidence"]:
            merged_decision = new_result["decision"]
        else:
            merged_decision = original_result["decision"]

    # Merge NLP features using weighted average of the three scores
    orig_nlp  = original_result.get("nlp_features", {})
    new_nlp   = new_result.get("nlp_features", {})
    merged_nlp_features = {
        "transcript":        original_result.get("transcript", "") +
                             "\n\n[Follow-up]\n" +
                             new_result.get("transcript", ""),
        "filler_rate":       weight_old * orig_nlp.get("filler_rate",       0.0) +
                             weight_new * new_nlp.get("filler_rate",        0.0),
        "repetition":        weight_old * orig_nlp.get("repetition",        0.0) +
                             weight_new * new_nlp.get("repetition",         0.0),
        "length_variation":  weight_old * orig_nlp.get("length_variation",  0.0) +
                             weight_new * new_nlp.get("length_variation",   0.0),
        "nlp_anomaly_score": weight_old * orig_nlp.get("nlp_anomaly_score", 0.0) +
                             weight_new * new_nlp.get("nlp_anomaly_score",  0.0),
    }

    # Merge gaze features using weighted average
    orig_gaze = original_result.get("gaze_features", {})
    new_gaze  = new_result.get("gaze_features", {})
    merged_gaze_features = {
        "eye_contact_ratio":  weight_old * orig_gaze.get("eye_contact_ratio",  0.0) +
                              weight_new * new_gaze.get("eye_contact_ratio",   0.0),
        "gaze_consistency":   weight_old * orig_gaze.get("gaze_consistency",   0.0) +
                              weight_new * new_gaze.get("gaze_consistency",    0.0),
        "blink_rate_per_min": weight_old * orig_gaze.get("blink_rate_per_min", 0.0) +
                              weight_new * new_gaze.get("blink_rate_per_min",  0.0),
        "avg_kp_confidence":  weight_old * orig_gaze.get("avg_kp_confidence",  0.0) +
                              weight_new * new_gaze.get("avg_kp_confidence",   0.0),
        "valid_frame_count":  orig_gaze.get("valid_frame_count", 0) +
                              new_gaze.get("valid_frame_count",  0),
        "frame_count":        orig_gaze.get("frame_count", 0) +
                              new_gaze.get("frame_count",  0),
    }

    # Combine all evidence from both rounds
    combined_evidence = original_result.get("evidence", []) + new_result.get("evidence", [])

    merged = {
        "decision":               merged_decision,
        "confidence":             merged_confidence,
        "uncertainty":            merged_uncertainty,
        "evidence":               combined_evidence,
        "closed_loop_iterations": iteration,
        "transcript":             merged_nlp_features["transcript"],
        "gaze_score":             merged_gaze_score,
        "nlp_score":              merged_nlp_score,
        "raw_score":              merged_raw_score,
        "weighted_sum":           merged_weighted_sum,
        "fusion_gaze_weight":     merged_gaze_weight,
        "fusion_nlp_weight":      merged_nlp_weight,
        "perceptron_output":      merged_perceptron,
        "rule_engine_rule":       merged_rule,
        "gaze_features":          merged_gaze_features,
        "nlp_features":           merged_nlp_features,
    }

    return merged
