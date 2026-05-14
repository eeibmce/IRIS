# fusion.py
# This file combines the gaze features and language features into one final decision.
# It uses a weighted average to blend the two sets of scores,
# then estimates how certain we are about the decision.

import math     # for the sigmoid function
import config   # our settings


def calculate_gaze_score(gaze_features):
    # Converts gaze features into a single anomaly likelihood score from 0 to 1.
    # 0 = completely normal gaze, 1 = completely suspicious gaze.

    eye_contact  = gaze_features["eye_contact_ratio"]
    consistency  = gaze_features["gaze_consistency"]
    blink_rate   = gaze_features["blink_rate_per_min"]
    pupil_var    = gaze_features["pupil_variability"]

    # Low eye contact → more suspicious (flip: 0 eye contact = 1.0 component)
    eye_contact_component = 1.0 - eye_contact

    # Low consistency → more suspicious
    consistency_component = 1.0 - consistency

    # High blink rate → more suspicious (normal is 10-20 per minute)
    blink_component = sigmoid(blink_rate - 25, steepness=0.15)

    # High pupil variability → more suspicious
    pupil_component = min(pupil_var * 3, 1.0)

    # When eye contact is clearly anomalous (below the clear threshold),
    # give it dominant weight so the gaze score reliably pushes the
    # perceptron to 1 even when NLP is clean and other gaze signals are moderate.
    # This prevents a 2.8% eye contact from being diluted to perceptron=0.
    if eye_contact < config.CLEAR_GAZE_ANOMALY_THRESHOLD:
        score = (
            0.70 * eye_contact_component  +   # dominant - clearly not looking
            0.15 * consistency_component  +
            0.10 * blink_component        +
            0.05 * pupil_component
        )
    else:
        # Normal weighting when eye contact is not clearly anomalous
        score = (
            0.40 * eye_contact_component  +
            0.30 * consistency_component  +
            0.20 * blink_component        +
            0.10 * pupil_component
        )

    score = max(0.0, min(1.0, score))
    return score


def calculate_nlp_score(nlp_features):
    # Returns the NLP anomaly score directly.
    # nlp.py already combines filler_rate, repetition and length_variation
    # into nlp_anomaly_score using the weights from config.py,
    # so there is nothing further to compute here.
    return float(nlp_features.get("nlp_anomaly_score", 0.5))


def estimate_uncertainty(gaze_score, nlp_score, gaze_features, nlp_features):
    # Works out how confident we are in the result.
    # High uncertainty means the two signals disagree or we don't have enough data.
    # Returns a number from 0 (very certain) to 1 (very uncertain).

    # 1. How much do the two scores disagree?
    # If gaze says 0.8 (fake) and NLP says 0.2 (real), disagreement = 0.6
    # Weight reduced from 0.40 to 0.25 - disagreement alone should not
    # dominate the uncertainty when one signal is simply neutral (near 0.5)
    disagreement = abs(gaze_score - nlp_score)

    # 2. How close is the combined score to the decision boundary (0.5)?
    # A score near 0.5 means we can't decide - that's high uncertainty
    combined = config.GAZE_WEIGHT * gaze_score + config.NLP_WEIGHT * nlp_score
    boundary_proximity = 1.0 - 2 * abs(combined - 0.5)   # max at 0.5, zero at 0 and 1

    # 3. Did we have enough video frames to be confident?
    # We now use valid_frame_count (frames that passed keypoint confidence filter)
    # rather than raw frame_count, so we only reward frames we actually trust
    valid_frames = gaze_features.get("valid_frame_count", gaze_features.get("frame_count", 0))
    frame_penalty = max(0.0, 1.0 - valid_frames / 50.0)   # 0 penalty if 50+ valid frames

    # 4. Was the transcript long enough to analyze properly?
    transcript_length = len(nlp_features.get("transcript", ""))
    if transcript_length < 50:
        transcript_penalty = 1.0   # very short transcript = high uncertainty
    else:
        transcript_penalty = 0.0

    # 5. Was YOLO's average keypoint confidence low?
    # If YOLO was barely detecting the face, the gaze score is unreliable
    avg_kp_conf = gaze_features.get("avg_kp_confidence", 1.0)
    kp_penalty = max(0.0, 1.0 - avg_kp_conf * 1.5)   # 0 penalty above ~0.67 confidence

    # Combine uncertainty components with weights (must sum to 1.0)
    uncertainty = (
        0.25 * disagreement       +   # disagreement between signals
        0.35 * boundary_proximity +   # how close to the borderline (most important)
        0.15 * frame_penalty      +   # not enough valid video frames
        0.10 * transcript_penalty +   # not enough text
        0.15 * kp_penalty             # YOLO was not confident about keypoints
    )

    # Clamp between 0 and 1
    uncertainty = max(0.0, min(1.0, uncertainty))
    return uncertainty


def apply_hard_rules(gaze_features, nlp_features, fusion_score):
    # Checks for extreme cases that should override the weighted score.
    # Uses only the three NLP measures (filler, repetition, variation)
    # and the gaze eye contact ratio.

    result = {
        "override":   False,
        "decision":   None,
        "confidence": 0.0,
        "evidence":   []
    }

    # Rule 1: If NLP anomaly score is critically high, call it fake
    nlp_anomaly = nlp_features.get("nlp_anomaly_score", 0.0)
    if nlp_anomaly >= config.MAX_DECEPTION_SCORE:
        print("Hard rule triggered: NLP anomaly = " + str(round(nlp_anomaly, 2)))
        result["override"]   = True
        result["decision"]   = "fake"
        result["confidence"] = min(nlp_anomaly + 0.10, 0.99)
        result["evidence"].append({
            "type":        "rule",
            "description": "Hard rule: NLP anomaly score critically high (" + str(round(nlp_anomaly * 100, 1)) + "%)",
            "weight":      0.90,
            "value":       nlp_anomaly
        })

    # Rule 2: If eye contact is almost zero, add as suspicious evidence
    eye_contact = gaze_features.get("eye_contact_ratio", 1.0)
    if eye_contact < config.MIN_EYE_CONTACT and not result["override"]:
        print("Hard rule triggered: eye contact = " + str(round(eye_contact, 2)))
        result["evidence"].append({
            "type":        "rule",
            "description": "Hard rule: eye contact critically low (" + str(round(eye_contact * 100, 1)) + "%)",
            "weight":      0.50,
            "value":       eye_contact
        })

    # Rule 3: If filler rate is very high AND fusion score is already high, flag it
    filler = nlp_features.get("filler_rate", 0.0)
    if filler > 0.10 and fusion_score > 0.6 and not result["override"]:
        print("Hard rule triggered: filler rate = " + str(round(filler, 2)))
        result["evidence"].append({
            "type":        "rule",
            "description": "Hard rule: excessive filler words (" + str(round(filler * 100, 1)) + "%) with high anomaly score",
            "weight":      0.35,
            "value":       filler
        })

    return result


def make_decision(raw_score, uncertainty):
    # Converts the raw 0-1 score + uncertainty into a final decision.
    # Returns a tuple of (decision_string, confidence_number).
    # decision_string is one of: "real", "fake", "uncertain"

    # If uncertainty is too high, we can't make a reliable decision
    if uncertainty >= config.UNCERTAINTY_THRESHOLD:
        confidence = 1.0 - uncertainty   # confidence inversely related to uncertainty
        return "uncertain", confidence

    # If score is above 0.55, lean towards fake
    if raw_score >= 0.55:
        # Use sigmoid to map score to confidence (smooth curve)
        confidence = sigmoid(raw_score - 0.5, steepness=10.0)
        confidence = max(0.50, min(0.99, confidence))   # clamp to sensible range
        return "fake", confidence

    # Otherwise lean towards real
    else:
        confidence = sigmoid(0.5 - raw_score, steepness=10.0)
        confidence = max(0.50, min(0.99, confidence))
        return "real", confidence


def select_fusion_weights(gaze_features, nlp_features):
    # Rule engine: picks the (w1, w2) fusion weight pair based on signal quality.
    # Returns (gaze_weight, nlp_weight, rule_name).

    avg_kp_conf  = gaze_features.get("avg_kp_confidence",  1.0)
    nlp_anomaly  = nlp_features.get("nlp_anomaly_score",   0.5)
    valid_frames = gaze_features.get("valid_frame_count",  50)
    eye_contact  = gaze_features.get("eye_contact_ratio",  1.0)

    # Rule 0: Eye contact is clearly very low - the gaze score IS the evidence.
    # Even if YOLO keypoint confidence was shaky, a 2-5% eye contact reading
    # is unambiguous. Always trust gaze heavily in this case so the perceptron
    # correctly outputs 1 (anomaly) rather than being dragged down by a neutral NLP.
    if eye_contact < config.CLEAR_GAZE_ANOMALY_THRESHOLD:
        weights = config.RULE_WEIGHT_SETS["high_gaze_trust"]
        rule    = "clear_gaze_anomaly"

    # Rule 1: Low YOLO keypoint confidence or very few valid frames → trust NLP more
    elif avg_kp_conf < config.RULE_GAZE_CONFIDENCE_THRESHOLD or valid_frames < 10:
        weights = config.RULE_WEIGHT_SETS["high_nlp_trust"]
        rule    = "low_gaze_quality"

    # Rule 2: High NLP anomaly and moderate gaze confidence → trust NLP more
    elif nlp_anomaly >= config.RULE_NLP_ANOMALY_THRESHOLD and avg_kp_conf < 0.75:
        weights = config.RULE_WEIGHT_SETS["high_nlp_trust"]
        rule    = "high_nlp_anomaly"

    # Rule 3: Good gaze signal, moderate NLP → trust gaze more
    elif avg_kp_conf >= 0.75 and nlp_anomaly < config.RULE_NLP_ANOMALY_THRESHOLD:
        weights = config.RULE_WEIGHT_SETS["high_gaze_trust"]
        rule    = "high_gaze_quality"

    # Rule 4: Both signals moderate → equal weight
    else:
        weights = config.RULE_WEIGHT_SETS["equal"]
        rule    = "equal_weight"

    gaze_w, nlp_w = weights
    print("Rule engine: " + rule + " → weights (w1=" + str(gaze_w) + ", w2=" + str(nlp_w) + ")")
    return gaze_w, nlp_w, rule


def fuse(gaze_features, nlp_features, gaze_evidence, nlp_evidence):
    # Main function: combines gaze and language features into a final result.
    # Returns a result dictionary with the decision, confidence, and all evidence.

    # Step 1: Convert each set of features into a 0-1 score
    gaze_score = calculate_gaze_score(gaze_features)
    nlp_score  = calculate_nlp_score(nlp_features)

    print("Gaze score: " + str(round(gaze_score, 3)) +
          " | NLP score: " + str(round(nlp_score, 3)))

    eye_contact = gaze_features.get("eye_contact_ratio", 1.0)
    nlp_anomaly = nlp_features.get("nlp_anomaly_score",  0.0)

    # ── Three-band gaze decision ──────────────────────────────────────────────
    # < GAZE_ANOMALY_THRESHOLD  → clearly not on screen → fake
    # >= GAZE_REAL_THRESHOLD    → clearly attentive     → real
    # between the two           → uncertain, continue to fusion

    if eye_contact < config.GAZE_ANOMALY_THRESHOLD:
        print("Clear gaze anomaly: eye contact = " +
              str(round(eye_contact * 100, 1)) + "% (< " +
              str(round(config.GAZE_ANOMALY_THRESHOLD * 100)) + "%) → fake")
        all_evidence = gaze_evidence + nlp_evidence
        all_evidence.append({
            "type":        "fusion",
            "description": "Clear gaze anomaly: eye contact " +
                           str(round(eye_contact * 100, 1)) + "% below " +
                           str(round(config.GAZE_ANOMALY_THRESHOLD * 100)) + "% threshold",
            "weight":      0.90,
            "value":       eye_contact,
        })
        return {
            "decision":           "fake",
            "confidence":         min(1.0 - eye_contact + 0.10, 0.99),
            "uncertainty":        0.05,
            "evidence":           all_evidence,
            "gaze_score":         gaze_score,
            "nlp_score":          nlp_score,
            "raw_score":          gaze_score,
            "weighted_sum":       gaze_score,
            "fusion_gaze_weight": 1.0,
            "fusion_nlp_weight":  0.0,
            "rule_engine_rule":   "clear_gaze_anomaly_override",
            "perceptron_output":  1,
            "gaze_features":      gaze_features,
            "nlp_features":       nlp_features,
        }

    if eye_contact >= config.GAZE_REAL_THRESHOLD:
        # Only call it real from gaze alone if NLP is also not anomalous
        if nlp_anomaly <= config.NLP_ANOMALY_THRESHOLD:
            print("Clear gaze real: eye contact = " +
                  str(round(eye_contact * 100, 1)) + "% (>= " +
                  str(round(config.GAZE_REAL_THRESHOLD * 100)) + "%) → real")
            all_evidence = gaze_evidence + nlp_evidence
            all_evidence.append({
                "type":        "fusion",
                "description": "Clear gaze real: eye contact " +
                               str(round(eye_contact * 100, 1)) + "% above " +
                               str(round(config.GAZE_REAL_THRESHOLD * 100)) + "% threshold",
                "weight":      0.90,
                "value":       eye_contact,
            })
            return {
                "decision":           "real",
                "confidence":         min(eye_contact + 0.05, 0.99),
                "uncertainty":        0.05,
                "evidence":           all_evidence,
                "gaze_score":         gaze_score,
                "nlp_score":          nlp_score,
                "raw_score":          gaze_score,
                "weighted_sum":       gaze_score,
                "fusion_gaze_weight": 1.0,
                "fusion_nlp_weight":  0.0,
                "rule_engine_rule":   "clear_gaze_real_override",
                "perceptron_output":  0,
                "gaze_features":      gaze_features,
                "nlp_features":       nlp_features,
            }
        # Gaze is real but NLP is suspicious - fall through to fusion

    # ── Three-band NLP decision ───────────────────────────────────────────────
    # > NLP_ANOMALY_THRESHOLD  → clearly anomalous speech → fake
    # <= NLP_REAL_THRESHOLD    → clearly normal speech    → real
    # between the two          → uncertain, continue to fusion

    if nlp_anomaly > config.NLP_ANOMALY_THRESHOLD:
        print("Clear NLP anomaly: score = " +
              str(round(nlp_anomaly, 3)) + " (> " +
              str(config.NLP_ANOMALY_THRESHOLD) + ") → fake")
        all_evidence = gaze_evidence + nlp_evidence
        all_evidence.append({
            "type":        "fusion",
            "description": "Clear NLP anomaly: score " +
                           str(round(nlp_anomaly, 3)) + " above " +
                           str(config.NLP_ANOMALY_THRESHOLD) + " threshold",
            "weight":      0.90,
            "value":       nlp_anomaly,
        })
        return {
            "decision":           "fake",
            "confidence":         min(nlp_anomaly + 0.10, 0.99),
            "uncertainty":        0.05,
            "evidence":           all_evidence,
            "gaze_score":         gaze_score,
            "nlp_score":          nlp_score,
            "raw_score":          nlp_score,
            "weighted_sum":       nlp_score,
            "fusion_gaze_weight": 0.0,
            "fusion_nlp_weight":  1.0,
            "rule_engine_rule":   "clear_nlp_anomaly_override",
            "perceptron_output":  1,
            "gaze_features":      gaze_features,
            "nlp_features":       nlp_features,
        }

    if nlp_anomaly <= config.NLP_REAL_THRESHOLD:
        # NLP is clearly normal - only call it real if gaze is also not in the uncertain zone
        if eye_contact >= config.GAZE_REAL_THRESHOLD:
            print("Clear NLP real: score = " +
                  str(round(nlp_anomaly, 3)) + " (<= " +
                  str(config.NLP_REAL_THRESHOLD) + ") with strong gaze → real")
            all_evidence = gaze_evidence + nlp_evidence
            all_evidence.append({
                "type":        "fusion",
                "description": "Clear NLP real: score " +
                               str(round(nlp_anomaly, 3)) + " below " +
                               str(config.NLP_REAL_THRESHOLD) + " threshold",
                "weight":      0.90,
                "value":       nlp_anomaly,
            })
            return {
                "decision":           "real",
                "confidence":         min(1.0 - nlp_anomaly + 0.05, 0.99),
                "uncertainty":        0.05,
                "evidence":           all_evidence,
                "gaze_score":         gaze_score,
                "nlp_score":          nlp_score,
                "raw_score":          nlp_score,
                "weighted_sum":       nlp_score,
                "fusion_gaze_weight": 0.0,
                "fusion_nlp_weight":  1.0,
                "rule_engine_rule":   "clear_nlp_real_override",
                "perceptron_output":  0,
                "gaze_features":      gaze_features,
                "nlp_features":       nlp_features,
            }
        # NLP is real but gaze is in uncertain zone - fall through to fusion

    # If we reach here, at least one signal is in the uncertain middle band.
    # Force decision = uncertain and trigger the closed-loop prompt.
    # The weighted fusion score is still computed for the terminal report.
    in_uncertain_band = (
        (config.GAZE_ANOMALY_THRESHOLD <= eye_contact < config.GAZE_REAL_THRESHOLD) or
        (config.NLP_REAL_THRESHOLD     <  nlp_anomaly <= config.NLP_ANOMALY_THRESHOLD)
    )

    # Step 2: Rule engine picks the fusion weights dynamically
    gaze_w, nlp_w, rule_name = select_fusion_weights(gaze_features, nlp_features)
    raw_score = gaze_w * gaze_score + nlp_w * nlp_score

    # Step 3: Estimate how uncertain we are
    uncertainty = estimate_uncertainty(gaze_score, nlp_score, gaze_features, nlp_features)

    print("Combined score: " + str(round(raw_score, 3)) +
          " | Uncertainty: " + str(round(uncertainty, 3)))

    # Step 4: Check hard rules for extreme cases
    rule_result = apply_hard_rules(gaze_features, nlp_features, raw_score)

    # Step 5: Determine final decision
    if rule_result["override"]:
        decision   = rule_result["decision"]
        confidence = rule_result["confidence"]
    elif in_uncertain_band:
        # At least one signal is in the uncertain middle band.
        # Force uncertain regardless of the weighted sum so the closed-loop
        # prompt is always triggered for mid-band readings.
        decision   = "uncertain"
        confidence = 1.0 - max(uncertainty, 0.30)
        print("Middle band detected → decision forced to uncertain")
    else:
        decision, confidence = make_decision(raw_score, uncertainty)

    # Step 6: Collect all evidence together
    all_evidence = gaze_evidence + nlp_evidence + rule_result["evidence"]

    all_evidence.append({
        "type":        "fusion",
        "description": "Rule engine weights (w1=" + str(gaze_w) + ", w2=" + str(nlp_w) + ") via rule: " + rule_name + " | score=" + str(round(raw_score, 3)),
        "weight":      0.60,
        "value":       raw_score
    })

    # Step 7: Build and return the result dictionary
    # Perceptron = 1 whenever an anomaly or uncertainty is flagged.
    # uncertain means something suspicious was detected - always output 1.
    perceptron_out = 1 if (decision in ("fake", "uncertain")) else 0

    result = {
        "decision":              decision,
        "confidence":            confidence,
        "uncertainty":           uncertainty,
        "evidence":              all_evidence,
        "gaze_score":            gaze_score,
        "nlp_score":             nlp_score,
        "raw_score":             raw_score,
        "fusion_gaze_weight":    gaze_w,
        "fusion_nlp_weight":     nlp_w,
        "rule_engine_rule":      rule_name,
        "weighted_sum":          raw_score,
        "perceptron_output":     perceptron_out,
    }

    return result


def sigmoid(x, steepness=1.0):
    # The sigmoid function converts any number to a value between 0 and 1.
    # It creates an S-shaped curve: very negative x → near 0, very positive x → near 1.
    # We use it to convert scores into smooth probabilities.

    # Numerically stable version (avoids math overflow for extreme values)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-steepness * x))
    else:
        exp_val = math.exp(steepness * x)
        return exp_val / (1.0 + exp_val)
