# transcript_analyzer.py
# Runs nlp.py on the transcript and returns the three scores
# plus the combined anomaly score ready for fusion.py.

import nlp


def analyze_transcript(transcript):
    # Takes a transcript string.
    # Returns a dictionary with filler_rate, repetition, length_variation,
    # nlp_anomaly_score, and the transcript itself.

    print("Analyzing transcript (" + str(len(transcript)) + " characters)...")

    if len(transcript.strip()) == 0:
        print("Warning: transcript is empty, returning default scores")
        return make_empty_features()

    r = nlp.analyse_transcript(transcript)

    print("Filler: "     + str(round(r["filler_rate"],       3)) +
          " | Repeat: "  + str(round(r["repetition"],        3)) +
          " | Variation: "+ str(round(r["length_variation"],  3)) +
          " | Anomaly: "  + str(round(r["nlp_anomaly_score"], 3)))

    return {
        "transcript":        transcript,
        "filler_rate":       r["filler_rate"],
        "repetition":        r["repetition"],
        "length_variation":  r["length_variation"],
        "nlp_anomaly_score": r["nlp_anomaly_score"],
    }


def make_empty_features():
    # Returns neutral defaults when there is no transcript.
    return {
        "transcript":        "",
        "filler_rate":       0.0,
        "repetition":        0.0,
        "length_variation":  0.0,
        "nlp_anomaly_score": 0.5,
    }


def build_nlp_evidence(features):
    # Builds plain-English evidence items from the three scores.

    evidence = []

    if features["filler_rate"] > 0.03:
        evidence.append({
            "type":        "linguistic",
            "description": "High filler word rate (" + str(round(features["filler_rate"] * 100, 1)) + "%) - frequent um, uh, like etc.",
            "weight":      0.35,
            "value":       features["filler_rate"],
        })

    if features["repetition"] > 0.1:
        evidence.append({
            "type":        "linguistic",
            "description": "Repeated sentences (" + str(round(features["repetition"] * 100, 1)) + "% of sentences repeated)",
            "weight":      0.40,
            "value":       features["repetition"],
        })

    if features["length_variation"] > 0.5:
        evidence.append({
            "type":        "linguistic",
            "description": "Inconsistent sentence lengths (variation score: " + str(round(features["length_variation"], 2)) + ")",
            "weight":      0.25,
            "value":       features["length_variation"],
        })

    if features["nlp_anomaly_score"] > 0.5:
        evidence.append({
            "type":        "linguistic",
            "description": "Overall speech anomaly score elevated (" + str(round(features["nlp_anomaly_score"] * 100, 1)) + "%)",
            "weight":      0.40,
            "value":       features["nlp_anomaly_score"],
        })

    return evidence
