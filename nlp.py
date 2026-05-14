# nlp.py
# Analyses a speech transcript and scores it on three measures:
#   filler_rate      - how often filler words appear
#   repetition       - how many sentences are similar to another sentence
#   length_variation - how inconsistent sentence lengths are
#
# Whisper often mishears filler words:
#   "um"  -> "I'm a" / "I'm" at the start of a phrase
#   "uh"  -> "a" or dropped entirely
#   "erm" -> "I'm" / "um"
# FILLER_PATTERNS covers both the spoken word and its common Whisper equivalent.
#
# Repetition uses fuzzy matching (word overlap ratio) rather than exact string
# comparison, so Whisper transcription variation does not hide repeated sentences.
#
# Run directly to test:  python nlp.py

import config

# Words that are actually spoken as fillers
FILLER_WORDS = ["um", "uh", "erm", "like", "eh"]

# Phrases that Whisper commonly transcribes instead of a filler word.
# These are checked as substrings at the start of a lowercased sentence.
WHISPER_FILLER_PATTERNS = [
    "i'm a ",    # Whisper mishearing of "um"
    "i'm,",      # "um," transcribed as "i'm,"
]


def split_into_sentences(text):
    # Splits text into sentences at full stops.
    sentences = []
    for s in text.replace("\n", " ").split("."):
        s = s.strip()
        if s != "":
            sentences.append(s)
    return sentences


def filler_rate(text):
    # Fraction of words that are filler words, including Whisper mishearing patterns.

    words     = text.lower().split()
    sentences = split_into_sentences(text.lower())

    if len(words) == 0:
        return 0.0

    # Count actual filler words
    filler_count = sum(1 for w in words if w in FILLER_WORDS)

    # Count sentences that start with a Whisper filler pattern.
    # Each such sentence gets one extra filler credit.
    for s in sentences:
        s_lower = s.lower().strip()
        for pattern in WHISPER_FILLER_PATTERNS:
            if s_lower.startswith(pattern):
                filler_count += 1
                break   # only count once per sentence

    return filler_count / len(words)


def _word_overlap(s1, s2):
    # Returns the Jaccard similarity between the word sets of two sentences.
    # 1.0 = identical words, 0.0 = no words in common.
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def repetition_score(sentences):
    # Fraction of sentences that are highly similar (>= 0.70 word overlap)
    # to at least one other sentence in the transcript.
    # Fuzzy matching handles Whisper transcription variation where the same
    # spoken sentence is transcribed slightly differently each time.

    if len(sentences) <= 1:
        return 0.0

    repeated = 0
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                if _word_overlap(sentences[i], sentences[j]) >= 0.70:
                    repeated += 1
                    break   # this sentence is repeated - no need to check more

    return repeated / len(sentences)


def length_variation(sentences):
    # How inconsistent sentence lengths are (0=all same length, higher=more varied).
    if len(sentences) == 0:
        return 0.0
    lengths    = [len(s.split()) for s in sentences]
    average    = sum(lengths) / len(lengths)
    total_diff = sum(abs(l - average) for l in lengths)
    return total_diff / (len(sentences) * average + 0.0001)


def analyse_transcript(transcript):
    # Main function. Returns filler_rate, repetition, length_variation,
    # and nlp_anomaly_score.

    sentences = split_into_sentences(transcript)

    fr  = filler_rate(transcript)
    rep = repetition_score(sentences)
    var = length_variation(sentences)

    anomaly = min(
        config.NLP_FILLER_WEIGHT     * fr  +
        config.NLP_REPETITION_WEIGHT * rep +
        config.NLP_VARIATION_WEIGHT  * var,
        1.0
    )

    return {
        "filler_rate":       fr,
        "repetition":        rep,
        "length_variation":  var,
        "nlp_anomaly_score": anomaly,
    }


if __name__ == "__main__":
    # Test with the actual Whisper output the user reported
    whisper_transcript = (
        "I'm a work well and I'm a team and deliver results. "
        "I'm a work well and the team and deliver results. "
        "I'm a work well and deliver results. "
        "I'm consistent and consistent. "
        "I'm consistent. Test test. Video Video All shown here"
    )

    print("Testing with actual Whisper output:")
    print("")
    r = analyse_transcript(whisper_transcript)
    for k, v in r.items():
        print(k + ": " + str(round(v, 3)))
