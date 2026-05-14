# audio_processor.py
# This file handles everything to do with audio:
#   1. Loading audio from a video or audio file
#   2. Reducing background noise
#   3. Converting speech to text using Whisper

import config   # our settings


def load_and_clean_audio(file_path):
    # Loads audio from a video or audio file, reduces noise, and normalizes the volume.
    # Returns a numpy array of audio samples ready for Whisper.
    # Returns None if librosa is not installed.

    try:
        import librosa   # library for loading and processing audio files
    except ImportError:
        print("Warning: librosa not installed. Cannot load audio.")
        print("Install it with: pip install librosa")
        return None

    print("Loading audio from: " + str(file_path))

    # Check the file exists before trying to open it
    import os as _os
    if not _os.path.exists(str(file_path)):
        print("Error: audio file not found: " + str(file_path))
        return None

    # Load the audio file and convert it to the sample rate Whisper expects
    # mono=True means convert to single channel (not stereo)
    try:
        audio, sample_rate = librosa.load(
            str(file_path),
            sr=config.SAMPLE_RATE,   # resample to 16000 Hz
            mono=True                # combine stereo channels into one
        )
    except Exception as error:
        print("Error loading audio: " + str(error))
        return None

    print("Audio loaded: " + str(round(len(audio) / config.SAMPLE_RATE, 1)) + " seconds")

    # Warn if the audio is very short - Whisper will produce an empty transcript
    if len(audio) / config.SAMPLE_RATE < 1.0:
        print("Warning: audio is less than 1 second - transcript may be empty")

    # Apply noise reduction if enabled in settings
    if config.NOISE_REDUCE:
        audio = reduce_noise(audio)

    # Normalize volume so the loudest point is at maximum
    audio = normalize_audio(audio)

    return audio


def reduce_noise(audio):
    # Reduces background noise from the audio using spectral subtraction.
    # Uses the first 0.5 seconds as a sample of what the "background noise" sounds like.
    # Returns the cleaned audio, or the original if noisereduce is not installed.

    try:
        import noisereduce as nr   # noise reduction library
        import numpy as np

        print("Reducing background noise...")

        # Use the first half-second as a noise profile
        # (assuming the person hasn't started talking yet)
        noise_sample = audio[:config.SAMPLE_RATE // 2]

        # Apply noise reduction
        cleaned_audio = nr.reduce_noise(
            y=audio,                         # the full audio
            sr=config.SAMPLE_RATE,           # sample rate
            y_noise=noise_sample,            # what the noise sounds like
            prop_decrease=0.75,              # reduce noise by 75%
            stationary=False                 # handles varying background noise
        )

        return cleaned_audio

    except ImportError:
        # noisereduce not installed - skip noise reduction
        print("Note: noisereduce not installed, skipping noise reduction")
        return audio


def normalize_audio(audio):
    # Scales the audio so the loudest sample is at the maximum possible volume.
    # This ensures Whisper gets consistently loud input regardless of recording volume.

    import numpy as np

    # Find the loudest single sample in the audio
    peak = abs(audio).max()

    if peak > 0:
        # Divide all samples by the peak to bring the max to 1.0
        audio = audio / peak

    return audio


def transcribe_audio(audio):
    # Converts an audio array into text using OpenAI's Whisper model.
    # Returns the transcribed text as a string, or empty string if Whisper is unavailable.

    if audio is None:
        print("No audio to transcribe")
        return ""

    try:
        import whisper   # OpenAI Whisper speech-to-text library
    except ImportError:
        print("Warning: openai-whisper not installed. Cannot transcribe audio.")
        print("Install it with: pip install openai-whisper")
        return ""

    print("Loading Whisper model: " + config.WHISPER_MODEL + " (this may take a moment)...")

    # Load the Whisper model (downloaded automatically first time)
    model = whisper.load_model(config.WHISPER_MODEL)

    print("Transcribing audio (" + str(round(len(audio) / config.SAMPLE_RATE, 1)) + " seconds)...")

    # Run the transcription
    result = model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,   # e.g. "en" for English
        fp16=False                           # fp16=False is safer on CPU
    )

    # Extract just the text from the result dictionary
    transcript = result.get("text", "").strip()

    print("Transcription complete (" + str(len(transcript)) + " characters)")
    return transcript
