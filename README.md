# Interview Authenticity Analyzer

Detects whether an interview response seems genuine or fabricated by combining
eye-gaze analysis (from the video) with language analysis (from the transcript).

---

## How to install

**Step 1 — Open a terminal in this folder**
In VS Code: press Ctrl+` (backtick) to open the terminal

**Step 2 — Install the packages**
```
pip install -r requirements.txt
```

---

## How to run

**Analyze a video file:**
```
python run.py --video interview.mp4
```

**Analyze just text (no video needed):**
```
python run.py --transcript "I worked at Acme for three years..."
```

**Save a report to a file:**
```
python run.py --video interview.mp4 --output report.txt
```

**See all options:**
```
python run.py --help
```

---

## How to run the tests
```
python tests.py
```

---

## File overview

```
simple_analyzer/
│
├── run.py                  ← START HERE: the file you run from the terminal
├── pipeline.py             ← joins all the steps together
│
├── config.py               ← all settings (thresholds, weights, model names)
│
├── frame_extractor.py      ← pulls image frames out of a video file
├── gaze_analyzer.py        ← measures eye contact and gaze patterns in frames
│
├── audio_processor.py      ← loads audio, removes noise, transcribes speech
│
├── deception_features.py   ← measures linguistic deception markers in text
├── transcript_analyzer.py  ← runs all text measurements and collects evidence
│
├── fusion.py               ← combines gaze + text scores into one decision
├── closed_loop.py          ← asks follow-up questions when uncertain
│
├── tests.py                ← checks everything works correctly
└── requirements.txt        ← list of packages to install
```

---

## How it works (plain English)

1. **Video frames** are extracted from the interview recording
2. **Eye gaze** is analyzed — is the person looking at the camera? Are they blinking a lot?
3. **Audio** is loaded and converted to text using Whisper
4. **Language** is analyzed — are they using vague words? Do they tell the story in order?
5. **Scores are combined** — gaze (45%) + language (55%) = one overall deception score
6. **Decision** is made: "real", "fake", or "uncertain"
7. If **uncertain**, a follow-up question is suggested to get more evidence

---

## What each score means

| Score | Range | What it means |
|---|---|---|
| Decision | real / fake / uncertain | The final verdict |
| Confidence | 0% – 100% | How sure we are of the decision |
| Uncertainty | 0% – 100% | How much the signals conflicted |
| Gaze score | 0 – 1 | 0 = genuine gaze, 1 = suspicious gaze |
| NLP score | 0 – 1 | 0 = genuine language, 1 = suspicious language |
