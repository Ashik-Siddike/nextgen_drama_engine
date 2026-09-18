# 🎬 NextGen Drama Engine

An end-to-end cinematic AI dubbing and localization engine designed specifically for Asian vertical short dramas (Chinese, Korean, Japanese) into natural, emotionally charged Hindi.

---

## 🌟 Key Architecture & Breakthroughs

Traditional automated dubbing engines process short video clips in 5-10 second micro-batches, resulting in severe context loss (*"আগা-মাথা না থাকা"*), character voice swapping, and awkward robotic pauses. NextGen Drama Engine completely overhauls this paradigm:

### 1. Lossless Stream-Copy Pre-Merge
- Concatenates 5 to 50+ raw episodes losslessly into a single timeline in **under 1.5 seconds** using FFmpeg stream copy (`-c copy`).
- Eliminates per-episode rendering overhead, audio desync across episode transitions, and redundant model loading.

### 2. GPU Demucs Audio Separation
- Uses Meta's `htdemucs` with 120-second sliding GPU chunks.
- Cleanly separates speech dialogue (`vocals.wav`) from cinematic BGM and sound effects (`no_vocals.wav`).

### 3. Natural Speech-Turn Consolidation (SenseVoice-Small + VAD + Punc)
- Employs `iic/SenseVoiceSmall` combined with `FSMN-VAD` and `punc_ct-transformer`.
- Intelligent speech-turn consolidation unifies micro-pauses within continuous monologues, yielding complete natural sentence turns (5s-9.5s) instead of fragmented 0.9s clips.

### 4. Single-Thread Conversational Narrative Memory (ChatGPT Web Automation)
- Automates ChatGPT Web UI via Playwright in a persistent Chrome session.
- **Master Drama Dossier**: Analyzes the opening dialogues to establish the overarching plot conflict, protagonist identity, and character relationships.
- **Conversational Scene Scripting**: Translates entire drama scenes within the same conversation thread, maintaining emotional momentum, plot callbacks, and colloquial Hindi phrasing.
- **Semantic Speaker Attribution**: Deduces character roles (Heroine, Maid, Villain, Father, Mother, King) based on honorifics (*"小姐"*, *"陛下"*, *"爹娘"*) and story relationships, completely overriding noisy audio diarization clusters.
- **Speaker Continuity Guard**: Prevents mid-sentence character voice flipping.

### 5. Formant-Preserving Time-Stretch & Silence Trimming (librubberband)
- **Zero Pitch Drift**: Uses FFmpeg's native `librubberband` (`rubberband=tempo=X:formant=preserved:pitchq=quality`) for tempo adjustments. Characters keep their exact vocal timbre and pitch without the chipmunk effect or pitch drifting.
- **Dead-Silence Removal**: Automatically strips encoder priming silence (`silenceremove`) so speech starts at 0.0s immediately, preventing swallowed opening syllables.
- **Character Budget & Pacing**: Prompts enforce a 14 char/sec limit in ChatGPT so lines naturally fit their visual duration without requiring aggressive time compression.
- **Character-Bound Emotional TTS**: Maps character roles to fine-tuned Microsoft Edge-TTS voice identities:
  - **Heroine (नायिका)**: `hi-IN-SwaraNeural` (+12Hz, +4%)
  - **Maid (दासी)**: `hi-IN-SwaraNeural` (+6Hz, +2%)
  - **Villain (खलनायक)**: `hi-IN-MadhurNeural` (-30Hz, -4%)
  - **Father (पिता)**: `hi-IN-MadhurNeural` (-22Hz, -6%)
  - **Mother (माता)**: `hi-IN-SwaraNeural` (-20Hz, -5%)
  - **Hero (नायक)**: `hi-IN-MadhurNeural` (+0Hz, +0%)

### 6. Memory-Safe Chunked Video Composition & EBU R128 Normalization
- Renders final video in safe 5-minute chunks to guarantee **zero FFmpeg memory overflow crashes**.
- **EBU R128 Broadcast Normalization**: Applies standard `-23 LUFS` (`loudnorm=I=-23:LRA=7:tp=-1.5`) across the mixed soundtrack for studio-grade dialogue-to-music balance.
- High-aesthetic translucent black pill box (`BorderStyle: 3`, `Nirmala UI`) completely covers original burned-in Chinese subtitles.
- Dynamic sidechain ducking automatically compresses background music under dialogue.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- FFmpeg installed and on PATH
- NVIDIA GPU with CUDA support

### Installation
```bash
git clone https://github.com/Ashik-Siddike/nextgen_drama_engine.git
cd nextgen_drama_engine
pip install -r requirements.txt
playwright install chromium
```

### Running Dubbing Pipeline
```bash
python pipeline.py --episodes path/to/ep_*.mp4 --series-name my_drama
```

---

## 📁 Project Structure
```
nextgen_drama_engine/
├── config/
│   ├── characters.json         # Character voice identities and pitch locks
│   └── settings.json           # Pipeline configuration (ASR, Demucs, TTS)
├── modules/
│   ├── merger.py               # Lossless stream copy video merger
│   ├── audio_separator.py      # Demucs GPU vocal/BGM separator
│   ├── transcriber.py          # SenseVoice-Small ASR + turn consolidation
│   ├── chatgpt_browser.py      # Playwright ChatGPT web automation
│   ├── voiceover.py            # Edge-TTS synthesis + runway sync
│   └── video_composer.py       # Chunked video composition + ASS styling
└── pipeline.py                 # Master CLI orchestrator
```

## 📜 License
MIT License.
