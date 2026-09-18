"""NextGen Drama Engine: Master Pipeline Orchestrator."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# Ensure nextgen_drama_engine is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules import (
    PipelineError,
    ensure_dir,
    ffprobe_duration,
    log,
    read_json,
    write_json,
)
from modules import merger
from modules import audio_separator
from modules import transcriber
from modules import chatgpt_browser
from modules import voiceover
from modules import video_composer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NextGen Drama Engine: Cinematic Narrative Dubbing.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        help="Input raw video episode files or wildcards (e.g. storage/raw/ep_*.mp4)",
    )
    parser.add_argument(
        "--series-name",
        default="master_drama",
        help="Output project / series stem name",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "config", "settings.json"),
        help="Path to settings.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore all caches and regenerate from scratch",
    )
    parser.add_argument(
        "--skip-browser",
        action="store_true",
        help="Skip ChatGPT browser automation and reuse cached script or raw text",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    config = read_json(args.config, default={})
    if not config:
        log.error("Could not load configuration from %s", args.config)
        return 1

    storage = config.get("storage_paths", {})
    raw_dir = ensure_dir(os.path.join(BASE_DIR, storage.get("raw", "storage/raw")))
    audio_dir = ensure_dir(os.path.join(BASE_DIR, storage.get("audio", "storage/audio")))
    transcripts_dir = ensure_dir(os.path.join(BASE_DIR, storage.get("transcripts", "storage/transcripts")))
    scripts_dir = ensure_dir(os.path.join(BASE_DIR, storage.get("scripts", "storage/scripts")))
    tts_dir = ensure_dir(os.path.join(BASE_DIR, storage.get("tts", "storage/tts")))
    output_dir = ensure_dir(os.path.join(BASE_DIR, storage.get("output", "storage/output")))

    # Discover episodes
    input_episodes: list[str] = []
    if args.episodes:
        for ep_pattern in args.episodes:
            matched = glob.glob(ep_pattern)
            if matched:
                input_episodes.extend(matched)
            elif os.path.isfile(ep_pattern):
                input_episodes.append(ep_pattern)
    else:
        # Check raw directory default
        input_episodes = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")))

    if not input_episodes:
        log.error("No raw episode files found! Specify via --episodes path/to/ep_*.mp4")
        return 1

    stem = args.series_name
    log.info("=" * 80)
    log.info("  🎬 NEXTGEN DRAMA ENGINE: %s", stem.upper())
    log.info("  Input Episodes: %d file(s)", len(input_episodes))
    log.info("=" * 80)

    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # STAGE 1: Lossless Stream-Copy Pre-Merge
    # -------------------------------------------------------------------------
    merged_raw_path = os.path.join(raw_dir, f"{stem}_raw.mp4")
    log.info("\n📦 [1/5] Lossless Pre-Merge...")
    merger.merge_episodes_lossless(input_episodes, merged_raw_path, overwrite=args.force)

    # -------------------------------------------------------------------------
    # STAGE 2: Demucs GPU Audio Separation
    # -------------------------------------------------------------------------
    log.info("\n🎵 [2/5] Demucs GPU Vocal & BGM Separation...")
    sep_cfg = config.get("audio_separation", {})
    vocals_wav, no_vocals_wav = audio_separator.separate_vocals_and_bgm(
        merged_raw_path,
        os.path.join(audio_dir, stem),
        model=sep_cfg.get("model", "htdemucs"),
        device=sep_cfg.get("device", "cuda"),
        chunk_sec=int(sep_cfg.get("chunk_sec", 120)),
        overwrite=args.force,
    )

    # -------------------------------------------------------------------------
    # STAGE 3: SenseVoice-Small ASR Transcription
    # -------------------------------------------------------------------------
    log.info("\n📝 [3/5] SenseVoice-Small Timestamp Transcription...")
    asr_cfg = config.get("asr", {})
    transcript_json = os.path.join(transcripts_dir, f"{stem}_transcript.json")
    segments = transcriber.transcribe_chinese_vocals(
        vocals_wav,
        transcript_json,
        device=asr_cfg.get("device", "cuda"),
        overwrite=args.force,
    )

    # -------------------------------------------------------------------------
    # STAGE 4: ChatGPT Browser Automation (Conversational Scene Dubbing)
    # -------------------------------------------------------------------------
    log.info("\n🎭 [4/5] ChatGPT Web Automation Dramatic Scripting...")
    script_json = os.path.join(scripts_dir, f"{stem}_cinematic_script.json")
    b_cfg = config.get("browser", {})

    if args.skip_browser and os.path.isfile(script_json):
        log.info("Skipping browser automation (--skip-browser); using cached script.")
        script_segments = read_json(script_json)
    else:
        script_segments = chatgpt_browser.generate_cinematic_script_via_chatgpt(
            segments,
            script_json,
            target_language=config.get("target_language", "Hindi"),
            batch_size=int(b_cfg.get("batch_dialogue_count", 45)),
            profile_dir=os.path.join(BASE_DIR, b_cfg.get("profile_dir", "storage/browser_profile")),
            overwrite=args.force,
        )

    # -------------------------------------------------------------------------
    # STAGE 5: Multi-Character Voice Synthesis & Time-Sync
    # -------------------------------------------------------------------------
    log.info("\n🎙️ [5/5] Voiceover Synthesis & Runway-Aware Alignment...")
    tts_cfg = config.get("tts", {})
    char_registry = os.path.join(BASE_DIR, "config", "characters.json")
    voice_tracks = voiceover.synthesize_and_align_voiceover(
        script_segments,
        os.path.join(tts_dir, stem),
        char_registry,
        default_voice=tts_cfg.get("default_voice", "hi-IN-MadhurNeural"),
        atempo_min=float(tts_cfg.get("atempo_min", 0.90)),
        atempo_max=float(tts_cfg.get("atempo_max", 1.35)),
        concurrency=int(tts_cfg.get("concurrency", 4)),
        overwrite=args.force,
    )

    # -------------------------------------------------------------------------
    # STAGE 6: Memory-Safe 5-Minute Chunked Video Composition
    # -------------------------------------------------------------------------
    log.info("\n🎬 [6/6] Safe Chunked Video Composition with Subtitle Plates & Audio Ducking...")
    vc_cfg = config.get("video_composition", {})
    output_master_path = os.path.join(output_dir, f"{stem}_dubbed_master.mp4")

    final_video = video_composer.render_composed_drama(
        merged_raw_path,
        no_vocals_wav,
        voice_tracks,
        output_master_path,
        chunk_minutes=int(vc_cfg.get("chunk_minutes", 5)),
        burn_subtitles=bool(vc_cfg.get("burn_subtitles", True)),
        overwrite=args.force,
    )

    elapsed = time.perf_counter() - start_time
    total_dur = ffprobe_duration(final_video)

    log.info("=" * 80)
    log.info("  🏆 NEXTGEN DUBBING PIPELINE COMPLETE!")
    log.info("  Output Video : %s", final_video)
    log.info("  Duration     : %.1fs (%.2f mins)", total_dur, total_dur / 60)
    log.info("  Total Time   : %.1fs (%.2f mins)", elapsed, elapsed / 60)
    log.info("=" * 80)
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        sys.exit(run_pipeline(args))
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Caches preserved.")
        sys.exit(130)
    except PipelineError as pe:
        log.error("Pipeline failure: %s", pe)
        sys.exit(1)


if __name__ == "__main__":
    main()
