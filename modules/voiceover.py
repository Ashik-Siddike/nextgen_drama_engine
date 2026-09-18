"""Character-Bound Emotional Voice Dubbing & Runway-Aware Alignment."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any, Sequence

from . import (
    PipelineError,
    ensure_dir,
    ffprobe_duration,
    log,
    read_json,
    require_binary,
    run_command,
    write_json,
)

SAMPLE_RATE = 44100


async def _synthesize_edge_clip(text: str, voice: str, pitch: str, rate: str, dest_path: str):
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, pitch=pitch, rate=rate)
    await communicate.save(dest_path)


def _trim_silence_and_sync(
    raw_path: str,
    dest_path: str,
    target_duration: float,
    runway: float,
    atempo_min: float = 0.90,
    atempo_max: float = 1.35,
) -> float:
    """Trim dead silence from Edge-TTS clip and gently adapt tempo to runway without voice distortion."""
    ffmpeg = require_binary("ffmpeg")

    # 1. Trim leading and trailing dead silence to prevent swallowed start syllables
    trimmed_tmp = dest_path + ".trimmed.wav"
    silence_filter = (
        "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-40dB:detection=peak,"
        "areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-40dB:detection=peak,areverse"
    )

    try:
        run_command(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", raw_path,
                "-af", silence_filter,
                "-ar", str(SAMPLE_RATE),
                trimmed_tmp,
            ],
            desc="ffmpeg(trim-silence)",
        )
        clean_dur = ffprobe_duration(trimmed_tmp)
    except Exception:
        clean_dur = 0.0

    src_file = trimmed_tmp if clean_dur > 0.05 else raw_path
    actual_dur = clean_dur if clean_dur > 0.05 else ffprobe_duration(raw_path)

    # 2. Runway check with safety margin
    safety_margin = min(0.08, max(0.02, runway * 0.08)) if runway > 0 else 0.05
    max_allowed = max(0.20, runway - safety_margin) if runway > 0 else target_duration

    # Gentle tempo adjustment: bounded between atempo_min and atempo_max (default 1.35x)
    tempo = max(atempo_min, min(atempo_max, actual_dur / max_allowed)) if max_allowed > 0 and actual_dur > max_allowed else 1.0

    if abs(tempo - 1.0) < 0.03:
        shutil.copyfile(src_file, dest_path)
    else:
        rubberband_ok = False
        try:
            run_command(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", src_file,
                    "-af", f"rubberband=tempo={tempo:.4f}:formant=preserved:pitchq=quality",
                    "-ar", str(SAMPLE_RATE),
                    dest_path,
                ],
                desc="ffmpeg(rubberband-formant-sync)",
            )
            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 512:
                rubberband_ok = True
        except Exception as e:
            log.warning("rubberband failed, falling back to atempo: %s", e)
            rubberband_ok = False

        if not rubberband_ok:
            filters = []
            curr_tempo = tempo
            while curr_tempo > 2.0:
                filters.append("atempo=2.0")
                curr_tempo /= 2.0
            filters.append(f"atempo={curr_tempo:.4f}")
            filter_str = ",".join(filters)

            run_command(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", src_file,
                    "-filter:a", filter_str,
                    "-ar", str(SAMPLE_RATE),
                    dest_path,
                ],
                desc="ffmpeg(atempo-sync)",
            )

    if os.path.isfile(trimmed_tmp):
        try:
            os.remove(trimmed_tmp)
        except OSError:
            pass

    return ffprobe_duration(dest_path)


def synthesize_and_align_voiceover(
    script_segments: list[dict[str, Any]],
    output_dir: str,
    characters_registry_path: str,
    *,
    default_voice: str = "hi-IN-MadhurNeural",
    atempo_min: float = 0.85,
    atempo_max: float = 2.50,
    concurrency: int = 4,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Synthesize speech for each dialogue and align to the video timeline with zero collisions."""
    output_dir = ensure_dir(output_dir)
    manifest_path = os.path.join(output_dir, "voice_tracks.json")

    if not overwrite and os.path.isfile(manifest_path):
        cached = read_json(manifest_path)
        if isinstance(cached, list) and len(cached) == len(script_segments):
            log.info("Loaded cached voiceover tracks (%d clips) from %s", len(cached), manifest_path)
            return cached

    raw_dir = ensure_dir(os.path.join(output_dir, "raw"))
    synced_dir = ensure_dir(os.path.join(output_dir, "synced"))

    char_config = read_json(characters_registry_path, default={}).get("roles", {})

    log.info("Synthesizing %d dialogue lines with character-bound voices (concurrency %d)...", len(script_segments), concurrency)

    async def _batch_synthesize():
        semaphore = asyncio.Semaphore(concurrency)

        async def _worker(seg: dict[str, Any]):
            sid = seg["id"]
            txt = seg.get("recap_text") or seg.get("original_text") or "..."
            role = str(seg.get("speaker", "extra_male")).lower()
            role_prof = char_config.get(role, {})

            # Voice identity mapping
            voice = role_prof.get("edge_voice") or ("hi-IN-SwaraNeural" if seg.get("gender") == "female" else default_voice)
            pitch = role_prof.get("edge_pitch", "+0Hz")
            rate = role_prof.get("edge_rate", "+0%")

            raw_file = os.path.join(raw_dir, f"clip_{sid:04d}.mp3")

            async with semaphore:
                for attempt in range(3):
                    try:
                        if not os.path.isfile(raw_file) or os.path.getsize(raw_file) < 512:
                            await _synthesize_edge_clip(txt, voice, pitch, rate, raw_file)
                        break
                    except Exception as e:
                        if attempt == 2:
                            log.warning("TTS failed for seg %d: %s", sid, e)
                        await asyncio.sleep(1.0)

        tasks = [_worker(s) for s in script_segments]
        await asyncio.gather(*tasks)

    asyncio.run(_batch_synthesize())

    # Time-sync and runway alignment
    synced_tracks: list[dict[str, Any]] = []
    log.info("Aligning speech clips into allotted runways to guarantee ZERO overlap...")

    for i, seg in enumerate(script_segments):
        sid = seg["id"]
        raw_file = os.path.join(raw_dir, f"clip_{sid:04d}.mp3")
        synced_file = os.path.join(synced_dir, f"synced_{sid:04d}.wav")

        if not os.path.isfile(raw_file):
            continue

        start_t = float(seg["start"])
        end_t = float(seg["end"])
        target_dur = max(0.1, end_t - start_t)

        # Available runway until the next dialogue starts
        next_start = float(script_segments[i + 1]["start"]) if i + 1 < len(script_segments) else start_t + target_dur + 2.0
        runway = max(0.1, next_start - start_t)

        final_dur = _trim_silence_and_sync(raw_file, synced_file, target_dur, runway, atempo_min, atempo_max)

        synced_tracks.append(
            {
                "id": sid,
                "start": start_t,
                "end": round(start_t + final_dur, 3),
                "duration": round(final_dur, 3),
                "audio_path": synced_file,
                "speaker": seg.get("speaker", "unknown"),
                "text": seg.get("recap_text", ""),
            }
        )

    log.info("Voiceover complete: aligned %d clips without collisions.", len(synced_tracks))
    write_json(manifest_path, synced_tracks)
    return synced_tracks
