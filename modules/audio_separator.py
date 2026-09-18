"""Demucs GPU Audio Separation with 2-Minute Sliding Chunks for Long Videos."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Sequence

from . import (
    PipelineError,
    ensure_dir,
    ffprobe_duration,
    log,
    require_binary,
    run_command,
)

SAMPLE_RATE = 44100
CHANNELS = 2


def _extract_wav(video_path: str, wav_path: str) -> str:
    ffmpeg = require_binary("ffmpeg")
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", video_path,
            "-vn",
            "-map", "0:a:0",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            wav_path,
        ],
        desc="ffmpeg(extract-audio)",
    )
    return wav_path


def _locate_stems(search_root: str) -> tuple[str | None, str | None]:
    vocals = no_vocals = None
    for current, _dirs, files in os.walk(search_root):
        for name in files:
            lowered = name.lower()
            if not lowered.endswith(".wav"):
                continue
            path = os.path.join(current, name)
            if lowered.startswith("no_vocals"):
                no_vocals = path
            elif lowered.startswith("vocals"):
                vocals = path
    return vocals, no_vocals


def separate_vocals_and_bgm(
    video_path: str,
    output_dir: str,
    *,
    model: str = "htdemucs",
    device: str = "cuda",
    chunk_sec: int = 120,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Isolate dialogue vocals and background instrumental tracks safely on GPU."""
    output_dir = ensure_dir(output_dir)
    vocals_final = os.path.join(output_dir, "vocals.wav")
    no_vocals_final = os.path.join(output_dir, "no_vocals.wav")

    if not overwrite and os.path.isfile(vocals_final) and os.path.isfile(no_vocals_final):
        log.info("Audio separation stems already cached in %s", output_dir)
        return vocals_final, no_vocals_final

    source_wav = os.path.join(output_dir, "source_full.wav")
    _extract_wav(video_path, source_wav)
    dur = ffprobe_duration(source_wav)
    log.info("Extracted audio track: %.1fs (%.1f mins)", dur, dur / 60)

    demucs_bin = shutil.which("demucs") or [sys.executable, "-m", "demucs"]
    demucs_cmd = [demucs_bin] if isinstance(demucs_bin, str) else demucs_bin

    if dur > 300.0:
        log.info("Audio duration exceeds 300s. Activating 2-minute GPU chunked separator...")
        temp_dir = ensure_dir(os.path.join(output_dir, "_temp_chunks"))
        chunks_dir = ensure_dir(os.path.join(temp_dir, "chunks"))
        ffmpeg = require_binary("ffmpeg")

        chunk_pattern = os.path.join(chunks_dir, "chunk_%04d.wav")
        run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", source_wav,
                "-f", "segment",
                "-segment_time", str(chunk_sec),
                "-c", "copy",
                chunk_pattern,
            ],
            desc="ffmpeg(chunk-split)",
        )

        chunk_files = sorted(
            [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(".wav")]
        )
        log.info("Separating %d audio chunk(s) via Demucs (%s on %s)...", len(chunk_files), model, device)

        v_chunks: list[str] = []
        nv_chunks: list[str] = []

        for idx, cpath in enumerate(chunk_files, start=1):
            c_out = ensure_dir(os.path.join(temp_dir, f"out_{idx:04d}"))
            argv = demucs_cmd + [
                "--two-stems=vocals",
                "-n", model,
                "-o", c_out,
                "--filename", "{stem}.{ext}",
                "-j", "1",
                "-d", device,
                cpath,
            ]
            run_command(argv, desc=f"demucs(chunk {idx}/{len(chunk_files)})")
            v_stem, nv_stem = _locate_stems(c_out)
            if not v_stem or not nv_stem:
                raise PipelineError(f"Demucs separation failed on chunk {idx}")
            v_dest = os.path.join(temp_dir, f"v_{idx:04d}.wav")
            nv_dest = os.path.join(temp_dir, f"nv_{idx:04d}.wav")
            shutil.move(v_stem, v_dest)
            shutil.move(nv_stem, nv_dest)
            v_chunks.append(v_dest)
            nv_chunks.append(nv_dest)

        def _concat_wavs(files: list[str], dest: str):
            lst = f"{dest}.list.txt"
            with open(lst, "w", encoding="utf-8") as h:
                for f in files:
                    h.write(f"file '{os.path.abspath(f).replace(chr(92), '/')}'\n")
            run_command(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", lst,
                    "-c", "copy",
                    dest,
                ],
                desc="ffmpeg(concat-stems)",
            )
            try:
                os.remove(lst)
            except OSError:
                pass

        _concat_wavs(v_chunks, vocals_final)
        _concat_wavs(nv_chunks, no_vocals_final)

        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass
    else:
        out_base = ensure_dir(os.path.join(output_dir, "_demucs"))
        argv = demucs_cmd + [
            "--two-stems=vocals",
            "-n", model,
            "-o", out_base,
            "--filename", "{stem}.{ext}",
            "-j", "1",
            "-d", device,
            source_wav,
        ]
        run_command(argv, desc="demucs(full)")
        v_stem, nv_stem = _locate_stems(out_base)
        if not v_stem or not nv_stem:
            raise PipelineError("Demucs single-pass failed to produce stems.")
        shutil.move(v_stem, vocals_final)
        shutil.move(nv_stem, no_vocals_final)
        try:
            shutil.rmtree(out_base)
        except OSError:
            pass

    try:
        os.remove(source_wav)
    except OSError:
        pass

    log.info("Audio separation complete: vocals & bgm ready.")
    return vocals_final, no_vocals_final
