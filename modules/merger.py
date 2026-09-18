"""Fast Lossless Stream-Copy Merger for Raw Episodes."""

from __future__ import annotations

import os
import re
from typing import Sequence

from . import (
    PipelineError,
    ensure_dir,
    ffprobe_duration,
    log,
    require_binary,
    run_command,
)


def _natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def _escape_concat_path(path: str) -> str:
    normalised = os.path.abspath(path).replace("\\", "/")
    return normalised.replace("'", "'\\''")


def merge_episodes_lossless(
    episode_paths: Sequence[str],
    output_path: str,
    overwrite: bool = False,
) -> str:
    """Merge episodes into a single master video file in seconds via stream-copy."""
    if not episode_paths:
        raise PipelineError("merge_episodes_lossless: No episodes provided to merge.")

    output_path = os.path.abspath(output_path)
    if not overwrite and os.path.isfile(output_path) and os.path.getsize(output_path) > 8192:
        log.info("Merged raw video already exists: %s", output_path)
        return output_path

    ffmpeg = require_binary("ffmpeg")
    sorted_paths = sorted(episode_paths, key=lambda p: _natural_sort_key(os.path.basename(p)))

    concat_list = f"{output_path}.concat.txt"
    ensure_dir(os.path.dirname(output_path))

    with open(concat_list, "w", encoding="utf-8") as handle:
        handle.write("# NextGen Drama Engine Concat List\n")
        for p in sorted_paths:
            if not os.path.isfile(p):
                raise PipelineError(f"Episode file missing: {p}")
            handle.write(f"file '{_escape_concat_path(p)}'\n")

    log.info("Fast-merging %d episode(s) into %s...", len(sorted_paths), os.path.basename(output_path))
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ],
        desc="ffmpeg(concat-copy)",
    )

    try:
        os.remove(concat_list)
    except OSError:
        pass

    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 8192:
        raise PipelineError(f"Lossless merge failed to produce {output_path}")

    dur = ffprobe_duration(output_path)
    log.info("Lossless merge complete: %.1fs (%.1f mins) -> %s", dur, dur / 60, output_path)
    return output_path
