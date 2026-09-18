"""NextGen Drama Engine: Core Utilities & Logging."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nextgen_drama")


class PipelineError(Exception):
    """Domain exception raised when a pipeline stage fails."""


def ensure_dir(path: str) -> str:
    """Ensure directory exists and return absolute normalized path."""
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def read_json(path: str, default: Any = None) -> Any:
    """Safely load JSON from path or return default."""
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        log.warning("Could not read JSON from %s: %s", path, exc)
        return default


def write_json(path: str, data: Any, indent: int = 2) -> str:
    """Atomically write data as pretty-printed JSON."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)
    return path


def require_binary(name: str, hint: str = "") -> str:
    """Verify that an external binary is on PATH and return its absolute path."""
    binary = shutil.which(name)
    if not binary:
        msg = f"Required binary '{name}' was not found on PATH."
        if hint:
            msg += f" {hint}"
        raise PipelineError(msg)
    return binary


def run_command(
    argv: Sequence[str],
    desc: str = "",
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess command cleanly."""
    cmd_desc = desc or os.path.basename(argv[0])
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise PipelineError(f"{cmd_desc} could not be launched: {exc}") from exc

    if check and proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise PipelineError(f"{cmd_desc} failed with exit code {proc.returncode}:\n{err[-800:]}")
    return proc


def ffprobe_duration(path: str) -> float:
    """Return video or audio duration in seconds via ffprobe."""
    if not os.path.isfile(path):
        return 0.0
    ffprobe = require_binary("ffprobe")
    try:
        proc = run_command(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                path,
            ],
            desc="ffprobe(duration)",
            check=False,
        )
        data = json.loads(proc.stdout or "{}")
        return max(0.0, float(data.get("format", {}).get("duration", 0.0)))
    except Exception:
        return 0.0
