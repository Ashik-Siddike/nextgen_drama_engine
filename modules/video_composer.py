"""Memory-Safe 5-Minute Chunked Video Composer with Subtitle Plates & Audio Ducking."""

from __future__ import annotations

import math
import os
import shutil
from typing import Any, Sequence

from . import (
    PipelineError,
    ensure_dir,
    ffprobe_duration,
    log,
    require_binary,
    run_command,
)

SAMPLE_RATE = 44100


def _build_narration_bed(
    synced_tracks: Sequence[dict[str, Any]],
    total_duration: float,
    output_path: str,
) -> str:
    """Mix all speech clips onto a single timeline WAV bed."""
    ffmpeg = require_binary("ffmpeg")
    output_path = os.path.abspath(output_path)
    ensure_dir(os.path.dirname(output_path))

    usable = [t for t in synced_tracks if os.path.isfile(t.get("audio_path", ""))]
    if not usable:
        run_command(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
                "-t", f"{max(0.1, total_duration):.3f}",
                "-acodec", "pcm_s16le", output_path,
            ],
            desc="ffmpeg(silent-bed)",
        )
        return output_path

    # Chunk into sets of 40 to avoid FFmpeg command line length overflow
    chunk_size = 40
    chunks = [usable[i : i + chunk_size] for i in range(0, len(usable), chunk_size)]
    temp_beds: list[str] = []

    for idx, chunk in enumerate(chunks):
        part_wav = f"{output_path}_part{idx:03d}.wav"
        argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        argv += ["-f", "lavfi", "-t", f"{max(0.1, total_duration):.3f}", "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}"]

        for c in chunk:
            argv += ["-i", c["audio_path"]]

        parts = ["[0:a]aresample=44100[base]"]
        labels = ["[base]"]
        for c_i, c in enumerate(chunk, start=1):
            delay_ms = max(0, int(round(float(c.get("start", 0.0)) * 1000)))
            parts.append(f"[{c_i}:a]aresample=44100,aformat=channel_layouts=stereo,adelay={delay_ms}:all=1[v{c_i}]")
            labels.append(f"[v{c_i}]")

        filter_str = ";".join(parts) + f";{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[out]"
        argv += ["-filter_complex", filter_str, "-map", "[out]", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", part_wav]
        run_command(argv, desc=f"ffmpeg(bed-chunk-{idx+1})")
        temp_beds.append(part_wav)

    if len(temp_beds) == 1:
        shutil.move(temp_beds[0], output_path)
    else:
        argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        for tb in temp_beds:
            argv += ["-i", tb]
        filter_str = "".join([f"[{i}:a]" for i in range(len(temp_beds))]) + f"amix=inputs={len(temp_beds)}:duration=first:normalize=0[out]"
        argv += ["-filter_complex", filter_str, "-map", "[out]", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_path]
        run_command(argv, desc="ffmpeg(bed-merge)")
        for tb in temp_beds:
            try:
                os.remove(tb)
            except OSError:
                pass

    return output_path


def _write_subtitles_ass(
    script_segments: Sequence[dict[str, Any]],
    output_ass_path: str,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Generate professional ASS subtitle file for high-aesthetic rendering."""
    ensure_dir(os.path.dirname(output_ass_path))
    margin_v = int(height * 0.27) if height > width else int(height * 0.12)
    font_size = 42 if height > width else 26

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Nirmala UI,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,0,0,3,14,0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def _format_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:01d}:{m:02d}:{s:05.2f}"

    def _wrap_subtitle(text: str, max_chars: int = 34) -> str:
        words = text.split()
        if not words:
            return ""
        lines_list = []
        cur_line = []
        cur_len = 0
        for w in words:
            w_len = len(w)
            if cur_len + w_len + (1 if cur_line else 0) > max_chars and cur_line:
                lines_list.append(" ".join(cur_line))
                cur_line = [w]
                cur_len = w_len
            else:
                cur_line.append(w)
                cur_len += w_len + (1 if len(cur_line) > 1 else 0)
        if cur_line:
            lines_list.append(" ".join(cur_line))
        return "\\N".join(lines_list)

    def _split_into_cues(text: str, start: float, end: float) -> list[tuple[float, float, str]]:
        import re
        raw_sents = [s.strip() for s in re.split(r"[।!?.\n]+", text) if s.strip()]
        if not raw_sents or len(text) <= 50 or (end - start) <= 5.0:
            return [(start, end, text)]

        total_len = sum(len(s) for s in raw_sents)
        total_dur = max(0.5, end - start)
        cues = []
        cur_st = start
        for i, s in enumerate(raw_sents):
            frac = len(s) / total_len
            dur = total_dur * frac
            cur_et = end if i == len(raw_sents) - 1 else round(cur_st + dur, 3)
            cues.append((round(cur_st, 3), cur_et, s))
            cur_st = cur_et
        return cues

    lines = [header]
    for seg in script_segments:
        txt = str(seg.get("text") or seg.get("recap_text") or "").strip()
        if not txt:
            continue
        st = float(seg["start"])
        et = float(seg["end"])
        cues = _split_into_cues(txt, st, et)
        for c_st, c_et, c_text in cues:
            formatted_st = _format_time(c_st)
            formatted_et = _format_time(c_et)
            wrapped = _wrap_subtitle(c_text, max_chars=34)
            lines.append(f"Dialogue: 0,{formatted_st},{formatted_et},Default,,0,0,0,,{wrapped}\n")

    with open(output_ass_path, "w", encoding="utf-8") as h:
        h.writelines(lines)
    return output_ass_path


def render_composed_drama(
    raw_video_path: str,
    no_vocals_path: str,
    voice_tracks: list[dict[str, Any]],
    output_video_path: str,
    *,
    chunk_minutes: int = 5,
    burn_subtitles: bool = True,
    overwrite: bool = False,
) -> str:
    """Compose master video safely in 5-minute chunks with audio ducking and subtitle plates."""
    output_video_path = os.path.abspath(output_video_path)
    if not overwrite and os.path.isfile(output_video_path) and os.path.getsize(output_video_path) > 8192:
        log.info("Composed video already exists: %s", output_video_path)
        return output_video_path

    ensure_dir(os.path.dirname(output_video_path))
    ffmpeg = require_binary("ffmpeg")
    total_dur = ffprobe_duration(raw_video_path)

    # 1. Build full audio narration bed
    work_dir = ensure_dir(os.path.join(os.path.dirname(output_video_path), "_compose_tmp"))
    narration_wav = os.path.join(work_dir, "narration_bed.wav")
    log.info("Building full narration bed (%.1fs timeline)...", total_dur)
    _build_narration_bed(voice_tracks, total_dur, narration_wav)

    # 2. Mix audio bed with dynamic sidechain ducking under BGM
    final_audio = os.path.join(work_dir, "final_soundtrack.wav")
    log.info("Mixing soundtrack with dynamic sidechain ducking...")
    duck_filter = (
        "[0:a]aresample=44100,aformat=channel_layouts=stereo,volume=0.28[bgm];"
        "[1:a]aresample=44100,aformat=channel_layouts=stereo,highpass=f=80,treble=g=2.5:f=3500[voice];"
        "[voice]asplit=2[sc][v_clean];"
        "[bgm][sc]sidechaincompress=threshold=0.03:ratio=8.0:attack=20:release=400[ducked_bgm];"
        "[ducked_bgm][v_clean]amix=inputs=2:duration=first:normalize=0[mixed];"
        "[mixed]loudnorm=I=-23:LRA=7:tp=-1.5[out]"
    )
    run_command(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", no_vocals_path,
            "-i", narration_wav,
            "-filter_complex", duck_filter,
            "-map", "[out]",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            final_audio,
        ],
        desc="ffmpeg(audio-ducking)",
    )

    # 3. Safe 5-Minute Chunked Video Rendering (eliminates expression memory overflow)
    chunk_sec = chunk_minutes * 60
    total_chunks = max(1, math.ceil(total_dur / chunk_sec))
    log.info("Rendering video in %d safe 5-minute chunk(s) to guarantee zero memory overflow...", total_chunks)

    ass_path = os.path.join(work_dir, "subtitles.ass")
    _write_subtitles_ass(voice_tracks, ass_path)

    rendered_chunks: list[str] = []
    for c_idx in range(total_chunks):
        c_start = c_idx * chunk_sec
        c_len = min(chunk_sec, total_dur - c_start)
        c_out = os.path.join(work_dir, f"render_chunk_{c_idx:03d}.mp4")

        # Subtitle filter for this chunk
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        vf = f"subtitles='{escaped_ass}'" if burn_subtitles else "null"

        log.info("  Encoding chunk %d/%d (%.1fs - %.1fs)...", c_idx + 1, total_chunks, c_start, c_start + c_len)

        # Try NVENC first, fallback to CPU libx264
        nvenc_ok = False
        try:
            run_command(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{c_start:.3f}", "-t", f"{c_len:.3f}", "-i", raw_video_path,
                    "-ss", f"{c_start:.3f}", "-t", f"{c_len:.3f}", "-i", final_audio,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-vf", vf,
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    c_out,
                ],
                desc=f"ffmpeg(nvenc-chunk-{c_idx+1})",
            )
            nvenc_ok = True
        except Exception:
            pass

        if not nvenc_ok:
            run_command(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{c_start:.3f}", "-t", f"{c_len:.3f}", "-i", raw_video_path,
                    "-ss", f"{c_start:.3f}", "-t", f"{c_len:.3f}", "-i", final_audio,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    c_out,
                ],
                desc=f"ffmpeg(cpu-chunk-{c_idx+1})",
            )

        rendered_chunks.append(c_out)

    # 4. Final Fast Concat of rendered chunks
    if len(rendered_chunks) == 1:
        shutil.move(rendered_chunks[0], output_video_path)
    else:
        concat_txt = os.path.join(work_dir, "chunks_concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as h:
            for rc in rendered_chunks:
                h.write(f"file '{rc.replace(chr(92), '/')}'\n")

        log.info("Concatenating %d rendered chunks into final master movie...", len(rendered_chunks))
        run_command(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_txt,
                "-c", "copy",
                "-movflags", "+faststart",
                output_video_path,
            ],
            desc="ffmpeg(concat-final)",
        )

    try:
        shutil.rmtree(work_dir)
    except OSError:
        pass

    log.info("Master dubbed movie complete: %s (%.1fs)", output_video_path, ffprobe_duration(output_video_path))
    return output_video_path
