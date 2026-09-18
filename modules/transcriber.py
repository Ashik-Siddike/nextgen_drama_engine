"""SenseVoice-Small + VAD Chinese ASR Transcriber."""

from __future__ import annotations

import os
import re
from typing import Any

from . import (
    PipelineError,
    ensure_dir,
    log,
    read_json,
    write_json,
)

_MODEL_CACHE: dict[str, Any] = {}


def _get_sensevoice_pipeline(device: str = "cuda"):
    global _MODEL_CACHE
    if "sensevoice" in _MODEL_CACHE:
        return _MODEL_CACHE["sensevoice"]

    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise PipelineError("funasr is not installed. Run: pip install funasr modelscope") from exc

    device_str = "cuda:0" if device.startswith("cuda") else "cpu"
    log.info("Loading SenseVoice-Small + FSMN-VAD + CAM++ on %s...", device_str)

    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
        trust_remote_code=False,
        device=device_str,
    )
    _MODEL_CACHE["sensevoice"] = model
    return model


def transcribe_chinese_vocals(
    vocals_path: str,
    output_json_path: str,
    *,
    device: str = "cuda",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Transcribe dialogue vocals with SenseVoice-Small, returning timestamped segments."""
    if not overwrite and os.path.isfile(output_json_path):
        cached = read_json(output_json_path)
        if isinstance(cached, list) and cached:
            log.info("Loaded cached transcript (%d segments) from %s", len(cached), output_json_path)
            return cached

    if not os.path.isfile(vocals_path):
        raise PipelineError(f"Vocals audio not found: {vocals_path}")

    model = _get_sensevoice_pipeline(device=device)
    log.info("Transcribing dialogue timeline: %s...", os.path.basename(vocals_path))

    res = model.generate(
        input=vocals_path,
        language="zh",
        use_itn=True,
        batch_size_s=60,
        merge_vad=False,
    )

    raw_sentences = []
    if res and isinstance(res, list) and len(res) > 0:
        raw_sentences = res[0].get("sentence_info", [])

    segments: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_sentences):
        txt = str(item.get("text", "")).strip()
        cleaned_text = re.sub(r"<\|.*?\|>", "", txt).strip()
        # Strip leading and trailing Chinese/Western punctuation
        cleaned_text = re.sub(r"^[，。！？；：“”‘’、,\.!\?\s]+|[，。！？；：“”‘’、,\.!\?\s]+$", "", cleaned_text).strip()
        # Ignore empty or punctuation-only strings
        if not cleaned_text or not re.search(r"[\w\u4e00-\u9fff]", cleaned_text):
            continue

        start_s = round(float(item.get("start", 0)) / 1000.0, 3)
        end_s = round(float(item.get("end", 0)) / 1000.0, 3)
        duration = round(end_s - start_s, 3)
        if duration < 0.25:
            continue

        spk_id = str(item.get("spk", f"spk_{idx % 4}"))
        if not spk_id.startswith("spk_"):
            spk_id = f"spk_{spk_id}"

        emo = "neutral"
        if "<|ANGRY|>" in txt:
            emo = "angry"
        elif "<|HAPPY|>" in txt:
            emo = "happy"
        elif "<|SAD|>" in txt:
            emo = "sad"
        elif "<|FEARFUL|>" in txt:
            emo = "fearful"

        segments.append(
            {
                "id": len(segments),
                "start": start_s,
                "end": end_s,
                "duration": duration,
                "original_text": cleaned_text,
                "speaker_cluster": spk_id,
                "detected_emotion": emo,
            }
        )

    # Consolidate rapid micro-segments (gap <= 0.65s, combined <= 9.5s) into natural speech turns
    consolidated: list[dict[str, Any]] = []
    for s in segments:
        if not consolidated:
            consolidated.append(dict(s))
            continue
        prev = consolidated[-1]
        gap = s["start"] - prev["end"]
        if -0.10 <= gap <= 0.65 and (prev["duration"] + max(0.0, gap) + s["duration"]) <= 9.5:
            prev["end"] = s["end"]
            prev["duration"] = round(prev["end"] - prev["start"], 3)
            prev["original_text"] = prev["original_text"] + "，" + s["original_text"]
            if s.get("detected_emotion") != "neutral":
                prev["detected_emotion"] = s["detected_emotion"]
        else:
            s_copy = dict(s)
            s_copy["id"] = len(consolidated)
            consolidated.append(s_copy)

    for idx, seg in enumerate(consolidated):
        seg["id"] = idx

    log.info(
        "Transcription complete: extracted %d raw -> %d natural dialogue turns.",
        len(segments),
        len(consolidated),
    )
    write_json(output_json_path, consolidated)
    return consolidated
