#!/usr/bin/env python3
"""
Single-file streaming diarization + transcription.

Chunks audio at Sortformer's cadence (480 ms by default), collects a
configurable transcription window, then runs DiCoW and prints each
speaker's output immediately. No files are written.

Usage:
    python run_streaming.py audio.wav
    python run_streaming.py audio.wav --config config.yaml --interval 5.0
"""

import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoFeatureExtractor, AutoTokenizer

from model.DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from model.Streaming_Sortformer.Streaming_Sortformer import StreamingSortformer
from dicow_inference import create_lower_uppercase_mapping
from dicow_pipeline import DiCoW_Pipeline

SAMPLING_RATE = 16_000

_TS_RE = re.compile(r'<\|([\d.]+)\|>')


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dicow(model_path: str, device: torch.device) -> DiCoW_Pipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = DiCoWForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    create_lower_uppercase_mapping(tokenizer)
    model.set_tokenizer(tokenizer)
    return DiCoW_Pipeline(
        model,
        speaker_embedding_model=None,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_segments(text: str, time_offset: float):
    """
    Parse DiCoW timestamp format '<|t0|>words<|t1|>' and yield
    (abs_start, abs_end, text) tuples. Skips empty segments.
    """
    matches = list(_TS_RE.finditer(text))
    for i in range(0, len(matches) - 1, 2):
        t0 = float(matches[i].group(1))
        t1 = float(matches[i + 1].group(1))
        seg = text[matches[i].end(): matches[i + 1].start()].strip()
        if seg:
            yield time_offset + t0, time_offset + t1, seg


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def stream(audio_path: str, cfg, interval_s: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print("Loading StreamingSortformer …")
    sortformer = StreamingSortformer(cfg.sortformer)

    print("Loading DiCoW …")
    pipeline = load_dicow(cfg.dicow.model_path, device)
    print("Models ready.\n")

    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True, dtype=np.float32)
    total_s = len(audio) / SAMPLING_RATE
    print(f"Audio    : {Path(audio_path).name}  ({total_s:.2f} s)")
    print(f"Interval : {interval_s} s\n")

    chunk_samples    = sortformer.chunk_samples
    interval_samples = int(interval_s * SAMPLING_RATE)

    sortformer.reset()

    acc_audio = np.zeros(0, dtype=np.float32)
    acc_preds: torch.Tensor | None = None
    window_start = 0.0
    pos = 0

    while pos < len(audio):
        # ── Sortformer chunk ────────────────────────────────────────────────
        end        = min(pos + chunk_samples, len(audio))
        true_audio = audio[pos:end]
        true_len   = len(true_audio)

        chunk = true_audio
        if true_len < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - true_len))

        preds = sortformer.process_chunk(chunk, true_length=true_len)  # [n_spk, frames@12.5fps]

        acc_audio = np.concatenate([acc_audio, true_audio])
        acc_preds = preds if acc_preds is None else torch.cat([acc_preds, preds], dim=-1)

        pos = end

        # ── Transcription window full (or end of file) ───────────────────
        if len(acc_audio) < interval_samples and pos < len(audio):
            continue

        window_end = window_start + len(acc_audio) / SAMPLING_RATE
        duration_s = len(acc_audio) / SAMPLING_RATE
        dicow_mask = sortformer.preds_to_dicow_mask(acc_preds, duration_s)

        print(f"── {window_start:.1f}s – {window_end:.1f}s " + "─" * 40)

        pipeline.diarization_mask = dicow_mask
        with torch.inference_mode():
            result = pipeline(
                {"array": acc_audio, "sampling_rate": SAMPLING_RATE},
                return_timestamps=True,
            )
        pipeline.diarization_mask = None

        for spk_idx, spk_text in enumerate(result.get("per_spk_outputs", [])):
            if not spk_text or not spk_text.strip():
                continue
            for t0, t1, seg in extract_segments(spk_text, window_start):
                print(f"  Speaker {spk_idx}  [{t0:.2f}s – {t1:.2f}s]  {seg}")

        window_start = window_end
        acc_audio    = np.zeros(0, dtype=np.float32)
        acc_preds    = None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Streaming diarization + transcription for a single audio file")
    parser.add_argument("audio_path",  help="Path to a 16 kHz mono WAV file")
    parser.add_argument("--config",    default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--interval",  type=float, default=None,
                        help="Transcription window in seconds (overrides config)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    interval_s = args.interval or cfg.pipeline.get("transcription_interval_s", 5.0)

    stream(args.audio_path, cfg, interval_s)


if __name__ == "__main__":
    main()
