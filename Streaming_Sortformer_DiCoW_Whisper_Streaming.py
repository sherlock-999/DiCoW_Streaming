#!/usr/bin/env python3
"""
Streaming diarization + transcription.

Each 480 ms audio chunk is fed to StreamingSortformer, which returns raw
sigmoid predictions at 12.5 fps. The accumulated predictions are converted to
a 50 fps binary mask via preds_to_dicow_mask() (onset/offset hysteresis),
and only the new frames are passed to each speaker's DiarisationOnlineASRProcessor.

DiarisationOnlineASRProcessor detects speech start/end transitions per speaker
and triggers DiCoW transcription after transcription_interval_s of active speech.

Usage:
    python Streaming_Sortformer_DiCoW_Whisper_Streaming.py audio.wav
    python Streaming_Sortformer_DiCoW_Whisper_Streaming.py audio.wav --config config.yaml --window 5.0
"""

import argparse
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
from DiCoW_Whisper_Streaming import DiCoWASR, DiarisationOnlineASRProcessor

SAMPLING_RATE = 16000


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


def stream(audio_path: str, cfg, online_chunk_size: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading StreamingSortformer …")
    sortformer = StreamingSortformer(cfg.sortformer)
    n_speakers = sortformer.n_speakers

    print("Loading DiCoW …")
    pipeline = load_dicow(cfg.dicow.model_path, device)
    print("Models ready.\n")

    # One DiCoWASR + DiarisationOnlineASRProcessor per speaker.
    # All DiCoWASR instances share the same pipeline.
    processors = []
    for spk_idx in range(n_speakers):
        asr = DiCoWASR(pipeline, lan="en", target_speaker_idx=spk_idx)
        proc = DiarisationOnlineASRProcessor(
            online_chunk_size=online_chunk_size,
            asr=asr,
            target_speaker_idx=spk_idx,
        )
        processors.append(proc)

    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True, dtype=np.float32)
    total_s = len(audio) / SAMPLING_RATE
    chunk_ms = sortformer.chunk_samples / SAMPLING_RATE * 1000
    print(f"Audio  : {Path(audio_path).name}  ({total_s:.2f} s)")
    print(f"Chunk  : {chunk_ms:.0f} ms (Sortformer)")
    print(f"Window : {online_chunk_size:.1f} s (transcription)\n")

    sortformer.reset()

    # Accumulate raw Sortformer predictions across all chunks so that
    # preds_to_dicow_mask() can apply onset/offset hysteresis over the full history.
    acc_preds = torch.zeros(n_speakers, 0)
    acc_samples = 0
    prev_mask_frames = 0  # how many 50fps frames we have already handed to processors

    pos = 0
    while pos < len(audio):
        end = min(pos + sortformer.chunk_samples, len(audio))
        true_audio = audio[pos:end]
        true_len = len(true_audio)

        # Pad last chunk to full chunk_samples
        chunk = true_audio
        if true_len < sortformer.chunk_samples:
            chunk = np.pad(chunk, (0, sortformer.chunk_samples - true_len))

        # [n_spk, ~6] raw sigmoid at 12.5 fps
        preds = sortformer.process_chunk(chunk, true_length=true_len)

        acc_preds = torch.cat([acc_preds, preds], dim=-1)
        acc_samples += true_len
        acc_duration_s = acc_samples / SAMPLING_RATE

        # Convert full accumulated predictions to a 50fps binary mask with hysteresis.
        # Slice only the new frames produced by this chunk.
        full_mask = sortformer.preds_to_dicow_mask(acc_preds, acc_duration_s)
        mask_chunk = full_mask[:, prev_mask_frames:]   # [n_spk, new_50fps_frames]
        prev_mask_frames = full_mask.shape[-1]

        for proc in processors:
            proc.insert_audio_chunk(true_audio, mask_chunk)

        for spk_idx, proc in enumerate(processors):
            beg, end_t, text = proc.process_iter()
            if text:
                print(f"  [{beg:.2f}s – {end_t:.2f}s] Speaker {spk_idx}: {text}")

        pos = end

    # Flush remaining hypotheses
    print("\n--- End of audio ---")
    for spk_idx, proc in enumerate(processors):
        beg, end_t, text = proc.finish()
        if text:
            print(f"  [{beg:.2f}s – {end_t:.2f}s] Speaker {spk_idx}: {text}")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming diarization + transcription (Sortformer + DiCoW)"
    )
    parser.add_argument("audio_path", help="Path to a 16 kHz mono WAV file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--window", type=float, default=None,
        help="Transcription window in seconds (overrides config pipeline.transcription_interval_s)"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    online_chunk_size = args.window or cfg.pipeline.get("transcription_interval_s", 5.0)

    stream(args.audio_path, cfg, online_chunk_size)


if __name__ == "__main__":
    main()
