"""
Evaluate DiCoW WER on chunked audio using oracle diarization.

Pipeline per file:
  1. load_reference(textgrid)              →  {speaker: transcript}, {speaker: [(t0,t1,text),…]}
  2. load_diarization_mask(rttm, duration) →  speakers, full_mask [num_spk, total_frames]
  3. transcribe_audio(audio, full_mask)    →  {speaker: hypothesis}
     - per chunk: pipeline.diarization_mask = full_mask[:, f_start:f_end]
  4. Compute WER(reference, hypothesis) per speaker
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import librosa
import torch
import yaml
import jiwer
from tqdm import tqdm

# Use scoring_dicow for TextGrid parsing and text normalization
sys.path.insert(0, str(Path(__file__).parent / "scoring_dicow" / "src"))
from scoring_dicow.reference import parse_textgrid, normalize_rows

from transformers import AutoFeatureExtractor, AutoTokenizer
from model.DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from dicow_pipeline import DiCoW_Pipeline
from dicow_inference import create_lower_uppercase_mapping
from DiCoW_Whisper_Streaming import DiCoWASR, DiarisationOnlineASRProcessor


_DIAR_FPS = 50  # frames per second used by DiCoW diarization mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Reference transcriptions from TextGrid
# ─────────────────────────────────────────────────────────────────────────────

def load_reference(textgrid_path: str) -> Tuple[Dict[str, str], Dict[str, List[Tuple[float, float, str]]]]:
    """
    Parse a Praat TextGrid and return:
      - {speaker: aggregated transcript}   (for WER)
      - {speaker: [(start, end, text), …]} (for timestamped transcript)
    """
    rows = parse_textgrid(Path(textgrid_path))
    by_speaker_words: Dict[str, List[str]] = defaultdict(list)
    by_speaker_segs:  Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for row in rows:
        by_speaker_words[row["speaker"]].append(row["words"])
        by_speaker_segs[row["speaker"]].append(
            (row["start_time"], row["end_time"], row["words"])
        )
    transcripts = {spk: " ".join(words) for spk, words in by_speaker_words.items()}
    return transcripts, dict(by_speaker_segs)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Full diarization mask from RTTM
# ─────────────────────────────────────────────────────────────────────────────

def load_diarization_mask(
    rttm_path: str,
    total_duration: float,
) -> Tuple[List[str], torch.Tensor]:
    """
    Parse an RTTM file and build the full binary diarization mask for the recording.
    Returns:
        speakers   — sorted list of speaker names (defines row order in the mask)
        full_mask  — [num_speakers, total_frames] at _DIAR_FPS
    Per chunk, slice as full_mask[:, f_start:f_end].
    """
    segments: Dict[str, List[Tuple[float, float]]] = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            end      = start + float(parts[4])
            speaker  = parts[7].lower()  # match parse_textgrid() which lowercases tier names
            segments.setdefault(speaker, []).append((start, end))

    speakers     = sorted(segments.keys())
    total_frames = max(1, round(total_duration * _DIAR_FPS))
    full_mask    = torch.zeros(len(speakers), total_frames)

    for i, spk in enumerate(speakers):
        for seg_start, seg_end in segments[spk]:
            f0 = max(0, min(round(seg_start * _DIAR_FPS), total_frames))
            f1 = max(0, min(round(seg_end   * _DIAR_FPS), total_frames))
            full_mask[i, f0:f1] = 1.0

    return speakers, full_mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Transcribe chunked audio
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio: np.ndarray,
    sr: int,
    speakers: List[str],
    full_diar_mask: torch.Tensor,
    pipeline: DiCoW_Pipeline,
    min_chunk_size: float,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Transcribe each speaker independently using DiarisationOnlineASRProcessor + DiCoWASR.
    Returns ({speaker: hypothesis}, {speaker: rtf}).
    """
    total_duration = len(audio) / sr

    per_spk_hyps: Dict[str, str] = {}
    per_spk_segs: Dict[str, List[Tuple[float, float, str]]] = {}
    per_spk_rtf:  Dict[str, float] = {}

    asr = DiCoWASR(pipeline)

    for spk_idx, spk in tqdm(enumerate(speakers)):
        asr.target_speaker_idx = spk_idx
        online = DiarisationOnlineASRProcessor(
            min_chunk_size,
            asr,
            target_speaker_idx=spk_idx,
        )

        committed_segs: List[Tuple[float, float, str]] = []
        chunk_times: List[float] = []
        beg = 0.0

        while beg < total_duration:
            end = min(beg + min_chunk_size, total_duration)

            s0 = round(beg * sr)
            s1 = round(end * sr)
            chunk_audio = audio[s0:s1]

            f_start = round(beg * _DIAR_FPS)
            f_end   = round(end * _DIAR_FPS)
            mask_chunk = full_diar_mask[:, f_start:f_end]

            online.insert_audio_chunk(chunk_audio, mask_chunk)
            t0 = time.perf_counter()
            o = online.process_iter()
            chunk_times.append(time.perf_counter() - t0)
            if o[0] is not None:
                committed_segs.append(o)

            beg = end

        o = online.finish()
        if o[0] is not None:
            committed_segs.append(o)

        per_spk_hyps[spk] = " ".join(seg[2] for seg in committed_segs)
        per_spk_segs[spk] = committed_segs

        spk_proc = sum(chunk_times)
        n = len(chunk_times)
        spk_rtf  = spk_proc / max(total_duration, 1e-9)
        per_spk_rtf[spk] = spk_rtf
        print(f"  [{spk}] RTF={spk_rtf:.3f}  proc={spk_proc:.1f}s  "
              f"chunks={n}  mean={spk_proc/n:.3f}s  max={max(chunk_times):.3f}s")

    return per_spk_hyps, per_spk_segs, per_spk_rtf


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one file
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_file(
    audio_path: str,
    rttm_path: str,
    tg_path: str,
    pipeline: DiCoW_Pipeline,
    min_chunk_size: float,
):
    """
    Evaluate one recording.
    Returns ({speaker: (hypothesis, reference)}, per_spk_segs, per_spk_ref_segs, per_spk_rtf).
    """
    references, ref_segs = load_reference(tg_path)

    audio, sr = librosa.load(audio_path, sr=16_000, mono=True)

    speakers, full_diar_mask = load_diarization_mask(rttm_path, len(audio) / sr)
    hypotheses, per_spk_segs, per_spk_rtf = transcribe_audio(
        audio, sr, speakers, full_diar_mask, pipeline, min_chunk_size,
    )

    return (
        {spk: (hypotheses.get(spk, ""), references.get(spk, "")) for spk in speakers},
        per_spk_segs,
        ref_segs,
        per_spk_rtf,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(model_path: str, device: torch.device) -> DiCoW_Pipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = DiCoWForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype
    ).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    tokenizer         = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    create_lower_uppercase_mapping(tokenizer)
    model.set_tokenizer(tokenizer)
    model.eval()

    return DiCoW_Pipeline(
        model,
        speaker_embedding_model=None,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "debug.log"
    _handler = logging.FileHandler(log_path, mode="w")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _streaming_logger = logging.getLogger("DiCoW_Whisper_Streaming")
    _streaming_logger.setLevel(logging.DEBUG)
    _streaming_logger.addHandler(_handler)
    _streaming_logger.propagate = False

    streaming_cfg  = cfg.get("streaming", {})
    min_chunk_size = streaming_cfg.get("min_chunk_size", 1.0)

    # Support single-file mode (audio_path) or directory mode (audio_dir)
    if "audio_path" in cfg:
        audio_path = Path(cfg["audio_path"])
        file_triples = [(
            audio_path,
            Path(cfg["rttm_path"]),
            Path(cfg["textgrid_path"]),
        )]
    else:
        audio_dir = Path(cfg["audio_dir"])
        rttm_dir  = Path(cfg["rttm_dir"])
        tg_dir    = Path(cfg["textgrid_dir"])
        file_triples = [
            (p, rttm_dir / (p.stem + ".rttm"), tg_dir / (p.stem + ".TextGrid"))
            for p in sorted(audio_dir.glob("*.wav"))
        ]

    print(f"Found {len(file_triples)} file(s).  min_chunk_size: {min_chunk_size}s\n")

    print(f"Loading model: {cfg['dicow_model']}")
    pipeline = load_pipeline(cfg["dicow_model"], device)
    print("Model ready.\n")

    file_wer_lines = []
    total_s = total_d = total_i = total_h = 0
    all_spk_wers: List[float] = []
    all_spk_rtfs: List[float] = []

    for audio_path, rttm_path, tg_path in tqdm(file_triples):
        print(f"Processing: {audio_path.name}")

        results, per_spk_segs, per_spk_ref_segs, per_spk_rtf = evaluate_file(
            str(audio_path), str(rttm_path), str(tg_path), pipeline, min_chunk_size
        )

        # Write timestamped transcript + reference side-by-side for inspection
        transcript_path = output_dir / f"{audio_path.stem}_transcript.txt"
        with open(transcript_path, "w") as tf:
            for spk, (hyp, ref) in results.items():
                tf.write(f"{'─'*60}\n")
                tf.write(f"SPEAKER: {spk}\n")
                tf.write(f"{'─'*60}\n")
                tf.write("HYPOTHESIS (with timestamps):\n")
                for seg_beg, seg_end, seg_text in per_spk_segs.get(spk, []):
                    tf.write(f"  [{seg_beg:8.2f} - {seg_end:8.2f}]  {seg_text}\n")
                tf.write("\nREFERENCE (with timestamps):\n")
                for seg_beg, seg_end, seg_text in per_spk_ref_segs.get(spk, []):
                    tf.write(f"  [{seg_beg:8.2f} - {seg_end:8.2f}]  {seg_text}\n")
                tf.write("\n")

        file_s = file_d = file_i = file_h = 0
        spk_lines = []
        for spk, (hyp, ref) in results.items():
            if not ref.strip():
                continue
            norm_rows = normalize_rows([{"words": ref}, {"words": hyp}])
            ref_n, hyp_n = norm_rows[0]["words"], norm_rows[1]["words"]
            m = jiwer.process_words(ref_n, hyp_n)
            file_s += m.substitutions
            file_d += m.deletions
            file_i += m.insertions
            file_h += m.hits
            spk_wer = (m.substitutions + m.deletions + m.insertions) / max(1, m.substitutions + m.deletions + m.hits)
            all_spk_wers.append(spk_wer)
            all_spk_rtfs.append(per_spk_rtf.get(spk, 0.0))
            spk_line = (f"    [{spk}] cpWER={spk_wer:.2%}  RTF={per_spk_rtf.get(spk, 0.0):.3f}  "
                        f"(S={m.substitutions} D={m.deletions} I={m.insertions} H={m.hits})")
            print(spk_line)
            spk_lines.append(spk_line.strip())

        file_wer = (file_s + file_d + file_i) / max(1, file_s + file_d + file_h)
        line = (f"{audio_path.stem}  WER={file_wer:.2%}  "
                f"(S={file_s} D={file_d} I={file_i} H={file_h})")
        print(f"  {line}")
        file_wer_lines.append(line)
        file_wer_lines.extend(spk_lines)

        total_s += file_s
        total_d += file_d
        total_i += file_i
        total_h += file_h

    overall     = (total_s + total_d + total_i) / max(1, total_s + total_d + total_h)
    avg_cpwer   = sum(all_spk_wers) / max(1, len(all_spk_wers))
    avg_rtf     = sum(all_spk_rtfs) / max(1, len(all_spk_rtfs))
    print(f"\n── Dataset WER " + "─" * 50)
    print(f"  Overall WER  = {overall:.2%}  (S={total_s} D={total_d} I={total_i} H={total_h})")
    print(f"  Avg cpWER    = {avg_cpwer:.2%}  (mean over {len(all_spk_wers)} speakers)")
    print(f"  Avg RTF      = {avg_rtf:.3f}  (mean per-speaker, over {len(all_spk_rtfs)} speakers)")

    report = (
        "\n".join(file_wer_lines)
        + f"\n\nOverall WER  = {overall:.4f}"
        + f"\nAvg cpWER    = {avg_cpwer:.4f}"
        + f"\nAvg RTF      = {avg_rtf:.4f}\n"
    )
    (output_dir / f"streaming_{min_chunk_size}s.txt").write_text(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
