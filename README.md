# DiCoW Streaming

Real-time diarization-conditioned ASR using [DiCoW](https://github.com/BUTSpeechFIT/DiCoW) (Diarization-Conditioned Whisper). The system takes a binary speaker diarization mask alongside audio and produces per-speaker transcripts with word-level timestamps — continuously, chunk by chunk.

## How it works

```
Audio chunks  ──►  DiarisationOnlineASRProcessor  ──►  DiCoWASR  ──►  per-speaker text
Diar mask    ──►  (detects speech start/end)            (DiCoW pipeline)
```

1. **Diarization mask** `[num_speakers, num_frames]` arrives at 50 fps alongside raw audio.
2. `DiarisationOnlineASRProcessor` tracks speech activity per speaker, trims silence at boundaries, and feeds audio+mask slices to `OnlineASRProcessor`.
3. `DiCoWASR` wraps `DiCoW_Pipeline` (a HuggingFace pipeline subclass) and extracts timestamped word segments from DiCoW's output format (`<|t|>word<|t|>`).
4. `HypothesisBuffer` deduplicates overlapping hypotheses across chunks and commits stable prefixes.

## Installation

```bash
conda create -n socow python=3.10
conda activate socow
pip install -r requirements.txt
```

The DiCoW model must be placed locally (or downloaded from HuggingFace) at the path specified in `config.yaml`:

```yaml
dicow_model: ./model/DiCoW
```

## Usage

### Streaming evaluation against ground-truth RTTM + TextGrid

Edit `config.yaml` to point at your audio, RTTM, and TextGrid directories, then run:

```bash
conda activate socow
python evaluate_streaming_wer.py --config config.yaml
```

Output files are written to `output_dir` (set in config):
- `<stem>_transcript.txt` — timestamped hypothesis vs. reference per speaker
- `streaming_<chunk>s.txt` — per-file and overall WER / cpWER / RTF summary
- `debug.log` — detailed frame-level debug trace

### Batch inference from a manifest

```bash
conda activate socow
python dicow_inference.py \
    --manifest path/to/dataset_manifest.json \
    --diar_mask_dir path/to/masks/ \
    --output_dir path/to/output/
```

The manifest is a JSONL file where each line has `audio_filepath`. Diarization masks are `.pt` files named `<audio_stem>_mask.pt`.

### Programmatic use

```python
import torch
from dicow_inference import DiCoWTranscriber

transcriber = DiCoWTranscriber(dicow_model_path="./model/DiCoW")

# masks: {audio_name: torch.Tensor[num_speakers, num_frames]}
transcriber.transcribe_with_masks(
    manifest_path="dataset_manifest.json",
    masks=masks,
    output_dir="./output",
)
```

## Configuration

Key fields in `config.yaml`:

| Key | Description |
|-----|-------------|
| `dicow_model` | Path to local DiCoW model directory |
| `audio_dir` / `audio_path` | Directory of `.wav` files, or single file |
| `rttm_dir` / `rttm_path` | RTTM diarization annotation |
| `textgrid_dir` / `textgrid_path` | Praat TextGrid reference transcripts |
| `output_dir` | Where results are written |
| `streaming.min_chunk_size` | Chunk size in seconds (default: 1.0) |

## Project structure

```
DiCoW_Streaming/
├── DiCoW_Whisper_Streaming.py   # DiCoWASR, OnlineASRProcessor, HypothesisBuffer
├── dicow_pipeline.py            # DiCoW_Pipeline (HuggingFace pipeline subclass)
├── dicow_inference.py           # DiCoWTranscriber + batch inference CLI
├── evaluate_streaming_wer.py    # End-to-end streaming WER evaluation
├── config.yaml                  # Runtime configuration
├── model/DiCoW/                 # Local DiCoW model weights & config
├── scoring_dicow/               # TextGrid parsing + text normalisation (subpackage)
└── whisper_streaming/           # Silero VAD iterator helper
```

## Metrics

The evaluator reports:
- **WER** — word error rate aggregated per file
- **cpWER** — concatenated minimum-permutation WER per speaker
- **RTF** — real-time factor (processing time / audio duration) per speaker
