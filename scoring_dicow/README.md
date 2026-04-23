# scoring_dicow

Standalone scoring toolkit for DiCoW outputs. It consumes existing `hypothesis_multi.jsonl` files and scores them against datasets stored as:

```text
<testset_root>/<dataset>/
  audio/
  textgrid/
  rttm/
```

## Install

```bash
cd /home3/adnan/DICOW/scoring_dicow
pip install -e .
```

## Config

Start from [`config.example.yaml`](/home3/adnan/DICOW/scoring_dicow/config.example.yaml).

```yaml
testset_root: /home3/adnan/DICOW/mt-asr-data-prep/testset_tse
output_root: ./output
collar: 5
datasets:
  ami:
    predictions: /path/to/ami/hypothesis_multi.jsonl
    mapping: ami
  nsf:
    predictions: /path/to/nsf/hypothesis_multi.jsonl
    mapping: nsf
  l2m:
    predictions: /path/to/l2m/shard_*/hypothesis_multi.jsonl
    mapping: l2m
```

## Usage

```bash
score-dicow --config config.example.yaml
score-dicow --config config.example.yaml --datasets ami nsf
score-dicow --config config.example.yaml --output-root ./custom_output --collar 3
```

## Outputs

For each dataset, the tool writes compatibility-oriented artifacts:

- `reference_multi.jsonl`, `reference_wer.jsonl`
- `hypothesis_multi.jsonl`, `hypothesis_wer.jsonl`
- `*.seglst.json`
- `normalized_eval/*`
- `missing_sessions.json`
- `failures.txt`
- `der_summary.json`
- `run_summary.json`

It also writes a top-level `metric_summary_normalized.json`.
