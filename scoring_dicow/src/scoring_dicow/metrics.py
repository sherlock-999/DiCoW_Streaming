from __future__ import annotations

import itertools
import json
import subprocess
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

from .config import DatasetConfig
from .io import read_jsonl_rows, write_json, write_jsonl, write_seglst_json
from .mappings import SESSION_MAPPERS
from .reference import aggregate_rows_by_session_speaker, compute_der, normalize_rows, parse_textgrid


def run_metric(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)


def load_hypothesis_map(pattern: str) -> dict[str, list[dict]]:
    paths = [Path(pattern)] if "*" not in pattern and "?" not in pattern and "[" not in pattern else [Path(p) for p in sorted(glob(pattern))]
    by_session = defaultdict(list)
    for path in paths:
        for row in read_jsonl_rows(path):
            by_session[row["session_id"]].append(row)
    return by_session


def validate_dataset_layout(dataset_root: Path) -> list[str]:
    failures = []
    for name in ("audio", "textgrid", "rttm"):
        if not (dataset_root / name).is_dir():
            failures.append(f"Missing directory: {dataset_root / name}")
    return failures


def edit_distance_words(a: str, b: str) -> int:
    a_t = a.split() if a else []
    b_t = b.split() if b else []
    n, m = len(a_t), len(b_t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a_t[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b_t[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    return dp[n][m]


def reassign_hypothesis_speakers(reference_rows: list[dict], hypothesis_rows: list[dict]) -> tuple[list[dict], dict]:
    ref_norm = normalize_rows(reference_rows)
    hyp_norm = normalize_rows(hypothesis_rows)

    by_ref = defaultdict(list)
    by_hyp = defaultdict(list)
    for row in ref_norm:
        by_ref[row["session_id"]].append(row)
    for row in hyp_norm:
        by_hyp[row["session_id"]].append(row)

    original_by_hyp = defaultdict(list)
    for row in hypothesis_rows:
        original_by_hyp[row["session_id"]].append(row)

    reassigned = []
    debug = {}
    for session_id, ref_rows_sess in by_ref.items():
        hyp_rows_sess_norm = by_hyp.get(session_id, [])
        hyp_rows_sess_orig = original_by_hyp.get(session_id, [])
        ref_speakers = sorted({r["speaker"] for r in ref_rows_sess})
        hyp_speakers = sorted({h["speaker"] for h in hyp_rows_sess_norm})

        ref_concat = {}
        for spk in ref_speakers:
            segs = sorted((r for r in ref_rows_sess if r["speaker"] == spk), key=lambda x: x.get("start_time", 0.0))
            ref_concat[spk] = " ".join(s["words"] for s in segs).strip()

        hyp_concat = {}
        for spk in hyp_speakers:
            segs = sorted(
                (h for h in hyp_rows_sess_norm if h["speaker"] == spk),
                key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0)),
            )
            hyp_concat[spk] = " ".join(s["words"] for s in segs).strip()

        pair_cost = {
            ref_spk: {hyp_spk: edit_distance_words(ref_concat[ref_spk], hyp_concat[hyp_spk]) for hyp_spk in hyp_speakers}
            for ref_spk in ref_speakers
        }

        mapping = {}
        rule = "none"
        if not hyp_speakers:
            rule = "no_hypothesis"
        elif len(hyp_speakers) >= len(ref_speakers):
            best = None
            for perm in itertools.permutations(hyp_speakers, len(ref_speakers)):
                total = sum(pair_cost[ref_speakers[i]][perm[i]] for i in range(len(ref_speakers)))
                if best is None or total < best[0]:
                    best = (total, perm)
            for i, ref_spk in enumerate(ref_speakers):
                mapping[best[1][i]] = ref_spk
            rule = "min_cost_distinct_permutation"
        else:
            remaining_h = set(hyp_speakers)
            for ref_spk in ref_speakers:
                if not remaining_h:
                    break
                best_h = min(remaining_h, key=lambda h: pair_cost[ref_spk][h])
                mapping[best_h] = ref_spk
                remaining_h.remove(best_h)
            rule = "greedy_undercomplete_hyp"

        for row in hyp_rows_sess_orig:
            old_spk = row["speaker"]
            if old_spk in mapping:
                item = dict(row)
                item["speaker"] = mapping[old_spk]
                reassigned.append(item)

        debug[session_id] = {
            "session_id": session_id,
            "reference_speakers": ref_speakers,
            "hypothesis_speakers": hyp_speakers,
            "assignment_rule": rule,
            "mapping_hyp_to_ref": mapping,
            "pairwise_cost": pair_cost,
        }

    reassigned.sort(
        key=lambda x: (x["session_id"], x.get("start_time", 0.0), x.get("end_time", 0.0), x.get("segment_index", 0))
    )
    return reassigned, debug


def prepare_metrics(raw_dir: Path, collar: int) -> None:
    py = sys.executable
    ref_multi_rows = read_jsonl_rows(raw_dir / "reference_multi.jsonl")
    hyp_multi_rows = read_jsonl_rows(raw_dir / "hypothesis_multi.jsonl")
    ref_wer_rows = read_jsonl_rows(raw_dir / "reference_wer.jsonl")
    hyp_wer_rows = read_jsonl_rows(raw_dir / "hypothesis_wer.jsonl")

    ref_multi_seglst = raw_dir / "reference_multi.seglst.json"
    hyp_multi_seglst = raw_dir / "hypothesis_multi.seglst.json"
    ref_wer_agg_seglst = raw_dir / "reference_wer_agg.seglst.json"
    hyp_wer_agg_seglst = raw_dir / "hypothesis_wer_agg.seglst.json"
    ref_multi_agg_seglst = raw_dir / "reference_multi_agg.seglst.json"
    hyp_multi_agg_seglst = raw_dir / "hypothesis_multi_agg.seglst.json"

    write_seglst_json(ref_multi_seglst, ref_multi_rows)
    write_seglst_json(hyp_multi_seglst, hyp_multi_rows)
    write_seglst_json(ref_wer_agg_seglst, aggregate_rows_by_session_speaker(ref_wer_rows))
    write_seglst_json(hyp_wer_agg_seglst, aggregate_rows_by_session_speaker(hyp_wer_rows))
    write_seglst_json(ref_multi_agg_seglst, aggregate_rows_by_session_speaker(ref_multi_rows))
    write_seglst_json(hyp_multi_agg_seglst, aggregate_rows_by_session_speaker(hyp_multi_rows))

    metric_specs = [
        ("wer", ref_wer_agg_seglst, hyp_wer_agg_seglst, []),
        ("cpwer", ref_multi_agg_seglst, hyp_multi_agg_seglst, []),
        ("tcpwer", ref_multi_seglst, hyp_multi_seglst, ["--collar", str(collar)]),
        ("tcorcwer", ref_multi_seglst, hyp_multi_seglst, ["--collar", str(collar)]),
    ]
    for metric_name, ref_path, hyp_path, extra_args in metric_specs:
        run_metric(
            [
                py,
                "-m",
                "meeteval.wer",
                metric_name,
                "-r",
                str(ref_path),
                "-h",
                str(hyp_path),
                *extra_args,
                "--average-out",
                str(raw_dir / f"{metric_name}_average.json"),
                "--per-reco-out",
                str(raw_dir / f"{metric_name}_per_reco.json"),
            ]
        )


def prepare_normalized_metrics(raw_dir: Path, norm_dir: Path, collar: int) -> None:
    norm_dir.mkdir(parents=True, exist_ok=True)
    ref_multi_norm = normalize_rows(read_jsonl_rows(raw_dir / "reference_multi.jsonl"))
    hyp_multi_norm = normalize_rows(read_jsonl_rows(raw_dir / "hypothesis_multi.jsonl"))
    ref_wer_norm = normalize_rows(read_jsonl_rows(raw_dir / "reference_wer.jsonl"))
    hyp_wer_norm = normalize_rows(read_jsonl_rows(raw_dir / "hypothesis_wer.jsonl"))

    write_jsonl(norm_dir / "reference_multi.norm.jsonl", ref_multi_norm)
    write_jsonl(norm_dir / "hypothesis_multi.norm.jsonl", hyp_multi_norm)
    write_jsonl(norm_dir / "reference_wer.norm.jsonl", ref_wer_norm)
    write_jsonl(norm_dir / "hypothesis_wer.norm.jsonl", hyp_wer_norm)

    py = sys.executable
    ref_multi_seglst = norm_dir / "reference_multi.norm.seglst.json"
    hyp_multi_seglst = norm_dir / "hypothesis_multi.norm.seglst.json"
    ref_wer_agg_seglst = norm_dir / "reference_wer_agg.norm.seglst.json"
    hyp_wer_agg_seglst = norm_dir / "hypothesis_wer_agg.norm.seglst.json"
    ref_multi_agg_seglst = norm_dir / "reference_multi_agg.norm.seglst.json"
    hyp_multi_agg_seglst = norm_dir / "hypothesis_multi_agg.norm.seglst.json"

    write_seglst_json(ref_multi_seglst, ref_multi_norm)
    write_seglst_json(hyp_multi_seglst, hyp_multi_norm)
    write_seglst_json(ref_wer_agg_seglst, aggregate_rows_by_session_speaker(ref_wer_norm))
    write_seglst_json(hyp_wer_agg_seglst, aggregate_rows_by_session_speaker(hyp_wer_norm))
    write_seglst_json(ref_multi_agg_seglst, aggregate_rows_by_session_speaker(ref_multi_norm))
    write_seglst_json(hyp_multi_agg_seglst, aggregate_rows_by_session_speaker(hyp_multi_norm))

    metric_specs = [
        ("wer", ref_wer_agg_seglst, hyp_wer_agg_seglst, []),
        ("cpwer", ref_multi_agg_seglst, hyp_multi_agg_seglst, []),
        ("tcpwer", ref_multi_seglst, hyp_multi_seglst, ["--collar", str(collar)]),
        ("tcorcwer", ref_multi_seglst, hyp_multi_seglst, ["--collar", str(collar)]),
    ]
    for metric_name, ref_path, hyp_path, extra_args in metric_specs:
        run_metric(
            [
                py,
                "-m",
                "meeteval.wer",
                metric_name,
                "-r",
                str(ref_path),
                "-h",
                str(hyp_path),
                *extra_args,
                "--average-out",
                str(norm_dir / f"{metric_name}_average.norm.json"),
                "--per-reco-out",
                str(norm_dir / f"{metric_name}_per_reco.norm.json"),
            ]
        )


def score_dataset(dataset_root: Path, dataset_cfg: DatasetConfig, out_dir: Path, collar: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = validate_dataset_layout(dataset_root)
    hyp_map = load_hypothesis_map(dataset_cfg.predictions)
    mapper = SESSION_MAPPERS[dataset_cfg.mapping]

    ref_multi, ref_wer, hyp_multi, hyp_wer = [], [], [], []
    missing = []
    wav_files = sorted((dataset_root / "audio").glob("*.wav"))
    rttm_stems = {p.stem for p in (dataset_root / "rttm").glob("*")}
    for wav in wav_files:
        stem = wav.stem
        session_id = mapper(stem)
        if not session_id:
            missing.append({"file": stem, "reason": "mapping_failed"})
            continue
        tg = dataset_root / "textgrid" / f"{stem}.TextGrid"
        if not tg.exists():
            missing.append({"file": stem, "session": session_id, "reason": "textgrid_missing"})
            continue
        if stem not in rttm_stems:
            failures.append(f"Missing RTTM for {stem}")
        ref_segments = parse_textgrid(tg)
        if not ref_segments:
            missing.append({"file": stem, "session": session_id, "reason": "empty_reference"})
            continue
        hyp_segments = hyp_map.get(session_id, [])
        if not hyp_segments:
            missing.append({"file": stem, "session": session_id, "reason": "hypothesis_missing"})
            continue

        for idx, seg in enumerate(ref_segments):
            ref_multi.append(
                {
                    "session_id": session_id,
                    "speaker": seg["speaker"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": idx,
                }
            )
            ref_wer.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": idx,
                }
            )

        for idx, row in enumerate(sorted(hyp_segments, key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0)))):
            item = dict(row)
            item["session_id"] = session_id
            item["segment_index"] = idx
            hyp_multi.append(item)
            hyp_wer.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": item.get("start_time", 0.0),
                    "end_time": item.get("end_time", 0.0),
                    "words": item.get("words", ""),
                    "segment_index": idx,
                }
            )

    reassigned_hyp_multi, assignment_debug = reassign_hypothesis_speakers(ref_multi, hyp_multi)
    reassigned_hyp_wer = []
    grouped_hyp = defaultdict(list)
    for row in reassigned_hyp_multi:
        grouped_hyp[row["session_id"]].append(row)
    for session_id, segs in grouped_hyp.items():
        segs = sorted(segs, key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0), x.get("segment_index", 0)))
        for idx, item in enumerate(segs):
            reassigned_hyp_wer.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": item.get("start_time", 0.0),
                    "end_time": item.get("end_time", 0.0),
                    "words": item.get("words", ""),
                    "segment_index": idx,
                }
            )

    write_jsonl(out_dir / "reference_multi.jsonl", ref_multi)
    write_jsonl(out_dir / "reference_wer.jsonl", ref_wer)
    write_jsonl(out_dir / "hypothesis_multi.jsonl", reassigned_hyp_multi)
    write_jsonl(out_dir / "hypothesis_wer.jsonl", reassigned_hyp_wer)
    write_json(out_dir / "missing_sessions.json", missing)
    write_json(out_dir / "speaker_assignment_debug.json", assignment_debug)
    (out_dir / "failures.txt").write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")

    prepare_metrics(out_dir, collar=collar)
    norm_dir = out_dir / "normalized_eval"
    prepare_normalized_metrics(out_dir, norm_dir, collar=collar)
    der = compute_der(ref_multi, reassigned_hyp_multi)
    write_json(out_dir / "der_summary.json", der)

    summary = {
        "dataset": dataset_cfg.name,
        "scored_files": len({row["session_id"] for row in ref_multi}),
        "missing_files": len(missing),
        "normalized_metrics": {
            "wer": json.loads((norm_dir / "wer_average.norm.json").read_text(encoding="utf-8"))["error_rate"],
            "cpwer": json.loads((norm_dir / "cpwer_average.norm.json").read_text(encoding="utf-8"))["error_rate"],
            "tcpwer": json.loads((norm_dir / "tcpwer_average.norm.json").read_text(encoding="utf-8"))["error_rate"],
            "tcorcwer": json.loads((norm_dir / "tcorcwer_average.norm.json").read_text(encoding="utf-8"))["error_rate"],
            "DER": der["rates"]["DER"],
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary
