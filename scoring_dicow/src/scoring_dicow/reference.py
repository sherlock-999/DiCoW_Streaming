from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from .text_norm import EnglishTextNormalizer

TS_SEGMENT_RE = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>(.*?)<\|([0-9]+(?:\.[0-9]+)?)\|>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
SPK_HEADER_RE = re.compile(r"Speaker\s*\d+\s*:", re.I)
TIER_RE = re.compile(
    r'item\s*\[\d+\]:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"(.*?)"\s*'
    r"xmin\s*=\s*[-+0-9.eE]+\s*xmax\s*=\s*[-+0-9.eE]+\s*"
    r"intervals:\s*size\s*=\s*\d+\s*(.*?)(?=\n\s*item\s*\[\d+\]:|\Z)",
    re.S,
)
INTERVAL_RE = re.compile(
    r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([-+0-9.eE]+)\s*xmax\s*=\s*([-+0-9.eE]+)\s*text\s*=\s*\"(.*?)\"",
    re.S,
)


def clean_text(text: str) -> str:
    text = SPK_HEADER_RE.sub(" ", str(text))
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_textgrid(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    seg_i = 0
    for tier_name, tier_body in TIER_RE.findall(content):
        speaker = clean_text(tier_name) or "spk"
        for start, end, text in INTERVAL_RE.findall(tier_body):
            words = clean_text(text.replace('""', '"'))
            st = float(start)
            et = float(end)
            if not words or et <= st:
                continue
            rows.append(
                {
                    "speaker": speaker,
                    "start_time": st,
                    "end_time": et,
                    "words": words,
                    "segment_index": seg_i,
                }
            )
            seg_i += 1
    rows.sort(key=lambda x: (x["start_time"], x["end_time"], x["speaker"]))
    return rows


def aggregate_rows_by_session_speaker(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["session_id"], row["speaker"])].append(row)
    out = []
    for (session_id, speaker), segs in grouped.items():
        segs.sort(key=lambda x: x.get("start_time", 0.0))
        words = " ".join(s.get("words", "") for s in segs).strip()
        out.append(
            {"session_id": session_id, "speaker": speaker, "start_time": 0, "end_time": 0, "words": words}
        )
    return out


def normalize_rows(rows: list[dict], normalizer: EnglishTextNormalizer | None = None) -> list[dict]:
    normalizer = normalizer or EnglishTextNormalizer(
        standardize_numbers=False,
        standardize_numbers_rev=True,
        remove_fillers=True,
    )
    out = []
    for row in rows:
        item = dict(row)
        item["words"] = re.sub(r"\s+", " ", normalizer(item.get("words", ""))).strip()
        out.append(item)
    return out


def rows_to_annotation(rows: list[dict]) -> Annotation:
    ann = Annotation()
    for idx, row in enumerate(rows):
        st = float(row.get("start_time", 0.0))
        et = float(row.get("end_time", st))
        if et <= st:
            continue
        ann[Segment(st, et), f"{idx}"] = str(row.get("speaker", "spk"))
    return ann


def compute_der(reference_multi_rows: list[dict], hypothesis_multi_rows: list[dict]) -> dict:
    by_ref = defaultdict(list)
    by_hyp = defaultdict(list)
    for row in reference_multi_rows:
        by_ref[row["session_id"]].append(row)
    for row in hypothesis_multi_rows:
        by_hyp[row["session_id"]].append(row)

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    scored = 0
    for session_id in sorted(by_ref.keys()):
        metric(rows_to_annotation(by_ref[session_id]), rows_to_annotation(by_hyp.get(session_id, [])))
        scored += 1

    details = metric[:]
    total = float(details.get("total", 0.0))
    fa = float(details.get("false alarm", 0.0))
    ms = float(details.get("missed detection", 0.0))
    sc = float(details.get("confusion", 0.0))
    der = float(abs(metric))
    return {
        "sessions_scored": scored,
        "total": total,
        "correct": float(details.get("correct", 0.0)),
        "false_alarm": fa,
        "missed_speech": ms,
        "speaker_confusion": sc,
        "rates": {
            "DER": der,
            "FA": 0.0 if total <= 0 else fa / total,
            "MS": 0.0 if total <= 0 else ms / total,
            "SC": 0.0 if total <= 0 else sc / total,
        },
    }
