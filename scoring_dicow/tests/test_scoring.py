import json
from pathlib import Path

from scoring_dicow.config import load_config
from scoring_dicow.config import DatasetConfig
from scoring_dicow.mappings import map_ami_session, map_l2m_session, map_nsf_session
from scoring_dicow.metrics import load_hypothesis_map, reassign_hypothesis_speakers, score_dataset
from scoring_dicow.reference import aggregate_rows_by_session_speaker, parse_textgrid


def test_parse_textgrid():
    path = Path(__file__).parent / "fixtures" / "sample.TextGrid"
    rows = parse_textgrid(path)
    assert len(rows) == 2
    assert rows[0]["speaker"] == "speaker a"
    assert rows[0]["words"] == "hello there"


def test_session_mappings():
    assert map_ami_session("sdm_ES2004a-2") == "ES2004a"
    assert map_nsf_session("sdm_MTG_32041_sc_rockfall_2-38") == "MTG_32041/sc_rockfall_2"
    assert map_l2m_session("2300-131720-0029_2830-3980-0067") == "2300-131720-0029_2830-3980-0067"


def test_load_hypothesis_map_glob(tmp_path):
    shard1 = tmp_path / "shard_0"
    shard2 = tmp_path / "shard_1"
    shard1.mkdir()
    shard2.mkdir()
    (shard1 / "hypothesis_multi.jsonl").write_text(
        json.dumps({"session_id": "a", "speaker": "s0", "start_time": 0, "end_time": 1, "words": "one"}) + "\n",
        encoding="utf-8",
    )
    (shard2 / "hypothesis_multi.jsonl").write_text(
        json.dumps({"session_id": "b", "speaker": "s0", "start_time": 0, "end_time": 1, "words": "two"}) + "\n",
        encoding="utf-8",
    )
    out = load_hypothesis_map(str(tmp_path / "shard_*" / "hypothesis_multi.jsonl"))
    assert sorted(out.keys()) == ["a", "b"]


def test_aggregate_rows_by_session_speaker():
    rows = [
        {"session_id": "x", "speaker": "spk1", "start_time": 0.0, "end_time": 1.0, "words": "hello"},
        {"session_id": "x", "speaker": "spk1", "start_time": 2.0, "end_time": 3.0, "words": "world"},
    ]
    out = aggregate_rows_by_session_speaker(rows)
    assert out == [{"session_id": "x", "speaker": "spk1", "start_time": 0, "end_time": 0, "words": "hello world"}]


def test_config_loader():
    cfg = load_config(Path(__file__).parents[1] / "config.example.yaml")
    assert "ami" in cfg.datasets
    assert cfg.collar == 5


def test_missing_textgrid_is_reported(tmp_path, monkeypatch):
    root = tmp_path / "testset_tse" / "ami"
    (root / "audio").mkdir(parents=True)
    (root / "textgrid").mkdir()
    (root / "rttm").mkdir()
    (root / "audio" / "sdm_ES2004a-2.wav").write_bytes(b"")
    (tmp_path / "preds.jsonl").write_text(
        json.dumps({"session_id": "ES2004a", "speaker": "s0", "start_time": 0, "end_time": 1, "words": "hello"}) + "\n",
        encoding="utf-8",
    )
    def fake_prepare_metrics(raw_dir, collar):
        for name in ("wer", "cpwer", "tcpwer", "tcorcwer"):
            (raw_dir / f"{name}_average.json").write_text('{"error_rate": 0.0}', encoding="utf-8")

    def fake_prepare_normalized_metrics(raw_dir, norm_dir, collar):
        norm_dir.mkdir(parents=True, exist_ok=True)
        for name in ("wer", "cpwer", "tcpwer", "tcorcwer"):
            (norm_dir / f"{name}_average.norm.json").write_text('{"error_rate": 0.0}', encoding="utf-8")

    monkeypatch.setattr("scoring_dicow.metrics.prepare_metrics", fake_prepare_metrics)
    monkeypatch.setattr("scoring_dicow.metrics.prepare_normalized_metrics", fake_prepare_normalized_metrics)

    score_dataset(
        root,
        DatasetConfig(name="ami", predictions=str(tmp_path / "preds.jsonl"), mapping="ami"),
        tmp_path / "out" / "ami",
        collar=5,
    )

    missing = json.loads((tmp_path / "out" / "ami" / "missing_sessions.json").read_text(encoding="utf-8"))
    assert missing == [{"file": "sdm_ES2004a-2", "session": "ES2004a", "reason": "textgrid_missing"}]


def test_reassign_hypothesis_speakers_matches_by_transcript():
    reference_rows = [
        {"session_id": "s1", "speaker": "alice", "start_time": 0.0, "end_time": 1.0, "words": "hello there"},
        {"session_id": "s1", "speaker": "bob", "start_time": 1.0, "end_time": 2.0, "words": "general kenobi"},
    ]
    hypothesis_rows = [
        {"session_id": "s1", "speaker": "hyp0", "start_time": 0.0, "end_time": 1.0, "words": "general kenobi"},
        {"session_id": "s1", "speaker": "hyp1", "start_time": 1.0, "end_time": 2.0, "words": "hello there"},
    ]

    reassigned, debug = reassign_hypothesis_speakers(reference_rows, hypothesis_rows)

    assert [row["speaker"] for row in reassigned] == ["bob", "alice"]
    assert debug["s1"]["mapping_hyp_to_ref"] == {"hyp0": "bob", "hyp1": "alice"}


def test_score_dataset_writes_reassigned_hypothesis(tmp_path, monkeypatch):
    root = tmp_path / "testset_tse" / "l2m"
    (root / "audio").mkdir(parents=True)
    (root / "textgrid").mkdir()
    (root / "rttm").mkdir()
    (root / "audio" / "mix.wav").write_bytes(b"")
    (root / "rttm" / "mix.rttm").write_text("", encoding="utf-8")
    (root / "textgrid" / "mix.TextGrid").write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "Speaker A"
        xmin = 0
        xmax = 2
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "hello there"
    item [2]:
        class = "IntervalTier"
        name = "Speaker B"
        xmin = 0
        xmax = 2
        intervals: size = 1
        intervals [1]:
            xmin = 1
            xmax = 2
            text = "general kenobi"
""",
        encoding="utf-8",
    )
    (tmp_path / "preds.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"session_id": "mix", "speaker": "hyp0", "start_time": 0, "end_time": 1, "words": "general kenobi"}),
                json.dumps({"session_id": "mix", "speaker": "hyp1", "start_time": 1, "end_time": 2, "words": "hello there"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_prepare_metrics(raw_dir, collar):
        for name in ("wer", "cpwer", "tcpwer", "tcorcwer"):
            (raw_dir / f"{name}_average.json").write_text('{"error_rate": 0.0}', encoding="utf-8")

    def fake_prepare_normalized_metrics(raw_dir, norm_dir, collar):
        norm_dir.mkdir(parents=True, exist_ok=True)
        for name in ("wer", "cpwer", "tcpwer", "tcorcwer"):
            (norm_dir / f"{name}_average.norm.json").write_text('{"error_rate": 0.0}', encoding="utf-8")

    def fake_compute_der(ref_multi, hyp_multi):
        return {
            "sessions_scored": 1,
            "total": 2.0,
            "correct": 2.0,
            "false_alarm": 0.0,
            "missed_speech": 0.0,
            "speaker_confusion": 0.0,
            "rates": {"DER": 0.0, "FA": 0.0, "MS": 0.0, "SC": 0.0},
        }

    monkeypatch.setattr("scoring_dicow.metrics.prepare_metrics", fake_prepare_metrics)
    monkeypatch.setattr("scoring_dicow.metrics.prepare_normalized_metrics", fake_prepare_normalized_metrics)
    monkeypatch.setattr("scoring_dicow.metrics.compute_der", fake_compute_der)

    score_dataset(
        root,
        DatasetConfig(name="l2m", predictions=str(tmp_path / "preds.jsonl"), mapping="l2m"),
        tmp_path / "out" / "l2m",
        collar=5,
    )

    hyp_multi = [
        json.loads(line)
        for line in (tmp_path / "out" / "l2m" / "hypothesis_multi.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    debug = json.loads((tmp_path / "out" / "l2m" / "speaker_assignment_debug.json").read_text(encoding="utf-8"))

    assert [row["speaker"] for row in hyp_multi] == ["speaker b", "speaker a"]
    assert debug["mix"]["mapping_hyp_to_ref"] == {"hyp0": "speaker b", "hyp1": "speaker a"}
