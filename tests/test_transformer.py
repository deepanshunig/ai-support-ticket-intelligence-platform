"""Offline tensor fixtures for frozen feature extraction and linear-head boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="Frozen-transformer tests require optional CPU Torch.")

from ticket_intelligence.training.transformer import embed_texts, train_transformer  # noqa: E402


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        self.calls.append(dict(kwargs))
        limit = int(kwargs["max_length"])
        rows = [[int(token) if token.isdigit() else 10 for token in text.split()][:limit]
                for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])
        return {"input_ids": ids, "attention_mask": mask}

    def save_pretrained(self, destination: Path) -> None:
        (destination / "tokenizer.json").write_text('{"fixture": true}', encoding="utf-8")


class FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=2)
        self.forward_states: list[tuple[bool, bool]] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        values = input_ids.float() * self.weight
        hidden = torch.stack((values, values.square()), dim=-1)
        hidden = torch.where(attention_mask.unsqueeze(-1).bool(), hidden, 10000.0)
        return SimpleNamespace(last_hidden_state=hidden)

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization
        (destination / "config.json").write_text('{"fixture": true}', encoding="utf-8")
        (destination / "model.safetensors").write_bytes(b"Offline mock checkpoint asset")


def test_masked_mean_excludes_padding_and_is_batch_invariant() -> None:
    model, tokenizer = FakeEncoder(), FakeTokenizer()
    batched = embed_texts(model, tokenizer, ["1 3", "5"], batch_size=2)
    individual = embed_texts(model, tokenizer, ["1 3", "5"], batch_size=1)
    np.testing.assert_allclose(batched, [[2, 5], [5, 25]])
    np.testing.assert_allclose(individual, batched)
    assert batched.dtype == np.float32
    assert all(not training and not grad_enabled for training, grad_enabled in model.forward_states)
    assert all(call["padding"] is True and call["truncation"] is True for call in tokenizer.calls)


def test_embedding_truncation_and_empty_shape() -> None:
    model, tokenizer = FakeEncoder(), FakeTokenizer()
    np.testing.assert_allclose(embed_texts(model, tokenizer, ["1 3 9"], max_length=2), [[2, 5]])
    assert embed_texts(model, tokenizer, []).shape == (0, 2)
    with pytest.raises(ValueError, match="positive"):
        embed_texts(model, tokenizer, ["1"], batch_size=0)


def test_head_training_uses_only_fit_development_and_preserves_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    mapping = [{"label_id": 0, "category": "z_card"}, {"label_id": 1, "category": "a_payment"}]
    (data_dir / "label_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    fit_rows = []
    for label, tokens in (("z_card", "1 1 1 1"), ("a_payment", "8 8 8 8")):
        fit_rows.extend({"text": f"{tokens} train{index}", "category": label} for index in range(6))
    development = pd.DataFrame([
        {"text": "1 1 2 2 unseen_dev", "category": "z_card"},
        {"text": "8 8 9 9 unseen_dev", "category": "a_payment"},
    ])
    pd.DataFrame(fit_rows).to_csv(data_dir / "fit.csv", index=False)
    development.to_csv(data_dir / "development.csv", index=False)
    for filename in ("official_test.csv", "policy.csv", "calibration.csv", "challenge.csv"):
        (data_dir / filename).write_text("DO NOT OPEN", encoding="utf-8")
    original_files = {path.name: path.read_bytes() for path in data_dir.iterdir()}
    opened_csvs = []
    original_read_csv = pd.read_csv

    def record_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        assert Path(path).name in {"fit.csv", "development.csv"}
        opened_csvs.append(Path(path).name)
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", record_read_csv)
    model, tokenizer = FakeEncoder(), FakeTokenizer()
    loader_calls = []

    def load_encoder(revision: str, cache_dir: Path) -> tuple[FakeEncoder, FakeTokenizer]:
        loader_calls.append((revision, cache_dir))
        return model, tokenizer

    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    revision = "a" * 40
    report = train_transformer(data_dir, artifacts, reports, revision, encoder_loader=load_encoder)
    assert loader_calls == [(revision, artifacts.resolve() / "cache")]
    assert opened_csvs == ["fit.csv", "development.csv"]
    assert report["encoder_frozen"] is True and report["encoder_fine_tuned"] is False
    assert not model.weight.requires_grad
    assert report["classes"] == ["a_payment", "z_card"]
    assert report["label_mapping"] == mapping
    assert len(report["candidates"]) == 3
    best = max((item["development_macro_f1"], item["development_accuracy"])
               for item in report["candidates"])
    assert (report["development_metrics"]["macro_f1"],
            report["development_metrics"]["accuracy"]) == best
    head = joblib.load(artifacts / "head.joblib")
    fit_embeddings = np.load(artifacts / "fit_embeddings.npy", allow_pickle=False)
    development_embeddings = np.load(artifacts / "development_embeddings.npy", allow_pickle=False)
    np.testing.assert_allclose(
        head.named_steps["scaler"].mean_, fit_embeddings.mean(axis=0, dtype=np.float64)
    )
    predicted = head.predict(development_embeddings)
    assert float((predicted == development["category"].to_numpy()).mean()) == (
        report["development_metrics"]["accuracy"]
    )
    assert report["latency"]["repetitions"] == 20
    assert all(not training and not grad for training, grad in model.forward_states)
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["sha256"].items():
        assert hashlib.sha256((artifacts / filename).read_bytes()).hexdigest() == expected
    assert {path.name: path.read_bytes() for path in data_dir.iterdir()} == original_files
    assert set(report["data_checksums"]) == {"fit.csv", "development.csv", "label_mapping.json"}


@pytest.mark.parametrize("revision", [None, "main", "shortsha", "a" * 39])
def test_unpinned_revision_is_rejected_before_loading(tmp_path: Path, revision: str | None) -> None:
    with pytest.raises(ValueError, match="pinned"):
        train_transformer(tmp_path / "data", tmp_path / "assets", tmp_path / "reports", revision)
