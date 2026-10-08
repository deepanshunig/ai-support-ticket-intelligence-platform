"""CPU comparison: frozen DistilBERT embeddings with a fitted linear classifier.

This experiment never fine-tunes the encoder and never selects a deployed model.
Only fit/development records are opened. A full, pinned model commit is required.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ticket_intelligence.training.baseline import _load_data, _sha256_file, evaluate_predictions

MODEL_ID = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 32
TORCH_THREADS = 4
POOLING = "Attention-mask mean of nonpadding tokens, including special tokens."


def _load_encoder(revision: str, cache_dir: Path) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=revision, cache_dir=str(cache_dir), trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=str(cache_dir),
        use_safetensors=True,
        trust_remote_code=False,
    )
    model.to("cpu").requires_grad_(False)
    model.eval()
    return model, tokenizer


def embed_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Return float32 CPU embeddings using dynamic padding and masked mean pooling."""
    if batch_size < 1 or max_length < 1:
        raise ValueError("Batch size and maximum length must be positive.")
    torch.set_num_threads(TORCH_THREADS)
    model.to("cpu")
    model.eval()
    if not texts:
        return np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
    batches = []
    with torch.inference_mode():
        for batch, start in enumerate(range(0, len(texts), batch_size), start=1):
            selected = texts[start : start + batch_size]
            inputs = tokenizer(
                selected,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {name: value.to("cpu") for name, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            completed = min(start + batch_size, len(texts))
            if len(texts) > batch_size and (batch % 25 == 0 or completed == len(texts)):
                print(f"Encoded {completed}/{len(texts)} records on CPU", flush=True)
    return np.concatenate(batches, axis=0)


def _end_to_end_latency(model: Any, tokenizer: Any, head: Pipeline, texts: list[str]) -> dict:
    def predict_one(text: str) -> None:
        embedding = embed_texts(model, tokenizer, [text])
        with threadpool_limits(limits=1):
            head.predict(embedding)

    for index in range(3):
        predict_one(texts[index % len(texts)])
    elapsed = []
    for index in range(20):
        started = time.perf_counter()
        predict_one(texts[index % len(texts)])
        elapsed.append((time.perf_counter() - started) * 1000)
    return {
        "batch_size": 1,
        "repetitions": 20,
        "warmup_repetitions": 3,
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "scope": "Tokenizer + CPU encoder + pooling + linear head; excludes HTTP/model loading.",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_transformer(
    data_dir: Path = Path("data/processed/banking77"),
    artifact_dir: Path = Path("artifacts/transformer"),
    report_dir: Path = Path("reports/transformer"),
    revision: str | None = None,
    *,
    encoder_loader: Callable[[str, Path], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Extract frozen features; fit scaling/heads on fit and select on development."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pass a pinned, full 40-character lowercase model commit via --revision.")
    data_dir, artifact_dir, report_dir = (
        Path(data_dir).resolve(),
        Path(artifact_dir).resolve(),
        Path(report_dir).resolve(),
    )
    if artifact_dir.is_relative_to(data_dir) or report_dir.is_relative_to(data_dir):
        raise ValueError("Artifacts and reports must be outside the prepared data directory.")
    fit, development, mapping = _load_data(data_dir)
    labels = [entry["category"] for entry in mapping]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, tokenizer = (encoder_loader or _load_encoder)(revision, artifact_dir / "cache")
    model.to("cpu").requires_grad_(False)
    model.eval()
    loading_seconds = time.perf_counter() - started
    embedding_seconds = {}
    embeddings = {}
    for name, records in (("fit", fit), ("development", development)):
        print(f"Encoding {name}: {len(records)} records with frozen DistilBERT", flush=True)
        started = time.perf_counter()
        embeddings[name] = embed_texts(model, tokenizer, records["text"].tolist())
        embedding_seconds[name] = time.perf_counter() - started
        np.save(artifact_dir / f"{name}_embeddings.npy", embeddings[name], allow_pickle=False)
    candidates = []
    best = None
    with threadpool_limits(limits=1):
        for regularization in (1.0, 4.0, 10.0):
            head = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(
                        C=regularization, max_iter=1000, random_state=42, solver="lbfgs"
                    )),
                ]
            )
            started = time.perf_counter()
            head.fit(embeddings["fit"], fit["category"])
            training_seconds = time.perf_counter() - started
            predicted = head.predict(embeddings["development"])
            metrics = evaluate_predictions(development["category"], predicted, labels)
            candidates.append({
                "C": regularization,
                "development_accuracy": metrics["accuracy"],
                "development_macro_f1": metrics["macro_f1"],
                "training_seconds": training_seconds,
            })
            print(
                f"Linear head C={regularization:g}: development macro-F1={metrics['macro_f1']:.4f}",
                flush=True,
            )
            if best is None or (metrics["macro_f1"], metrics["accuracy"]) > (
                best[2]["macro_f1"], best[2]["accuracy"]
            ):
                best = (head, regularization, metrics, training_seconds)
    if best is None:
        raise ValueError("No linear head was trained.")
    head, regularization, metrics, training_seconds = best
    probabilities = head.predict_proba(embeddings["development"])
    classes = head.classes_.tolist()
    top = np.argsort(-probabilities, axis=1, kind="stable")[:, : min(3, len(classes))]
    metrics["top_3_accuracy"] = float(
        (np.asarray(classes)[top] == development["category"].to_numpy()[:, None]).any(axis=1).mean()
    )
    metrics["top_k_used"] = min(3, len(classes))
    predictions = head.predict(embeddings["development"])
    errors = development.loc[:, ["text", "category"]].rename(columns={"category": "true_category"})
    errors["predicted_category"] = predictions
    errors["uncalibrated_confidence"] = probabilities.max(axis=1)
    errors.loc[errors["true_category"] != predictions].to_csv(
        report_dir / "errors.csv", index=False, lineterminator="\n"
    )
    encoder_dir = artifact_dir / "encoder"
    encoder_dir.mkdir(exist_ok=True)
    model.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(encoder_dir)
    head_path = artifact_dir / "head.joblib"
    joblib.dump(head, head_path, compress=3)
    asset_paths = [head_path, *(path for path in encoder_dir.rglob("*") if path.is_file())]
    manifest = {
        "schema_version": 1,
        "sha256": {
            path.relative_to(artifact_dir).as_posix(): _sha256_file(path)
            for path in sorted(asset_paths)
        },
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    metadata = {
        "schema_version": 1,
        "experiment": "frozen_distilbert_plus_linear_head",
        "encoder_fine_tuned": False,
        "encoder_frozen": True,
        "model_id": MODEL_ID,
        "revision": revision,
        "device": "cpu",
        "torch_threads": TORCH_THREADS,
        "pooling": POOLING,
        "max_length": MAX_LENGTH,
        "dynamic_padding": True,
        "encoding_batch_size": BATCH_SIZE,
        "classes": classes,
        "label_mapping": mapping,
        "data_counts": {"fit": len(fit), "development": len(development)},
        "data_checksums": {
            name: _sha256_file(data_dir / name)
            for name in ("fit.csv", "development.csv", "label_mapping.json")
        },
        "embedding_files": {
            name: {
                "path": f"{name}_embeddings.npy",
                "sha256": _sha256_file(artifact_dir / f"{name}_embeddings.npy"),
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
            }
            for name, matrix in embeddings.items()
        },
        "head": {"path": "head.joblib", "sha256": manifest["sha256"]["head.joblib"],
                 "C": regularization, "classes": classes},
        "versions": {"torch": str(torch.__version__), "transformers": version("transformers"),
                     "scikit-learn": version("scikit-learn")},
        "model_asset_manifest": "manifest.json",
        "model_total_mb": sum(path.stat().st_size for path in asset_paths) / 1_000_000,
        "loading_seconds": loading_seconds,
        "embedding_seconds": embedding_seconds,
        "training_seconds": training_seconds,
        "head_search_training_seconds": sum(item["training_seconds"] for item in candidates),
        "development_metrics": metrics,
        "latency": _end_to_end_latency(model, tokenizer, head, development["text"].tolist()),
        "selection_partition": "development",
    }
    _write_json(artifact_dir / "metadata.json", metadata)
    report = {
        **metadata,
        "candidates": candidates,
        "notes": [
            "Frozen pretrained encoder plus trained linear head; no encoder weight updates.",
            "StandardScaler and all linear heads fit only fit.csv; no development refit.",
            "Encoder is in eval/inference mode; padding excluded from pooling.",
            "Sequences longer than 64 tokens are truncated, including special-token overhead.",
            "Calibration, policy, challenge and official test records were not opened.",
            "Development results select head C and are not final test estimates.",
            "Model MB excludes cache/embeddings; latency measures warmed CPU inference.",
            "Deployment choice requires comparison with baseline quality and runtime cost.",
        ],
    }
    _write_json(report_dir / "report.json", report)
    pd.DataFrame(candidates).to_csv(report_dir / "candidates.csv", index=False, lineterminator="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/banking77"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/transformer"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/transformer"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    try:
        report = train_transformer(args.data_dir, args.artifact_dir, args.report_dir, args.revision)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Transformer comparison failed: {error}\n")
    print(json.dumps({
        "experiment": report["experiment"],
        "development_accuracy": report["development_metrics"]["accuracy"],
        "development_macro_f1": report["development_metrics"]["macro_f1"],
        "development_top_3_accuracy": report["development_metrics"]["top_3_accuracy"],
        "model_total_mb": report["model_total_mb"],
        "latency": report["latency"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
