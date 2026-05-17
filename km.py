#!/usr/bin/env python3
"""
K-means clustering for GPFF II structured logs (log.jsonl).

Usage:
    python km.py train [--log log.jsonl] [--model artifacts/kmeans_model.joblib]
    python km.py predict [--log log.jsonl] [--model ...] [--out artifacts/log_clusters.jsonl]
    python km.py viz [--pred artifacts/log_clusters.jsonl] [--out artifacts/cluster_heatmap.png]
    python km.py run   # train + predict + viz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config.kmeans_config import (
    HEATMAP_PATH,
    KNOWN_EVENTS,
    LEVEL_ORDINAL,
    LOG_PATH,
    MODEL_PATH,
    NUMERIC_DATA_KEYS,
    PREDICTIONS_PATH,
    resolve_hyperparameters,
)


def _strip_bom(raw: bytes) -> bytes:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw[2:]
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:]
    return raw


def _decode_line(line: bytes) -> str | None:
    line = line.strip(b"\r")
    if not line:
        return None
    for encoding in ("utf-8", "utf-16-le", "latin-1"):
        try:
            return line.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def load_logs(path: str | Path) -> list[dict]:
    """Load JSONL log file (utf-8 lines; tolerates UTF-16 BOM prefix on Windows)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")

    raw = _strip_bom(path.read_bytes())
    if not raw.strip():
        raise ValueError(f"Log file is empty: {path}")

    entries = []
    skipped = 0
    for line_no, line in enumerate(raw.split(b"\n"), start=1):
        text = _decode_line(line)
        if text is None:
            skipped += 1
            continue
        try:
            entries.append(json.loads(text))
        except json.JSONDecodeError:
            skipped += 1

    if skipped:
        print(f"Warning: skipped {skipped} non-JSON line(s) in {path}", file=sys.stderr)

    if not entries:
        raise ValueError(f"No log entries parsed from {path}")
    return entries


def _event_bucket(event: str) -> str:
    return event if event in KNOWN_EVENTS[:-1] else "other"


def feature_names() -> list[str]:
    names = ["level_ord", "t", "delta_t", "has_error"]
    names.extend(f"event_{e}" for e in KNOWN_EVENTS)
    names.extend(f"data_{k}" for k in NUMERIC_DATA_KEYS)
    return names


def featurize(entries: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Build feature matrix and metadata rows aligned with entries."""
    names = feature_names()
    rows = []
    meta = []
    prev_t = 0.0

    for entry in entries:
        level = entry.get("level", "INFO")
        level_ord = LEVEL_ORDINAL.get(level, 1)
        t = float(entry.get("t", 0.0))
        delta_t = t - prev_t if meta else 0.0
        prev_t = t

        event = _event_bucket(str(entry.get("event", "other")))
        has_error = 1.0 if entry.get("error") else 0.0

        row = [level_ord, t, delta_t, has_error]
        row.extend(1.0 if event == e else 0.0 for e in KNOWN_EVENTS)

        data = entry.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        for key in NUMERIC_DATA_KEYS:
            val = data.get(key, 0.0)
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                row.append(0.0)

        rows.append(row)
        meta.append({
            "seq": entry.get("seq"),
            "event": entry.get("event", "other"),
            "level": level,
            "t": t,
        })

    return np.array(rows, dtype=np.float64), meta


def train(log_path: str, model_path: str) -> dict:
    entries = load_logs(log_path)
    n = len(entries)
    if n < 2:
        raise ValueError(
            f"Need at least 2 log entries to train (found {n}). "
            "Run the simulator first: python main.py"
        )

    X, _ = featurize(entries)
    hp = resolve_hyperparameters(n)

    zero_rows = int(np.all(X == 0, axis=1).sum())
    if zero_rows:
        print(f"Warning: {zero_rows} all-zero feature row(s) in training data", file=sys.stderr)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) if hp["use_scaler"] else X

    kmeans = KMeans(
        n_clusters=hp["n_clusters"],
        n_init=hp["n_init"],
        max_iter=hp["max_iter"],
        algorithm=hp["algorithm"],
        random_state=hp["random_state"],
    )
    kmeans.fit(X_scaled)

    bundle = {
        "model": kmeans,
        "scaler": scaler,
        "feature_names": feature_names(),
        "tier": hp["tier"],
        "hyperparameters": hp,
        "n_samples": n,
    }

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)

    print(f"Trained K-means: tier={hp['tier']}, k={hp['n_clusters']}, n={n}")
    print(f"Model saved to {out}")
    return bundle


def _load_bundle(model_path: str) -> dict:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Run: python km.py train"
        )
    return joblib.load(path)


def predict(log_path: str, model_path: str, out_path: str) -> list[dict]:
    bundle = _load_bundle(model_path)
    entries = load_logs(log_path)
    X, meta = featurize(entries)

    scaler = bundle["scaler"]
    X_scaled = scaler.transform(X)

    labels = bundle["model"].predict(X_scaled)
    tier_used = bundle.get("tier", "unknown")

    results = []
    for entry, label, m in zip(entries, labels, meta):
        row = dict(entry)
        row["cluster_id"] = int(label)
        row["tier_used"] = tier_used
        results.append(row)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} predictions to {out}")
    return results


def _load_predictions(pred_path: str) -> list[dict]:
    path = Path(pred_path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}. Run: python km.py predict")

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def viz(pred_path: str, out_path: str, normalize_rows: bool = True) -> Path:
    """Cluster x event count heatmap (row-normalized by default)."""
    rows = _load_predictions(pred_path)

    cluster_ids = sorted({int(r["cluster_id"]) for r in rows})
    events = sorted({str(r.get("event", "other")) for r in rows})

    counts: dict[int, dict[str, int]] = defaultdict(lambda: Counter())
    for r in rows:
        cid = int(r["cluster_id"])
        ev = str(r.get("event", "other"))
        counts[cid][ev] += 1

    matrix = np.zeros((len(cluster_ids), len(events)), dtype=np.float64)
    for i, cid in enumerate(cluster_ids):
        for j, ev in enumerate(events):
            matrix[i, j] = counts[cid][ev]

    if normalize_rows and matrix.size > 0:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        plot_matrix = matrix / row_sums
        cbar_label = "Fraction of cluster"
    else:
        plot_matrix = matrix
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(max(8, len(events) * 0.9), max(4, len(cluster_ids) * 0.6)))
    im = ax.imshow(plot_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(events, rotation=45, ha="right")
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"cluster {c}" for c in cluster_ids])
    ax.set_xlabel("Event")
    ax.set_ylabel("Cluster")
    ax.set_title("Log clusters by event type")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Heatmap saved to {out}")
    return out


def cmd_train(args: argparse.Namespace) -> None:
    train(args.log, args.model)


def cmd_predict(args: argparse.Namespace) -> None:
    predict(args.log, args.model, args.out)


def cmd_viz(args: argparse.Namespace) -> None:
    viz(args.pred, args.out)


def cmd_run(args: argparse.Namespace) -> None:
    train(args.log, args.model)
    predict(args.log, args.model, args.out)
    viz(args.out, args.heatmap)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K-means clustering for GPFF II logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train K-means on log.jsonl")
    p_train.add_argument("--log", default=LOG_PATH)
    p_train.add_argument("--model", default=MODEL_PATH)
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Assign cluster labels to log lines")
    p_predict.add_argument("--log", default=LOG_PATH)
    p_predict.add_argument("--model", default=MODEL_PATH)
    p_predict.add_argument("--out", default=PREDICTIONS_PATH)
    p_predict.set_defaults(func=cmd_predict)

    p_viz = sub.add_parser("viz", help="Render cluster x event heatmap")
    p_viz.add_argument("--pred", default=PREDICTIONS_PATH)
    p_viz.add_argument("--out", default=HEATMAP_PATH)
    p_viz.set_defaults(func=cmd_viz)

    p_run = sub.add_parser("run", help="Train, predict, and visualize (default workflow)")
    p_run.add_argument("--log", default=LOG_PATH)
    p_run.add_argument("--model", default=MODEL_PATH)
    p_run.add_argument("--out", default=PREDICTIONS_PATH)
    p_run.add_argument("--heatmap", default=HEATMAP_PATH)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
