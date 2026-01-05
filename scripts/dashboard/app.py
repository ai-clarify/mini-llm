from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from flask import Flask, jsonify, render_template, request, send_from_directory
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "data"
DATASET_ROOT = REPO_ROOT / "dataset"
OUT_ROOT = REPO_ROOT / "out"
CONFIG_ROOT = REPO_ROOT / "configs" / "dashboard"


DATA_EXTS = {".jsonl", ".json", ".csv", ".txt"}
METRIC_TAGS = ("train/loss", "train/lr", "eval/loss", "train/accuracy", "eval/accuracy")


@dataclass
class DatasetRow:
    path: Path
    size_bytes: int
    line_count: int | None
    line_count_capped: bool
    modified_at: datetime
    preview: list[str]

    def to_payload(self) -> dict:
        return {
            "path": str(self.path.relative_to(REPO_ROOT)),
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "line_count_capped": self.line_count_capped,
            "modified_at": self.modified_at.isoformat(),
            "preview": self.preview,
        }


@dataclass
class RunSummary:
    run_id: str
    name: str
    stage: str
    kind: str
    latest_checkpoint: str | None
    modified_at: datetime
    metrics: dict[str, float]
    tensorboard_root: str | None
    event_file: Path | None = None

    def to_payload(self) -> dict:
        return {
            "id": self.run_id,
            "name": self.name,
            "stage": self.stage,
            "kind": self.kind,
            "latest_checkpoint": self.latest_checkpoint,
            "modified_at": self.modified_at.isoformat(),
            "metrics": self.metrics,
            "tensorboard_root": self.tensorboard_root,
        }


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _read_preview(path: Path, limit: int = 3) -> list[str]:
    preview: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            content = line.strip()
            if content:
                preview.append(content)
    return preview


def _count_lines(path: Path, cap: int = 50_000) -> tuple[int | None, bool]:
    count = 0
    capped = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for count, _ in enumerate(handle, start=1):
            if count >= cap:
                capped = True
                break
    return (count if count else None), capped


def _iter_data_files() -> Iterable[Path]:
    roots = [DATA_ROOT, DATASET_ROOT, OUT_ROOT / "datasets"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DATA_EXTS and not path.name.startswith("."):
                yield path


def _gather_datasets() -> list[DatasetRow]:
    datasets: list[DatasetRow] = []
    for path in sorted(_iter_data_files()):
        stat = path.stat()
        line_count, capped = _count_lines(path)
        datasets.append(
            DatasetRow(
                path=path,
                size_bytes=stat.st_size,
                line_count=line_count,
                line_count_capped=capped,
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                preview=_read_preview(path),
            )
        )
    return datasets


def _safe_snapshot_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("._-")
    return cleaned or "snapshot"


def _resolve_paths(paths: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if REPO_ROOT not in candidate.parents and candidate != REPO_ROOT:
            raise ValueError(f"Path outside repository: {raw}")
        if not candidate.exists():
            raise FileNotFoundError(raw)
        resolved.append(candidate)
    return resolved


def _materialize_dataset(name: str, inputs: Sequence[Path]) -> dict:
    snapshot_dir = OUT_ROOT / "datasets" / name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    combined = snapshot_dir / "combined.jsonl"

    total_lines = 0
    total_bytes = 0
    with combined.open("w", encoding="utf-8") as writer:
        for path in inputs:
            with path.open("r", encoding="utf-8", errors="ignore") as reader:
                for line in reader:
                    writer.write(line)
                    total_lines += 1
            total_bytes += path.stat().st_size

    manifest = {
        "name": name,
        "created_at": _now(),
        "source_files": [str(p.relative_to(REPO_ROOT)) for p in inputs],
        "combined_path": str(combined.relative_to(REPO_ROOT)),
        "line_count": total_lines,
        "size_bytes": total_bytes,
    }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _infer_stage(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for token, stage in {
        "pretrain": "pretrain",
        "sft": "sft",
        "dpo": "dpo",
        "ppo": "ppo",
        "grpo": "grpo",
        "distill": "distill",
    }.items():
        if any(token in part for part in parts):
            return stage
    return "unknown"


def _iter_run_roots() -> Iterable[Path]:
    if not OUT_ROOT.exists():
        return []
    candidates: set[Path] = set()
    for child in OUT_ROOT.iterdir():
        candidates.add(child)
    for event_file in OUT_ROOT.rglob("events.out.tfevents.*"):
        candidates.add(event_file.parent)
    for ckpt_dir in OUT_ROOT.rglob("checkpoints"):
        if ckpt_dir.parent.exists():
            candidates.add(ckpt_dir.parent)
        parent = ckpt_dir.parent.parent
        if parent != OUT_ROOT and parent.exists():
            candidates.add(parent)
    return sorted(candidates)


def _latest_checkpoint(path: Path) -> Path | None:
    checkpoints: list[Path] = []
    if path.is_file() and path.suffix in {".pth", ".safetensors"}:
        checkpoints.append(path)
    if path.is_dir():
        for suffix in ("*.pth", "*.safetensors"):
            checkpoints.extend(path.rglob(suffix))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def _find_event_file(path: Path) -> Path | None:
    if path.is_file() and path.name.startswith("events.out.tfevents"):
        return path
    if not path.exists():
        return None
    candidates = sorted(path.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_scalars(event_path: Path) -> dict[str, list[dict]]:
    try:
        acc = EventAccumulator(str(event_path))
        acc.Reload()
    except Exception:
        return {}

    tag_map = acc.Tags().get("scalars", [])
    series: dict[str, list[dict]] = {}
    for tag in METRIC_TAGS:
        if tag not in tag_map:
            continue
        events = acc.Scalars(tag)
        series[tag] = [{"step": e.step, "value": e.value, "wall_time": e.wall_time} for e in events]
    return series


def _latest_mlx_checkpoint_dir(root: Path) -> Path | None:
    ckpt_root = root / "checkpoints"
    if not ckpt_root.exists():
        return None
    candidates = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_mlx_metadata(root: Path) -> dict | None:
    latest_dir = _latest_mlx_checkpoint_dir(root)
    if latest_dir is None:
        return None
    state = _read_json(latest_dir / "state.json") or {}
    config = _read_json(latest_dir / "config.json") or {}
    args = state.get("args", {}) if isinstance(state, dict) else {}
    model = config if isinstance(config, dict) else {}
    return {
        "step": state.get("step"),
        "seen_tokens": state.get("seen_tokens"),
        "checkpoint_dir": str(latest_dir.relative_to(REPO_ROOT)),
        "task": args.get("task"),
        "preset": args.get("preset"),
        "dtype": args.get("dtype"),
        "seq_len": args.get("seq_len"),
        "batch_size": args.get("batch_size"),
        "accum_steps": args.get("accum_steps"),
        "learning_rate": args.get("learning_rate"),
        "epochs": args.get("epochs"),
        "max_steps": args.get("max_steps"),
        "model": {
            "hidden_size": model.get("hidden_size"),
            "num_hidden_layers": model.get("num_hidden_layers"),
            "num_attention_heads": model.get("num_attention_heads"),
            "num_key_value_heads": model.get("num_key_value_heads"),
            "vocab_size": model.get("vocab_size"),
        },
    }


def _build_run_summary(root: Path) -> RunSummary | None:
    if not root.exists():
        return None
    latest = _latest_checkpoint(root)
    event_file = _find_event_file(root)
    mlx = _load_mlx_metadata(root)
    if not latest and not event_file and not mlx:
        return None
    metrics: dict[str, float] = {}
    if event_file:
        scalars = _load_scalars(event_file)
        for key, values in scalars.items():
            if values:
                metrics[key] = values[-1]["value"]
    if mlx:
        if mlx.get("step") is not None:
            metrics["step"] = float(mlx["step"])
        if mlx.get("seen_tokens") is not None:
            metrics["seen_tokens"] = float(mlx["seen_tokens"])
    try:
        modified = datetime.fromtimestamp(root.stat().st_mtime)
    except FileNotFoundError:
        return None
    stage = mlx.get("task") if mlx and mlx.get("task") else _infer_stage(root)
    return RunSummary(
        run_id=str(root.relative_to(REPO_ROOT)),
        name=root.name,
        stage=stage,
        kind="mlx" if mlx else "torch",
        latest_checkpoint=str(latest.relative_to(REPO_ROOT)) if latest else None,
        modified_at=modified,
        metrics=metrics,
        tensorboard_root=str(event_file.parent.relative_to(REPO_ROOT)) if event_file else None,
        event_file=event_file,
    )


def _summaries() -> list[RunSummary]:
    runs: list[RunSummary] = []
    for root in _iter_run_roots():
        summary = _build_run_summary(root)
        if summary:
            runs.append(summary)
    runs.sort(key=lambda r: r.modified_at, reverse=True)
    return runs


def _load_configs() -> list[dict]:
    configs: list[dict] = []
    if not CONFIG_ROOT.exists():
        return configs
    for path in sorted(CONFIG_ROOT.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        configs.append(
            {
                "name": meta.get("name") or path.stem,
                "version": meta.get("version"),
                "stage": meta.get("stage"),
                "description": meta.get("description"),
                "path": str(path.relative_to(REPO_ROOT)),
                "content": data,
            }
        )
    return configs


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/static/<path:filename>")
    def static_assets(filename: str):  # type: ignore[override]
        return send_from_directory(app.static_folder, filename)

    @app.route("/api/overview")
    def overview():
        datasets = _gather_datasets()
        runs = _summaries()
        configs = _load_configs()
        return jsonify(
            {
                "datasets": len(datasets),
                "runs": len(runs),
                "configs": len(configs),
                "latest_run": runs[0].to_payload() if runs else None,
            }
        )

    @app.route("/api/datasets")
    def datasets():
        return jsonify([row.to_payload() for row in _gather_datasets()])

    @app.route("/api/datasets/materialize", methods=["POST"])
    def datasets_materialize():
        payload = request.get_json(force=True) or {}
        files = payload.get("files") or []
        name_raw = payload.get("name") or f"snapshot-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        if not files:
            return jsonify({"error": "No input files supplied"}), 400
        resolved = _resolve_paths(files)
        manifest = _materialize_dataset(_safe_snapshot_name(name_raw), resolved)
        return jsonify(manifest)

    @app.route("/api/configs")
    def configs():
        return jsonify(_load_configs())

    @app.route("/api/runs")
    def runs():
        return jsonify([run.to_payload() for run in _summaries()])

    @app.route("/api/runs/<path:run_id>")
    def run_detail(run_id: str):
        run_path = (REPO_ROOT / run_id).resolve()
        if REPO_ROOT not in run_path.parents and run_path != REPO_ROOT:
            return jsonify({"error": "Run path outside repository"}), 400
        summary = _build_run_summary(run_path)
        if not summary:
            return jsonify({"error": "Run not found"}), 404
        scalars: dict[str, list[dict]] = {}
        if summary.event_file:
            scalars = _load_scalars(summary.event_file)
        mlx = _load_mlx_metadata(run_path)
        return jsonify({"run": summary.to_payload(), "mlx": mlx, "scalars": scalars})

    @app.route("/api/runs/<path:run_id>/scalars")
    def run_scalars(run_id: str):
        run_path = (REPO_ROOT / run_id).resolve()
        if REPO_ROOT not in run_path.parents and run_path != REPO_ROOT:
            return jsonify({"error": "Run path outside repository"}), 400
        event_file = _find_event_file(run_path)
        if not event_file:
            return jsonify({"scalars": {}})
        return jsonify({"scalars": _load_scalars(event_file)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM dashboard server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
