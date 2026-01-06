from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
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
JOBS_ROOT = OUT_ROOT / "dashboard_jobs"
JOBS_DB_PATH = JOBS_ROOT / "jobs.json"


DATA_EXTS = {".jsonl", ".json", ".csv", ".txt"}
METRIC_TAGS = ("train/loss", "train/lr", "eval/loss", "train/accuracy", "eval/accuracy")

TRAINING_SCRIPTS = {
    "pretrain": "trainer/train_pretrain.py",
    "sft": "trainer/train_full_sft.py",
    "dpo": "trainer/train_dpo.py",
}


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, "TrainingJob"] = {}
_JOB_PROCS: dict[str, subprocess.Popen] = {}


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


def _ensure_jobs_root() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _persist_jobs_locked() -> None:
    _ensure_jobs_root()
    payload = {job_id: job.to_payload() for job_id, job in _JOBS.items()}
    tmp = JOBS_DB_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(JOBS_DB_PATH)


def _persist_jobs() -> None:
    with _JOBS_LOCK:
        _persist_jobs_locked()


def _load_jobs_from_disk() -> None:
    _ensure_jobs_root()
    if not JOBS_DB_PATH.exists():
        return
    try:
        raw = json.loads(JOBS_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    loaded: dict[str, TrainingJob] = {}
    for job_id, item in raw.items():
        if not isinstance(job_id, str) or not isinstance(item, dict):
            continue
        try:
            loaded[job_id] = TrainingJob(
                job_id=job_id,
                config_path=str(item.get("config_path") or ""),
                stage=str(item.get("stage") or ""),
                kind=str(item.get("kind") or ""),
                command=list(item.get("command") or []),
                run_id=item.get("run_id"),
                log_path=str(item.get("log_path") or ""),
                created_at=str(item.get("created_at") or ""),
                started_at=item.get("started_at"),
                finished_at=item.get("finished_at"),
                pid=item.get("pid"),
                return_code=item.get("return_code"),
                state=str(item.get("state") or "queued"),
                stop_requested=bool(item.get("stop_requested") or False),
            )
        except Exception:
            continue
    with _JOBS_LOCK:
        _JOBS.clear()
        _JOBS.update(loaded)


def _tail_text(path: Path, *, max_bytes: int = 96_000) -> str:
    if not path.exists():
        return ""
    max_bytes = max(1024, int(max_bytes))
    with path.open("rb") as handle:
        try:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        except OSError:
            handle.seek(0, os.SEEK_SET)
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    return text[-200_000:]


def _resolve_repo_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if REPO_ROOT not in candidate.parents and candidate != REPO_ROOT:
        raise ValueError(f"Path outside repository: {raw}")
    return candidate


def _choose_device(value: str | None) -> str | None:
    if value is None:
        return None
    if value != "auto":
        return value
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _build_cli_args(args: dict) -> list[str]:
    argv: list[str] = []
    for key, value in (args or {}).items():
        if value is None:
            continue
        flag = f"--{key}"

        if key == "device":
            resolved = _choose_device(str(value))
            if resolved:
                argv.extend([flag, resolved])
            continue

        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue

        if key == "use_moe":
            if bool(value):
                argv.extend([flag, "True"])
            continue

        argv.extend([flag, str(value)])
    return argv


def _build_env_overrides(env_payload: dict | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not env_payload:
        return overrides
    for key, value in env_payload.items():
        if not key or not isinstance(key, str):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            overrides[key] = "1" if value else "0"
        else:
            overrides[key] = str(value)
    return overrides


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Expected list of strings")
        if item.strip() == "":
            continue
        out.append(item)
    return out


def _job_state(job_id: str) -> str:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        proc = _JOB_PROCS.get(job_id)
    if not job:
        return "missing"
    if job.state not in {"queued", "running"}:
        return job.state
    if proc is None:
        if job.pid and _pid_alive(job.pid):
            return "running"
        return job.state
    if proc.poll() is None:
        return "running"
    return job.state


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _infer_mlx_pipeline_stage(log_tail: str) -> str | None:
    if not log_tail:
        return None
    markers = [
        ("[stage] pretrain", "pretrain"),
        ("[stage] sft", "sft"),
        ("[stage] infer", "infer"),
        ("[done] MLX pipeline finished.", "done"),
        ("[abort] Interrupted", "aborted"),
    ]
    last_pos = -1
    last_stage: str | None = None
    lower = log_tail.lower()
    for needle, stage in markers:
        pos = lower.rfind(needle.lower())
        if pos > last_pos:
            last_pos = pos
            last_stage = stage
    return last_stage


def _watch_process(job_id: str, proc: subprocess.Popen) -> None:
    code = proc.wait()
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        _JOB_PROCS.pop(job_id, None)
        if not job:
            return
        job.return_code = int(code) if code is not None else None
        job.finished_at = _now()
        if job.stop_requested:
            job.state = "canceled"
        else:
            job.state = "succeeded" if code == 0 else "failed"
        _persist_jobs_locked()


def _start_subprocess(job: "TrainingJob", cmd: list[str], env: dict[str, str]) -> None:
    _ensure_jobs_root()
    log_path = _resolve_repo_path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    header = f"[dashboard] job_id={job.job_id} stage={job.stage} created_at={job.created_at}\n"
    try:
        log_handle.write(header)
        log_handle.flush()
    except Exception:
        pass

    def preexec() -> None:
        os.setsid()

    try:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=preexec if os.name != "nt" else None,
            )
        except Exception:
            with _JOBS_LOCK:
                _JOBS.pop(job.job_id, None)
                _persist_jobs_locked()
            raise
    finally:
        log_handle.close()
    with _JOBS_LOCK:
        _JOB_PROCS[job.job_id] = proc
        job.pid = proc.pid
        job.started_at = _now()
        job.state = "running"
        _persist_jobs_locked()
    threading.Thread(target=_watch_process, args=(job.job_id, proc), daemon=True).start()


@dataclass
class TrainingJob:
    job_id: str
    config_path: str
    stage: str
    kind: str
    command: list[str]
    run_id: str | None
    log_path: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    return_code: int | None = None
    state: str = "queued"
    stop_requested: bool = False

    def to_payload(self) -> dict:
        log_file = _resolve_repo_path(self.log_path)
        log_updated_at = None
        try:
            if log_file.exists():
                log_updated_at = datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
        except Exception:
            log_updated_at = None

        return {
            "id": self.job_id,
            "config_path": self.config_path,
            "stage": self.stage,
            "kind": self.kind,
            "command": self.command,
            "run_id": self.run_id,
            "log_path": self.log_path,
            "log_updated_at": log_updated_at,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pid": self.pid,
            "return_code": self.return_code,
            "state": _job_state(self.job_id),
            "stop_requested": self.stop_requested,
        }


def _start_training_from_config(config_path: str) -> TrainingJob:
    path = _resolve_repo_path(config_path)
    if CONFIG_ROOT not in path.parents:
        raise ValueError("Config not under configs/dashboard")
    cfg = _read_json(path)
    if not cfg:
        raise ValueError("Invalid config JSON")
    meta = cfg.get("meta", {}) if isinstance(cfg, dict) else {}
    stage = str(meta.get("stage") or "unknown").strip().lower()
    job_id = uuid.uuid4().hex[:12]
    created_at = _now()
    log_path = str((JOBS_ROOT / f"{job_id}.log").relative_to(REPO_ROOT))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH','')}".rstrip(os.pathsep)

    if stage in TRAINING_SCRIPTS:
        args = cfg.get("args", {}) if isinstance(cfg.get("args", {}), dict) else {}
        script = TRAINING_SCRIPTS[stage]
        cmd = [sys.executable, script] + _build_cli_args(args)
        run_id = str(Path(str(args.get("out_dir", "out"))))
        job = TrainingJob(
            job_id=job_id,
            config_path=str(path.relative_to(REPO_ROOT)),
            stage=stage,
            kind="python",
            command=cmd,
            run_id=run_id,
            log_path=log_path,
            created_at=created_at,
        )
        with _JOBS_LOCK:
            _JOBS[job.job_id] = job
        _start_subprocess(job, cmd, env)
        return job

    if stage == "mlx_pipeline":
        script = str(cfg.get("script") or "scripts/run_mlx.sh")
        script_path = _resolve_repo_path(script)
        if script_path != (REPO_ROOT / "scripts" / "run_mlx.sh"):
            raise ValueError("Only scripts/run_mlx.sh is supported for mlx_pipeline")

        script_args = _string_list(cfg.get("script_args"))
        train_args = _string_list(cfg.get("train_args"))
        env_overrides = cfg.get("env", {}) if isinstance(cfg.get("env", {}), dict) else {}
        env.update(_build_env_overrides(env_overrides))

        cmd = ["bash", "scripts/run_mlx.sh", *script_args]
        if train_args:
            cmd.extend(["--", *train_args])

        out_dir = env.get("OUT_DIR")
        if not out_dir:
            out_dir = "out/mlx_smoke" if "--smoke-test" in script_args else "out/mlx"

        job = TrainingJob(
            job_id=job_id,
            config_path=str(path.relative_to(REPO_ROOT)),
            stage=stage,
            kind="bash",
            command=cmd,
            run_id=out_dir,
            log_path=log_path,
            created_at=created_at,
        )
        with _JOBS_LOCK:
            _JOBS[job.job_id] = job
        _start_subprocess(job, cmd, env)
        return job

    if stage == "pipeline":
        paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
        stages = cfg.get("stages", {}) if isinstance(cfg.get("stages", {}), dict) else {}
        env["PRETRAIN_JSON"] = str(Path(str(paths.get("pretrain_data", "dataset/identity_cn_sample.jsonl"))))
        env["SFT_JSON"] = str(Path(str(paths.get("sft_data", "data/chinese/identity_conversations.jsonl"))))
        env["DPO_JSON"] = str(Path(str(paths.get("dpo_data", "dataset/preference_cn_sample.jsonl"))))
        env["TF_DIR"] = str(Path(str(paths.get("tensorboard_root", "out/logs/dashboard"))))
        env["OUT_DIR"] = str(Path(str(paths.get("out_dir", "out/pipeline_dashboard"))))

        pretrain = stages.get("pretrain", {}) if isinstance(stages.get("pretrain", {}), dict) else {}
        env["MODEL_HIDDEN_SIZE"] = str(pretrain.get("hidden_size", 512))
        env["MODEL_NUM_LAYERS"] = str(pretrain.get("num_hidden_layers", 8))

        def extra_args(block: dict) -> str:
            extras: list[str] = []
            for k, v in block.items():
                if k == "script":
                    continue
                if v is None:
                    continue
                if k == "use_moe":
                    if bool(v):
                        extras.extend([f"--{k}", "True"])
                    continue
                if isinstance(v, bool):
                    if v:
                        extras.append(f"--{k}")
                    continue
                if k == "device":
                    resolved = _choose_device(str(v))
                    if resolved:
                        extras.extend([f"--{k}", resolved])
                    continue
                extras.extend([f"--{k}", str(v)])
            return " ".join(extras)

        env["PRETRAIN_ARGS"] = extra_args(stages.get("pretrain", {}) if isinstance(stages.get("pretrain", {}), dict) else {})
        env["SFT_ARGS"] = extra_args(stages.get("sft", {}) if isinstance(stages.get("sft", {}), dict) else {})
        env["DPO_ARGS"] = extra_args(stages.get("dpo", {}) if isinstance(stages.get("dpo", {}), dict) else {})

        cmd = ["bash", "scripts/run.sh"]
        job = TrainingJob(
            job_id=job_id,
            config_path=str(path.relative_to(REPO_ROOT)),
            stage=stage,
            kind="bash",
            command=cmd,
            run_id=env["OUT_DIR"],
            log_path=log_path,
            created_at=created_at,
        )
        with _JOBS_LOCK:
            _JOBS[job.job_id] = job
        _start_subprocess(job, cmd, env)
        return job

    raise ValueError(f"Unsupported stage: {stage}")


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    _load_jobs_from_disk()

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/static/<path:filename>")
    def static_assets(filename: str):  # type: ignore[override]
        res = send_from_directory(app.static_folder, filename, max_age=0)
        res.headers["Cache-Control"] = "no-store, max-age=0"
        return res

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

    @app.route("/api/jobs")
    def jobs():
        with _JOBS_LOCK:
            job_list = [job.to_payload() for job in _JOBS.values()]
        job_list.sort(key=lambda j: j.get("created_at") or "", reverse=True)
        return jsonify(job_list)

    @app.route("/api/jobs", methods=["POST"])
    def jobs_create():
        payload = request.get_json(force=True) or {}
        config_path = payload.get("config_path")
        if not config_path or not isinstance(config_path, str):
            return jsonify({"error": "Missing config_path"}), 400
        try:
            job = _start_training_from_config(config_path)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(job.to_payload())

    @app.route("/api/jobs/<job_id>")
    def jobs_detail(job_id: str):
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        max_bytes = request.args.get("max_bytes", default="96000")
        try:
            max_bytes_i = int(max_bytes)
        except ValueError:
            max_bytes_i = 96_000
        tail = _tail_text(_resolve_repo_path(job.log_path), max_bytes=max_bytes_i)
        derived: dict[str, object] = {}
        if job.stage == "mlx_pipeline" and job.run_id:
            out_dir = job.run_id
            associated_runs = [out_dir, f"{out_dir}/pretrain", f"{out_dir}/sft"]
            current = _infer_mlx_pipeline_stage(tail)
            primary_run = out_dir
            if current in {"pretrain", "sft"}:
                primary_run = f"{out_dir}/{current}"
            elif current in {"infer", "done"}:
                primary_run = f"{out_dir}/sft"
            derived = {
                "current_stage": current,
                "associated_runs": associated_runs,
                "primary_run_id": primary_run,
            }
        return jsonify({"job": job.to_payload(), "log_tail": tail, "derived": derived})

    @app.route("/api/jobs/<job_id>/stop", methods=["POST"])
    def jobs_stop(job_id: str):
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            proc = _JOB_PROCS.get(job_id)
            if job:
                job.stop_requested = True
                _persist_jobs_locked()
        if not job:
            return jsonify({"error": "Job not found"}), 404
        pid = None
        if proc and proc.poll() is None:
            pid = proc.pid
        elif job.pid:
            pid = int(job.pid)

        if pid:
            def _send(sig: int) -> None:
                if os.name != "nt":
                    try:
                        os.killpg(pid, sig)
                        return
                    except Exception:
                        pass
                try:
                    os.kill(pid, sig)
                except Exception:
                    pass

            # Prefer SIGINT because scripts/run_mlx.sh traps INT.
            _send(signal.SIGINT)
            time.sleep(0.6)
            if _pid_alive(pid):
                _send(signal.SIGTERM)
            time.sleep(0.6)
            if _pid_alive(pid):
                _send(signal.SIGKILL)

            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
                if job and job.pid and not _pid_alive(int(job.pid)):
                    job.state = "canceled"
                    job.finished_at = _now()
                    _persist_jobs_locked()
        return jsonify(job.to_payload())

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
