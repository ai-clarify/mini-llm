from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse


class DataDownloadError(RuntimeError):
    pass


def _max_bytes(max_download_mb: int) -> Optional[int]:
    if max_download_mb <= 0:
        return None
    return int(max_download_mb) * 1024 * 1024


def _split_csv(spec: str) -> List[str]:
    return [p.strip() for p in spec.split(",") if p.strip()]


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _normalize_data_source(source: Optional[str]) -> str:
    if source is None or not str(source).strip():
        source = os.environ.get("MINIMIND_DATA_SOURCE") or os.environ.get("DATA_SOURCE") or "modelscope"
    value = str(source).strip().lower()
    if value in {"modelscope", "ms", "model-scope"}:
        return "modelscope"
    if value in {"hf", "huggingface", "huggingface_hub"}:
        return "huggingface"
    raise ValueError(f"Unsupported data source: {source}")


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name:
        raise DataDownloadError(f"Cannot infer filename from url: {url}")
    return name


def _atomic_write(path: Path, data_iter, *, max_bytes: Optional[int] = None) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = 0
        with open(tmp, "wb") as f:
            for chunk in data_iter:
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise DataDownloadError(
                        f"Refusing to download large file (> {max_bytes/1024/1024:.0f} MiB). "
                        "Increase --max_download_mb or set it to 0 to disable the limit."
                    )
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _url_content_length(url: str, *, timeout: int) -> Optional[int]:
    import requests

    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None
    value = resp.headers.get("Content-Length") or resp.headers.get("content-length")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def ensure_url_file(
    url: str,
    *,
    data_dir: str,
    force: bool = False,
    timeout: int = 60,
    max_download_mb: int = 0,
) -> str:
    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise DataDownloadError("`requests` is required for URL downloads. Please `pip install requests`.") from e

    filename = _filename_from_url(url)
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return str(out_path)

    max_bytes = _max_bytes(max_download_mb)
    content_length = _url_content_length(url, timeout=timeout)
    if max_bytes is not None and content_length is not None and content_length > max_bytes:
        raise DataDownloadError(
            f"Refusing to download {url} ({content_length/1024/1024:.1f} MiB) because it exceeds "
            f"--max_download_mb={max_download_mb}. Increase the limit or set it to 0."
        )

    print(f"[data] downloading {url} -> {out_path}")
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    _atomic_write(out_path, resp.iter_content(chunk_size=1024 * 1024), max_bytes=max_bytes)
    if out_path.stat().st_size <= 0:
        raise DataDownloadError(f"Downloaded file is empty: {out_path}")
    return str(out_path)


def ensure_hf_dataset_file(
    *,
    repo_id: str,
    filename: str,
    data_dir: str,
    endpoint: Optional[str] = None,
    force: bool = False,
    max_download_mb: int = 0,
) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    local_path = Path(data_dir) / filename
    if local_path.exists() and local_path.stat().st_size > 0 and not force:
        return os.fspath(local_path)

    max_bytes = _max_bytes(max_download_mb)
    if max_bytes is not None:
        api = HfApi(endpoint=endpoint) if endpoint else HfApi()
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, expand=True):
            if item.path == filename:
                size = item.size
                if size > max_bytes:
                    raise DataDownloadError(
                        f"Refusing to download {repo_id}/{filename} ({size/1024/1024:.1f} MiB) because it exceeds "
                        f"--max_download_mb={max_download_mb}. Increase the limit or set it to 0."
                    )
                break

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=data_dir,
        force_download=force,
        endpoint=endpoint,
    )
    return os.fspath(path)


def ensure_ms_dataset_file(
    *,
    repo_id: str,
    filename: str,
    data_dir: str,
    force: bool = False,
    max_download_mb: int = 0,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        try:
            from modelscope.hub.snapshot_download import dataset_snapshot_download
        except Exception:
            dataset_snapshot_download = None
    except Exception as e:  # pragma: no cover
        raise DataDownloadError(
            "Missing `modelscope` for dataset download. Install via `pip install modelscope` "
            "or set MINIMIND_DATA_SOURCE=hf to use HuggingFace."
        ) from e

    local_path = Path(data_dir) / filename
    if local_path.exists() and local_path.stat().st_size > 0 and not force:
        return os.fspath(local_path)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    if force and local_path.exists():
        try:
            local_path.unlink()
        except OSError:
            pass

    snapshot_fn = snapshot_download
    if "repo_type" not in inspect.signature(snapshot_download).parameters:
        if dataset_snapshot_download is not None:
            snapshot_fn = dataset_snapshot_download
        else:  # pragma: no cover - legacy fallback
            raise DataDownloadError(
                "Your modelscope version does not support dataset downloads. "
                "Upgrade modelscope or set MINIMIND_DATA_SOURCE=hf."
            )

    sig = inspect.signature(snapshot_fn)
    params = sig.parameters
    kwargs = {}
    if "local_dir" in params:
        kwargs["local_dir"] = data_dir
    if "local_dir_use_symlinks" in params:
        kwargs["local_dir_use_symlinks"] = False
    if "cache_dir" in params and cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if "revision" in params and revision is not None:
        kwargs["revision"] = revision
    if "force_download" in params:
        kwargs["force_download"] = force
    elif "force" in params:
        kwargs["force"] = force
    if "repo_type" in params:
        kwargs["repo_type"] = "dataset"

    if "allow_file_pattern" in params:
        kwargs["allow_file_pattern"] = [filename]
    elif "allow_patterns" in params:
        kwargs["allow_patterns"] = [filename]
    elif "allow_pattern" in params:
        kwargs["allow_pattern"] = [filename]

    snapshot_fn(repo_id, **kwargs)

    candidate = local_path
    if not candidate.exists():
        matches = list(Path(data_dir).rglob(filename))
        if matches:
            candidate = matches[0]
        else:
            raise DataDownloadError(f"Downloaded dataset file not found: {filename}")

    max_bytes = _max_bytes(max_download_mb)
    if max_bytes is not None and candidate.stat().st_size > max_bytes:
        raise DataDownloadError(
            f"Refusing to download {repo_id}/{filename} ({candidate.stat().st_size/1024/1024:.1f} MiB) because it exceeds "
            f"--max_download_mb={max_download_mb}. Increase the limit or set it to 0."
        )

    return os.fspath(candidate)


def ensure_dataset_file(
    *,
    data_source: Optional[str],
    repo_id: str,
    filename: str,
    data_dir: str,
    endpoint: Optional[str],
    force: bool,
    max_download_mb: int,
    ms_revision: Optional[str] = None,
    ms_cache_dir: Optional[str] = None,
) -> str:
    source = _normalize_data_source(data_source)
    if source == "modelscope":
        return ensure_ms_dataset_file(
            repo_id=repo_id,
            filename=filename,
            data_dir=data_dir,
            force=force,
            max_download_mb=max_download_mb,
            revision=ms_revision,
            cache_dir=ms_cache_dir,
        )
    return ensure_hf_dataset_file(
        repo_id=repo_id,
        filename=filename,
        data_dir=data_dir,
        endpoint=endpoint,
        force=force,
        max_download_mb=max_download_mb,
    )


def resolve_data_path_spec(
    spec: str,
    *,
    task: str,
    data_dir: str,
    hf_repo_id: str,
    hf_endpoint: Optional[str],
    force_download: bool,
    max_download_mb: int,
    data_source: Optional[str] = None,
    ms_repo_id: Optional[str] = None,
    ms_revision: Optional[str] = None,
    ms_cache_dir: Optional[str] = None,
) -> str:
    """
    Expand a comma-separated `--data_path` spec into local paths.

    Supported entries:
    - Local file/dir/glob (passed through).
    - URL (http/https): downloaded into `data_dir`.
    - `minimind:auto` / `minimind`: maps to the recommended dataset for `task`.
    - `minimind:small`: maps to a smaller SFT dataset (good for quick runs).
    - `minimind:smoke`: maps to a tiny SFT dataset (smoke test).
    - `minimind:<filename>`: downloads from ModelScope or HuggingFace dataset repo (see MINIMIND_DATA_SOURCE).
    """

    pieces = _split_csv(spec)
    if not pieces:
        raise ValueError("data_path is empty")

    source = _normalize_data_source(data_source)
    ms_repo_id = ms_repo_id or hf_repo_id

    out: List[str] = []
    for piece in pieces:
        if piece in ("minimind", "minimind:auto", "minimind:full", "minimind:recommended"):
            if task == "pretrain":
                piece = "minimind:pretrain_hq.jsonl"
            elif task == "sft":
                piece = "minimind:sft_mini_512.jsonl"
            else:
                raise ValueError(f"Unsupported task for minimind:auto: {task}")
        elif piece == "minimind:small":
            piece = "minimind:lora_medical.jsonl"
        elif piece == "minimind:smoke":
            piece = "minimind:lora_identity.jsonl"

        if piece.startswith("minimind:"):
            filename = piece.split(":", 1)[1].strip()
            if not filename:
                raise ValueError("minimind: spec requires a filename, e.g. minimind:pretrain_hq.jsonl")
            if not filename.endswith(".jsonl"):
                filename = f"{filename}.jsonl"
            requested_filename = filename
            remote_filename = filename
            if requested_filename == "dpo_pairs.jsonl":
                remote_filename = "dpo.jsonl"
            repo_id = ms_repo_id if source == "modelscope" else hf_repo_id
            local_path = ensure_dataset_file(
                data_source=source,
                repo_id=repo_id,
                filename=remote_filename,
                data_dir=data_dir,
                endpoint=hf_endpoint,
                force=force_download,
                max_download_mb=max_download_mb,
                ms_revision=ms_revision,
                ms_cache_dir=ms_cache_dir,
            )
            if requested_filename != remote_filename:
                alias_path = Path(data_dir) / requested_filename
                target_path = Path(local_path)
                try:
                    target_abs = target_path.resolve()
                    alias_parent_abs = alias_path.parent.resolve()
                    try:
                        link_target: Path = target_abs.relative_to(alias_parent_abs)
                    except ValueError:
                        link_target = target_abs
                except OSError:
                    link_target = target_path
                # Note: Path.exists() returns False for broken symlinks, but the path still exists.
                # Use is_symlink() to detect and fix stale aliases safely.
                if alias_path.is_symlink() and not alias_path.exists():
                    try:
                        alias_path.unlink()
                    except OSError:
                        pass

                if force_download and (alias_path.exists() or alias_path.is_symlink()):
                    try:
                        alias_path.unlink()
                    except OSError:
                        pass

                if not alias_path.exists():
                    try:
                        alias_path.symlink_to(link_target)
                    except FileExistsError:
                        try:
                            alias_path.unlink()
                        except OSError:
                            pass
                        alias_path.symlink_to(link_target)
                    except OSError:
                        if alias_path.is_symlink():
                            try:
                                alias_path.unlink()
                            except OSError:
                                pass
                        alias_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(local_path, alias_path)
                local_path = os.fspath(alias_path)
            print(f"[data] ready: {local_path}")
            out.append(local_path)
            continue

        if _is_url(piece):
            out.append(
                ensure_url_file(piece, data_dir=data_dir, force=force_download, max_download_mb=max_download_mb)
            )
            continue

        out.append(piece)

    return ",".join(out)
