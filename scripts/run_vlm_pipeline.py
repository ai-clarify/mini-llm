#!/usr/bin/env python3
"""One-click pipeline for MiniLLM VLM training and evaluation.

The script automates three stages:

1. Download required assets from ModelScope (vision encoder, base LLM
   checkpoints, datasets and evaluation images).
2. Launch supervised fine-tuning for the VLM model with sensible
   defaults that run on a single GPU.
3. Evaluate the trained checkpoint on the bundled evaluation images.

By default the script downloads the light-weight single/multi image SFT
subset of the official ``gongjy/minimind-v_dataset`` so that the
pipeline can run in a constrained environment.  Pass ``--download-full``
if you would like to fetch the full dataset (this requires more than
1GB of disk space).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def ensure_snapshot(
    repo_id: str,
    repo_type: str,
    target_dir: Path,
    allow_patterns: Iterable[str] | None,
    cache_dir: Path,
    force: bool,
) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        cache_dir=str(cache_dir),
        local_dir=str(target_dir),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    return target_dir


def extract_zip(archive: Path, target_dir: Path) -> None:
    if not archive.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'r') as zf:
        zf.extractall(target_dir)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prepare_dataset(args) -> tuple[Path, Path]:
    dataset_root = Path('dataset/vlm')
    download_dir = dataset_root / '_download'
    dataset_root.mkdir(parents=True, exist_ok=True)

    allow_patterns = args.allow_pattern
    if not allow_patterns:
        allow_patterns = [
            'sft_vlm_data_multi.jsonl',
            'sft_multi_images_trans_image.zip',
            'dataset/eval_images/*',
            'README.md',
        ]
        if args.download_full:
            allow_patterns.extend([
                'sft_data.jsonl',
                'sft_images.zip',
                'pretrain_data.jsonl',
                'pretrain_images.zip',
            ])

    ensure_snapshot(
        repo_id=args.dataset_repo,
        repo_type='dataset',
        target_dir=download_dir,
        allow_patterns=allow_patterns,
        cache_dir=args.cache_dir,
        force=args.force,
    )

    for json_name in ('sft_data.jsonl', 'sft_vlm_data_multi.jsonl'):
        src = download_dir / json_name
        if src.exists():
            shutil.copy2(src, dataset_root / json_name)

    extract_zip(download_dir / 'sft_images.zip', dataset_root / 'sft_images')
    extract_zip(download_dir / 'sft_multi_images_trans_image.zip', dataset_root / 'sft_images')

    eval_src = download_dir / 'dataset' / 'eval_images'
    if eval_src.exists():
        copy_tree(eval_src, dataset_root / 'eval_images')

    image_dir = dataset_root / 'sft_images'
    has_jpg = any(image_dir.glob('**/*.jpg'))
    has_png = any(image_dir.glob('**/*.png'))
    has_jpeg = any(image_dir.glob('**/*.jpeg'))
    if not (has_jpg or has_png or has_jpeg):
        raise RuntimeError(
            'No training images were extracted. Use --download-full to fetch the full dataset or provide custom patterns.'
        )

    primary_json = dataset_root / 'sft_data.jsonl'
    if not primary_json.exists():
        # Fall back to the smaller multi-image dataset when the full json is not downloaded.
        primary_json = dataset_root / 'sft_vlm_data_multi.jsonl'
        if not primary_json.exists():
            raise RuntimeError('Could not locate an SFT json file in the downloaded dataset.')
    return primary_json, image_dir


def prepare_vision_encoder(args) -> Path:
    vision_dir = Path('model/vision_model/clip-vit-base-patch16')
    ensure_snapshot(
        repo_id=args.vision_repo,
        repo_type='model',
        target_dir=vision_dir,
        allow_patterns=None,
        cache_dir=args.cache_dir,
        force=args.force,
    )
    return vision_dir


def prepare_base_llm(args) -> Path:
    out_dir = Path('out')
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path('model/_llm_download')
    ensure_snapshot(
        repo_id=args.model_repo,
        repo_type='model',
        target_dir=download_dir,
        allow_patterns=['llm_512.pth'],
        cache_dir=args.cache_dir,
        force=args.force,
    )
    src = download_dir / 'llm_512.pth'
    if not src.exists():
        raise RuntimeError('The base llm_512.pth weight was not downloaded; please check the repository layout.')
    dst = out_dir / 'pretrain_vlm_512.pth'
    if not dst.exists() or args.force:
        shutil.copy2(src, dst)
    return dst


def run_training(args, data_json: Path, image_dir: Path, vision_dir: Path) -> None:
    train_cmd = [
        sys.executable,
        'trainer/train_vlm_sft.py',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--hidden_size', str(args.hidden_size),
        '--num_hidden_layers', str(args.num_hidden_layers),
        '--max_seq_len', str(args.max_seq_len),
        '--data_path', str(data_json.resolve()),
        '--images_path', str(image_dir.resolve()),
        '--vision_model_path', str(vision_dir.resolve()),
        '--tokenizer_path', str(Path('model').resolve()),
        '--save_dir', str(Path('out').resolve()),
    ]
    if args.use_moe:
        train_cmd.extend(['--use_moe', '1'])
    if args.use_wandb:
        train_cmd.append('--use_wandb')
    run_cmd(train_cmd)


def run_evaluation(args, vision_dir: Path) -> None:
    eval_cmd = [
        sys.executable,
        'eval_vlm.py',
        '--load_from', 'model',
        '--save_dir', 'out',
        '--weight', args.eval_weight,
        '--hidden_size', str(args.hidden_size),
        '--num_hidden_layers', str(args.num_hidden_layers),
        '--image_dir', str(Path('dataset/vlm/eval_images').resolve()),
        '--vision_model_path', str(vision_dir.resolve()),
        '--max_new_tokens', str(args.eval_max_new_tokens),
    ]
    if args.use_moe:
        eval_cmd.extend(['--use_moe', '1'])
    run_cmd(eval_cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MiniLLM VLM automation pipeline')
    parser.add_argument('--dataset-repo', default='gongjy/minimind-v_dataset', help='ModelScope dataset repo id')
    parser.add_argument('--model-repo', default='gongjy/MiniMind2-V-PyTorch', help='ModelScope base LLM repo id')
    parser.add_argument('--vision-repo', default='openai-mirror/clip-vit-base-patch16', help='ModelScope vision encoder repo id')
    parser.add_argument('--cache-dir', type=Path, default=Path.home() / '.cache' / 'modelscope', help='ModelScope cache directory')
    parser.add_argument('--allow-pattern', nargs='*', help='Custom allow_patterns passed to ModelScope snapshot_download')
    parser.add_argument('--download-full', action='store_true', help='Download the full dataset (slow and large)')
    parser.add_argument('--force', action='store_true', help='Redownload assets even if they already exist locally')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs for the demo run')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for the demo run')
    parser.add_argument('--learning-rate', type=float, default=5e-6, help='Learning rate for the demo run')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the VLM')
    parser.add_argument('--num-hidden-layers', type=int, default=8, help='Transformer layer count')
    parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length during training')
    parser.add_argument('--use-moe', action='store_true', help='Enable MoE configuration for the VLM')
    parser.add_argument('--use-wandb', action='store_true', help='Enable SwanLab/W&B logging during training')
    parser.add_argument('--eval-weight', default='sft_vlm', help='Checkpoint prefix used during evaluation')
    parser.add_argument('--eval-max-new-tokens', type=int, default=256, help='Maximum tokens generated during evaluation')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print('=== 1. Downloading assets from ModelScope ===')
    data_json, image_dir = prepare_dataset(args)
    vision_dir = prepare_vision_encoder(args)
    prepare_base_llm(args)

    print('=== 2. Launching training run ===')
    run_training(args, data_json, image_dir, vision_dir)

    print('=== 3. Evaluating checkpoint ===')
    run_evaluation(args, vision_dir)


if __name__ == '__main__':
    main()
