import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings

# Suppress pynvml deprecation warning from torch.cuda
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda')

import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM
from dataset.lm_dataset import PretrainDataset
from checkpoint_loader import CheckpointLoader
from trainer import checkpoint_manager
from trainer.loss_utils import compute_mtp_loss
from trainer.muon import Muon, get_muon_param_groups
from trainer.training_schedule import apply_projection_zero_init, apply_back_out_scaling

warnings.filterwarnings('ignore')


training_state = {"max_steps": None, "global_step": 0, "stop": False}
ckpt_root = None


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    if training_state["stop"]:
        return

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if step < training_state.get("step_in_epoch", 0):
            continue
        if training_state["stop"]:
            break
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss += compute_mtp_loss(res.mtp_logits, Y, loss_mask, weight=lm_config.mtp_loss_weight)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        did_update = False
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            training_state["global_step"] += 1
            did_update = True

        if step % args.log_interval == 0 or step == iter_per_epoch - 1:
            spend_time = time.time() - start_time
            # Calculate ETA: (time_per_step * remaining_steps) / 60
            time_per_step = spend_time / (step + 1)
            remaining_steps = iter_per_epoch - step - 1
            eta_min = time_per_step * remaining_steps / 60
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.6f} lr:{:.12f} ETA:{:.1f}min'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    eta_min))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "eta_min": eta_min})

            if writer is not None and (not ddp or dist.get_rank() == 0):
                global_step = training_state["global_step"]
                writer.add_scalar("train/loss", loss.item() * args.accumulation_steps, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[-1]['lr'], global_step)

        if (
            (did_update and training_state["global_step"] % args.save_interval == 0)
            or step == iter_per_epoch - 1
        ) and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            state = {
                "epoch": epoch,
                "step_in_epoch": step + 1,
                "global_step": training_state["global_step"],
                "args": vars(args),
            }
            checkpoint_manager.save_checkpoint(
                ckpt_root=ckpt_root,
                step=training_state["global_step"],
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                state=state,
            )
            checkpoint_manager.prune_checkpoints(ckpt_root, keep_last=args.keep_last_checkpoints)
            model.train()

        max_steps = training_state["max_steps"]
        if max_steps is not None and training_state["global_step"] >= max_steps:
            Logger(f"[smoke] Reached max_steps={max_steps}, stopping early")
            training_state["stop"] = True
            break


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    model = MiniLLMForCausalLM(lm_config)

    # Determine checkpoint path with multiple fallback strategies
    ckp_path = CheckpointLoader.resolve_checkpoint_path(
        explicit_path=args.pretrained_path,
        stage='pretrain' if not args.load_from_remote else None,
        hidden_size=args.hidden_size if not args.load_from_remote else None,
        use_moe=lm_config.use_moe,
        env_var='MINILLM_PRETRAINED_PATH',
        local_dir=args.out_dir,
        remote_dir='/openbayes/home/out',
        logger=Logger,
    )

    # Load checkpoint if found
    if ckp_path:
        CheckpointLoader.load_checkpoint(model, ckp_path, device=args.device, logger=Logger)
    else:
        Logger('No pretrained checkpoint found, initializing with random weights')

    model = model.to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    # torch.compile for faster training (PyTorch 2.0+)
    if args.use_compile and hasattr(torch, 'compile'):
        Logger('[optim] Compiling model with torch.compile (may take a few minutes on first run)')
        model = torch.compile(model, mode=args.compile_mode)

    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniLLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniLLM-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None, help="Path to step checkpoint dir")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Root dir for step checkpoints")
    parser.add_argument("--keep_last_checkpoints", type=int, default=3)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "jsonl", "bin", "bin2d"],
        help="Dataset format: auto/jsonl/bin/bin2d (bin2d uses packed fixed-length ids).",
    )
    parser.add_argument("--bin_cache", type=str, default="mmap", choices=["mmap", "memory"])
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CPU memory for faster H2D transfers (only when using CUDA).",
    )
    parser.add_argument("--tensorboard_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Limit total training iterations (for smoke tests)")
    parser.add_argument("--mtp_loss_weight", type=float, default=0.1, help="Weight for MTP auxiliary loss.")
    parser.add_argument("--paired_heads", action="store_true", help="Enable paired attention heads.")
    parser.add_argument("--qk_norm", action="store_true", help="Enable QK RMSNorm inside attention.")
    parser.add_argument("--qk_norm_eps", type=float, default=1e-6)
    parser.add_argument("--value_mix", type=float, default=0.0)
    parser.add_argument("--logit_softcap", type=float, default=0.0)

    # Optimizer arguments (modded-nanogpt style)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
                        help="Optimizer: adamw or muon (Newton-Schulz orthogonalization)")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Learning rate for Muon (2D params)")
    parser.add_argument("--muon_momentum", type=float, default=0.95, help="Momentum for Muon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--cautious_wd", action="store_true", help="Cautious weight decay (only when grad and param agree)")

    # Training optimizations (modded-nanogpt style)
    parser.add_argument("--zero_init_proj", action="store_true",
                        help="Zero-init output projections (muP-like)")
    parser.add_argument("--back_out_layers", type=int, default=0,
                        help="Number of early layers to scale down (back out contributions)")

    # Pretrained model checkpoint arguments
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained model checkpoint (supports /openbayes/home/out)")
    # torch.compile for faster training
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile for faster training")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")

    parser.add_argument("--load_from_remote", action="store_true",
                        help="Load pretrained model from /openbayes/home/out instead of local directory")
    args = parser.parse_args()

    if args.max_steps is not None and args.max_steps <= 0:
        args.max_steps = None

    lm_config = MiniLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        mtp_loss_weight=args.mtp_loss_weight,
        paired_heads=bool(args.paired_heads),
        qk_norm=bool(args.qk_norm),
        qk_norm_eps=float(args.qk_norm_eps),
        value_mix=float(args.value_mix),
        logit_softcap=float(args.logit_softcap),
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_root = args.ckpt_dir or os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniLLM-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import swanlab as wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    global writer
    writer = None
    if args.tensorboard_dir and (not ddp or dist.get_rank() == 0):
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tensorboard_dir)

    model, tokenizer = init_model(lm_config)

    # Apply modded-nanogpt style optimizations
    if args.zero_init_proj:
        Logger("[optim] Applying projection zero-init (muP-like)")
        apply_projection_zero_init(model)

    if args.back_out_layers > 0:
        Logger(f"[optim] Applying back-out scaling to first {args.back_out_layers} layers")
        apply_back_out_scaling(model, back_out_layers=args.back_out_layers)

    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        data_format=args.data_format,
        bin_cache=args.bin_cache,
    )
    train_sampler = DistributedSampler(train_ds) if ddp else None
    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
    persistent_workers = bool(args.persistent_workers) if args.num_workers > 0 else False
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # Optimizer selection
    if args.optimizer == "muon":
        Logger("[optim] Using Muon optimizer (Newton-Schulz orthogonalization)")
        param_groups = get_muon_param_groups(
            model,
            lr=args.muon_lr,
            adamw_lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer = Muon(
            param_groups,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            adamw_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            cautious=args.cautious_wd,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    global iter_per_epoch
    iter_per_epoch = len(train_loader)
    training_state["max_steps"] = args.max_steps
    training_state["global_step"] = 0
    training_state["stop"] = False
    training_state["step_in_epoch"] = 0

    if args.resume:
        resume_state = checkpoint_manager.load_checkpoint(
            ckpt_path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=str(args.device),
        )
        training_state["global_step"] = int(resume_state.get("global_step", 0))
        training_state["step_in_epoch"] = int(resume_state.get("step_in_epoch", 0))
        start_epoch = int(resume_state.get("epoch", 0))
        Logger(
            f"[resume] step={training_state['global_step']} epoch={start_epoch + 1} "
            f"step_in_epoch={training_state['step_in_epoch']}"
        )
        if training_state["step_in_epoch"] >= iter_per_epoch:
            start_epoch += 1
            training_state["step_in_epoch"] = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        if training_state["stop"]:
            break
        train_sampler and train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb)
        if training_state["stop"]:
            Logger("[smoke] Early stop triggered, exiting training loop")
            break
        training_state["step_in_epoch"] = 0

    if writer is not None:
        writer.flush()
        writer.close()
