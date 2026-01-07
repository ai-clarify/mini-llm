import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.vlm_dataset import VLMDataset
from model.model_vlm import VLMConfig
from trainer.trainer_utils import (
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_vlm_model,
    is_main_process,
    setup_seed,
    vlm_checkpoint,
    Logger,
)

warnings.filterwarnings('ignore')


def train_epoch(args, vlm_config, model, optimizer, scaler, autocast_ctx, loader, epoch, iters, start_step, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    model.train()

    for step, (X, Y, loss_mask, pixel_values) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X, pixel_values=pixel_values)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = (loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min')
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
            clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(
                vlm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler,
            )
            model.train()


def main():
    parser = argparse.ArgumentParser(description="MiniLLM-V SFT")
    parser.add_argument('--save_dir', type=str, default='../out', help='模型保存目录')
    parser.add_argument('--save_weight', type=str, default='sft_vlm', help='保存权重前缀')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='初始学习率')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'], help='混合精度类型')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=100, help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='模型保存间隔')
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='隐藏层数量')
    parser.add_argument('--max_seq_len', type=int, default=1536, help='最大序列长度')
    parser.add_argument('--use_moe', type=int, default=0, choices=[0, 1], help='是否启用MoE结构')
    parser.add_argument('--data_path', type=str, default='../dataset/vlm/sft_data.jsonl', help='训练数据路径')
    parser.add_argument('--images_path', type=str, default='../dataset/vlm/sft_images', help='训练图像路径')
    parser.add_argument('--tokenizer_path', type=str, default='../model', help='分词器路径')
    parser.add_argument('--vision_model_path', type=str, default='../model/vision_model/clip-vit-base-patch16', help='视觉模型路径')
    parser.add_argument('--from_weight', type=str, default='pretrain_vlm', help='继承的基础权重名称，为none表示不继承')
    parser.add_argument('--from_resume', type=int, default=0, choices=[0, 1], help='是否自动检测续训')
    parser.add_argument('--use_wandb', action='store_true', help='是否启用SwanLab/W&B日志')
    parser.add_argument('--wandb_project', type=str, default='MiniLLM-V-SFT', help='SwanLab项目名')
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniLLM-V-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=args.tokenizer_path,
        vision_model_path=args.vision_model_path,
        save_dir=args.save_dir,
        device=args.device,
    )

    train_ds = VLMDataset(
        args.data_path,
        args.images_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        max_length=vlm_config.max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        if ckp_data.get('optimizer'):
            optimizer.load_state_dict(ckp_data['optimizer'])
        if ckp_data.get('scaler'):
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            sampler_iterable = train_sampler or range(len(train_ds))
            batch_sampler = SkipBatchSampler(sampler_iterable, args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(args, vlm_config, model, optimizer, scaler, autocast_ctx, loader, epoch, len(loader) + start_step + 1, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(args, vlm_config, model, optimizer, scaler, autocast_ctx, loader, epoch, len(loader), 0, wandb)


if __name__ == '__main__':
    main()
