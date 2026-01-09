"""Evaluate a trained MiniLLM VLM checkpoint."""
from __future__ import annotations

import argparse
import os
import warnings

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from model.model_vlm import MiniLLMVLM, VLMConfig
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniLLMVLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            vision_model_path=args.vision_model_path,
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniLLMVLM.get_vision_model(args.vision_model_path)

    print(f'VLM模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M(illion)')
    preprocess = model.processor
    return model.eval().to(args.device), tokenizer, preprocess


def main():
    parser = argparse.ArgumentParser(description='MiniLLM-V Chat')
    parser.add_argument('--load_from', default='model', type=str, help='模型加载路径（model=原生权重，其他路径=transformers格式）')
    parser.add_argument('--save_dir', default='out', type=str, help='模型权重目录')
    parser.add_argument('--weight', default='sft_vlm', type=str, help='权重名称前缀（pretrain_vlm, sft_vlm）')
    parser.add_argument('--hidden_size', default=512, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用MoE架构')
    parser.add_argument('--max_new_tokens', default=256, type=int, help='最大生成长度')
    parser.add_argument('--temperature', default=0.65, type=float, help='生成温度')
    parser.add_argument('--top_p', default=0.85, type=float, help='nucleus采样阈值')
    parser.add_argument('--image_dir', default='./dataset/vlm/eval_images', type=str, help='测试图像目录')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备')
    parser.add_argument('--vision_model_path', default='./model/vision_model/clip-vit-base-patch16', type=str, help='视觉模型路径')
    args = parser.parse_args()

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    prompt = "仔细看一下这张图：\n\n<image>\n\n描述一下这个图像的内容。"

    for image_file in sorted(os.listdir(args.image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            setup_seed(2026)
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            pixel_values = MiniLLMVLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)

            messages = [{"role": "user", "content": prompt.replace('<image>', model.params.image_special_token)}]
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(inputs_text, return_tensors='pt', truncation=True).to(args.device)

            print(f'[图像]: {image_file}')
            prompt_display = prompt.replace("\n", "\\n")
            print(f'👶: {prompt_display}')
            print('🤖️: ', end='')
            model.generate(
                inputs=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                pixel_values=pixel_values,
            )
            print('\n')


if __name__ == '__main__':
    main()
