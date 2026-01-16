from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

import mlx.core as mx

from .bench import (
    Example,
    _load_config,
    _resolve_checkpoint,
    _system_prompt_for,
    build_copy_bench,
    build_json_bench,
    build_knowledge_bench,
    build_logic_bench,
    build_math_mcq,
    build_qa_bench,
    build_sort_bench,
    extract_choice,
    score_example,
)
from .infer import generate, load_state_seq_len
from ..config import MiniLLMConfig
from ..models import MiniLLMForCausalLM
from ..nn.lora import merge_lora


@dataclass(frozen=True)
class DemoExample:
    title: str
    prompt: str
    expected_contains: Optional[List[str]] = None


def _fmt_sep(char: str = "-", n: int = 72) -> str:
    return char * n


def _chat_generate(
    *,
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    system: str,
    user: str,
    max_new_tokens: int,
    max_seq_len: Optional[int],
    seed: int,
) -> str:
    mx.random.seed(int(seed))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids: List[int] = tokenizer.encode(prompt_text, add_special_tokens=False)
    out_ids = generate(
        model,
        input_ids=prompt_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        min_new_tokens=0,
        banned_token_ids=list(tokenizer.all_special_ids),
        temperature=0.0,
        top_p=1.0,
        max_seq_len=max_seq_len,
    )
    resp_ids = out_ids[len(prompt_ids) :]
    return tokenizer.decode(resp_ids, skip_special_tokens=True).strip()


def _print_scored_example(
    *,
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    ex: Example,
    max_new_tokens: int,
    max_seq_len: Optional[int],
    seed: int,
) -> None:
    system = _system_prompt_for(ex)
    t0 = time.perf_counter()
    out = _chat_generate(
        model=model,
        tokenizer=tokenizer,
        system=system,
        user=ex.prompt,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    t1 = time.perf_counter()

    valid, correct = score_example(ex, out)
    status = "OK" if (valid and correct) else ("INV" if not valid else "WRONG")

    s = (out or "").strip()
    strict_valid = False
    strict_correct = False
    strict_note = ""
    if ex.answer_kind == "choice":
        strict_valid = bool(s) and len(s) == 1 and s.upper() in ("A", "B", "C", "D")
        strict_correct = bool(strict_valid and s.upper() == str(ex.answer))
        parsed = extract_choice(s) or ""
        strict_note = f"parsed={parsed or '-'}"
    elif ex.answer_kind == "json":
        try:
            obj = json.loads(s)
            strict_valid = isinstance(obj, dict)
            strict_correct = bool(strict_valid and all(obj.get(k) == v for k, v in dict(ex.answer).items()))
        except Exception:
            strict_valid = False
            strict_correct = False
        strict_note = "strict=json.loads(full_text)"
    elif ex.answer_kind == "int_list":
        # Expected format for sort bench: comma-separated integers.
        try:
            parts = [p.strip() for p in s.split(",")]
            vals = [int(p) for p in parts if p != ""]
            strict_valid = len(vals) == len(ex.answer) and "," in s and all(p.strip() and p.strip().lstrip("-").isdigit() for p in parts)
            strict_correct = bool(strict_valid and vals == list(ex.answer))
        except Exception:
            strict_valid = False
            strict_correct = False
        strict_note = "strict=comma_ints"
    elif ex.answer_kind == "int":
        strict_valid = bool(s) and (s.lstrip("-").isdigit())
        strict_correct = bool(strict_valid and int(s) == int(ex.answer))
    elif ex.answer_kind == "string":
        strict_valid = bool(s)
        strict_correct = bool(strict_valid and s == str(ex.answer))
    strict_status = "OK" if strict_correct else ("INV" if not strict_valid else "WRONG")
    print(_fmt_sep("-", 72))
    print(
        f"[{ex.suite}/{ex.category}] {ex.id}  "
        f"soft={status} strict={strict_status}  {strict_note}  time_ms={(t1 - t0) * 1000:.1f}"
    )
    print(_fmt_sep("-", 72))
    print("PROMPT:")
    print(ex.prompt)
    print()
    print("MODEL:")
    print(out if out else "<empty>")
    print()
    print("EXPECTED:")
    if ex.answer_kind == "json":
        print(json.dumps(ex.answer, ensure_ascii=False, indent=2))
    else:
        print(str(ex.answer))


def _print_open_ended(
    *,
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    demo: DemoExample,
    max_new_tokens: int,
    max_seq_len: Optional[int],
    seed: int,
) -> None:
    system = "你是一个乐于助人的助手。"
    t0 = time.perf_counter()
    out = _chat_generate(
        model=model,
        tokenizer=tokenizer,
        system=system,
        user=demo.prompt,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    t1 = time.perf_counter()
    print(_fmt_sep("=", 72))
    print(f"[chat] {demo.title}  time_ms={(t1 - t0) * 1000:.1f}")
    print(_fmt_sep("=", 72))
    print("PROMPT:")
    print(demo.prompt)
    print()
    print("MODEL:")
    print(out if out else "<empty>")

def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    t = _normalize(text)
    return any(_normalize(n) in t for n in needles if _normalize(n))


def _iter_knowledge_prompts() -> List[DemoExample]:
    # Focus on "knowledge Q&A" style prompts (not MCQ).
    return [
        DemoExample(
            title="地理常识",
            prompt="中国的首都是哪里？请只回答地名。",
            expected_contains=["北京"],
        ),
        DemoExample(
            title="地理常识",
            prompt="日本的首都是哪里？请只回答地名。",
            expected_contains=["东京"],
        ),
        DemoExample(
            title="科学常识",
            prompt="水的化学式是什么？请只输出化学式。",
            expected_contains=["h2o"],
        ),
        DemoExample(
            title="科学常识",
            prompt="二氧化碳的化学式是什么？请只输出化学式。",
            expected_contains=["co2"],
        ),
        DemoExample(
            title="计算机常识",
            prompt="HTTP 默认端口是多少？请只输出数字。",
            expected_contains=["80"],
        ),
        DemoExample(
            title="计算机常识",
            prompt="HTTPS 默认端口是多少？请只输出数字。",
            expected_contains=["443"],
        ),
        DemoExample(
            title="概念解释",
            prompt="用一句话解释什么是 JSON。",
            expected_contains=["javascript", "对象", "notation", "数据", "格式"],
        ),
        DemoExample(
            title="历史常识",
            prompt="中华人民共和国成立于哪一年？请只输出年份（四位数字）。",
            expected_contains=["1949"],
        ),
        DemoExample(
            title="语文常识",
            prompt="《红楼梦》的作者是谁？请只输出姓名。",
            expected_contains=["曹雪芹"],
        ),
        DemoExample(
            title="数学常识",
            prompt="圆周率 π 通常近似写作多少？请只输出数字。",
            expected_contains=["3.14"],
        ),
        DemoExample(
            title="单位常识",
            prompt="1 打（dozen）等于多少？请只输出数字。",
            expected_contains=["12"],
        ),
        DemoExample(
            title="推理+表达",
            prompt="用不超过 30 字解释：为什么天空看起来是蓝色的？",
            expected_contains=["散射", "瑞利", "短波"],
        ),
    ]


def _print_knowledge_demo(
    *,
    model: MiniLLMForCausalLM,
    tokenizer: Any,
    prompts: Iterable[DemoExample],
    max_new_tokens: int,
    max_seq_len: Optional[int],
    seed: int,
) -> None:
    system = "你是一个知识问答助手。尽量简洁、直接回答问题。"
    for i, demo in enumerate(prompts, start=1):
        t0 = time.perf_counter()
        out = _chat_generate(
            model=model,
            tokenizer=tokenizer,
            system=system,
            user=demo.prompt,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            seed=seed + i,
        )
        t1 = time.perf_counter()
        ok = None
        if demo.expected_contains:
            ok = _contains_any(out, demo.expected_contains)

        print(_fmt_sep("=", 72))
        status = "" if ok is None else ("OK" if ok else "WRONG?")
        print(f"[knowledge] {demo.title}  {status}  time_ms={(t1 - t0) * 1000:.1f}")
        print(_fmt_sep("=", 72))
        print("Q:")
        print(demo.prompt)
        print()
        print("A:")
        print(out if out else "<empty>")
        if demo.expected_contains:
            print()
            print(f"EXPECTED contains: {demo.expected_contains}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer demo: show multiple representative prompts + model outputs")
    parser.add_argument("--out_dir", type=str, default="out/mlx")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help=(
            "Max total tokens (prompt + generation). "
            "Defaults to checkpoint seq_len when available; otherwise 512."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="knowledge",
        choices=["knowledge", "bench"],
        help="Demo mode: knowledge (open Q&A) or bench (scored synthetic tasks).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="copy,json,sort,math_mcq,logic,qa,knowledge",
        help="[bench mode] Comma-separated: copy,json,sort,math_mcq,logic,qa,knowledge.",
    )
    parser.add_argument("--n", type=int, default=2, help="Examples per suite (default: 2).")
    parser.add_argument("--sort_list_len", type=int, default=8)
    parser.add_argument("--copy_length", type=int, default=24)
    parser.add_argument("--no_chat", action="store_true", help="Skip the open-ended chat demo prompt.")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Failed to import `transformers`. Install MLX training deps via "
            "`python3 -m pip install -r mlx_train/requirements.txt`."
        ) from e

    ckpt = _resolve_checkpoint(args.checkpoint, out_dir=args.out_dir)
    cfg_dict = _load_config(ckpt)
    cfg = MiniLLMConfig(**cfg_dict).finalize()
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(ckpt / "model.safetensors"))
    model.eval()
    if int(cfg.lora_r) > 0:
        merge_lora(model)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    seed = int(args.seed)
    max_seq_len: Optional[int] = args.max_seq_len
    if max_seq_len is None:
        max_seq_len = load_state_seq_len(ckpt)
        if max_seq_len is None:
            max_seq_len = 512
    if max_seq_len is not None:
        max_seq_len = min(int(max_seq_len), int(cfg.max_position_embeddings))

    print(_fmt_sep("=", 72))
    print("MiniLLM Infer Demo (MLX)")
    print(_fmt_sep("=", 72))
    print(f"ckpt: {ckpt}")
    print(
        f"mode: {args.mode}  max_new_tokens={int(args.max_new_tokens)}  "
        f"max_seq_len={max_seq_len if max_seq_len is not None else 'none'}"
    )
    print()

    if str(args.mode) == "knowledge":
        prompts = _iter_knowledge_prompts()
        _print_knowledge_demo(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=int(args.max_new_tokens),
            max_seq_len=max_seq_len,
            seed=seed,
        )
        return

    # bench mode (original behavior)
    suites = [s.strip() for s in str(args.suite).split(",") if s.strip()]
    suites = list(dict.fromkeys(suites))
    n = int(args.n)
    print(f"suites: {', '.join(suites)}  n_per_suite={n}")
    print()

    examples: List[Example] = []
    if "math_mcq" in suites:
        k = max(1, n)
        examples.extend(build_math_mcq(seed=seed + 0, n_add=k, n_sub=k, n_mul=k, n_div=k)[:n])
    if "qa" in suites:
        examples.extend(build_qa_bench(seed=seed + 1, n=n)[:n])
    if "logic" in suites:
        examples.extend(build_logic_bench(seed=seed + 2, n=n)[:n])
    if "knowledge" in suites:
        examples.extend(build_knowledge_bench(seed=seed + 3, n=n)[:n])
    if "sort" in suites:
        examples.extend(build_sort_bench(seed=seed + 4, n=n, list_len=int(args.sort_list_len))[:n])
    if "json" in suites:
        examples.extend(build_json_bench(seed=seed + 5, n=n)[:n])
    if "copy" in suites:
        examples.extend(build_copy_bench(seed=seed + 6, n=n, length=int(args.copy_length))[:n])

    if not args.no_chat:
        _print_open_ended(
            model=model,
            tokenizer=tokenizer,
            demo=DemoExample(
                title="Instruction following + formatting",
                prompt="用 5 条要点解释什么是 Transformer 的 attention。每条不超过 20 个字。",
            ),
            max_new_tokens=int(args.max_new_tokens),
            max_seq_len=max_seq_len,
            seed=seed,
        )
        print()

    for ex in examples:
        _print_scored_example(
            model=model,
            tokenizer=tokenizer,
            ex=ex,
            max_new_tokens=int(args.max_new_tokens),
            max_seq_len=max_seq_len,
            seed=seed,
        )
        print()


if __name__ == "__main__":
    main()
