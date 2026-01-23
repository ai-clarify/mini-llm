from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import mlx.core as mx

from .infer import generate
from ..config import MiniLLMConfig
from ..models import MiniLLMForCausalLM
from ..nn.lora import merge_lora
from ..stats import _find_latest_checkpoint as _find_latest_checkpoint_impl


def _load_config(checkpoint_dir: Path) -> Dict[str, Any]:
    path = checkpoint_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint(checkpoint: Optional[str], *, out_dir: str) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        if p.is_file() and p.name.endswith(".safetensors"):
            p = p.parent
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    latest = _find_latest_checkpoint_impl(Path(out_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found under {out_dir} (expected out/mlx/(sft|pretrain)/checkpoints)")
    return latest


ANSWER_RE = re.compile(r"(?<![A-Za-z0-9])[ABCD](?![A-Za-z0-9])", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def extract_choice(text: str) -> Optional[str]:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    return m.group(0).upper()


@dataclass(frozen=True)
class Example:
    id: str
    suite: str
    category: str
    prompt: str
    answer: Any
    answer_kind: str  # choice|int|int_list|json|string
    choices: Optional[Dict[str, str]] = None


def _mcq_prompt(question: str, choices: Dict[str, str]) -> str:
    lines = [
        question.strip(),
        "",
        f"A. {choices['A']}",
        f"B. {choices['B']}",
        f"C. {choices['C']}",
        f"D. {choices['D']}",
        "",
        "请只输出正确选项的字母（A/B/C/D），不要输出其它内容。",
    ]
    return "\n".join(lines)


def _unique_distractors(rng, *, correct: int, make_candidate) -> List[int]:
    values = {int(correct)}
    out: List[int] = []
    while len(out) < 3:
        cand = int(make_candidate())
        if cand < 0 or cand in values:
            continue
        values.add(cand)
        out.append(cand)
    return out


def build_math_mcq(
    *,
    seed: int,
    n_add: int,
    n_sub: int,
    n_mul: int,
    n_div: int,
) -> List[Example]:
    import random

    rng = random.Random(seed)
    examples: List[Example] = []
    labels = ["A", "B", "C", "D"]

    def add_one(i: int) -> None:
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        ans = a + b

        distract = _unique_distractors(rng, correct=ans, make_candidate=lambda: ans + rng.randint(-99, 99))
        opts = [ans] + distract
        rng.shuffle(opts)
        choices = {k: str(v) for k, v in zip(labels, opts)}
        answer = labels[opts.index(ans)]
        prompt = _mcq_prompt(f"【加法】{a} + {b} = ?", choices)
        examples.append(
            Example(
                id=f"add_{i:04d}",
                suite="math_mcq",
                category="math_add",
                prompt=prompt,
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )

    def sub_one(i: int) -> None:
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        if b > a:
            a, b = b, a
        ans = a - b
        distract = _unique_distractors(rng, correct=ans, make_candidate=lambda: ans + rng.randint(-99, 99))
        opts = [ans] + distract
        rng.shuffle(opts)
        choices = {k: str(v) for k, v in zip(labels, opts)}
        answer = labels[opts.index(ans)]
        prompt = _mcq_prompt(f"【减法】{a} - {b} = ?", choices)
        examples.append(
            Example(
                id=f"sub_{i:04d}",
                suite="math_mcq",
                category="math_sub",
                prompt=prompt,
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )

    def mul_one(i: int) -> None:
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        ans = a * b
        distract = _unique_distractors(rng, correct=ans, make_candidate=lambda: ans + rng.randint(-200, 200))
        opts = [ans] + distract
        rng.shuffle(opts)
        choices = {k: str(v) for k, v in zip(labels, opts)}
        answer = labels[opts.index(ans)]
        prompt = _mcq_prompt(f"【乘法】{a} × {b} = ?", choices)
        examples.append(
            Example(
                id=f"mul_{i:04d}",
                suite="math_mcq",
                category="math_mul",
                prompt=prompt,
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )

    def div_one(i: int) -> None:
        b = rng.randint(2, 99)
        q = rng.randint(2, 99)
        a = b * q
        ans = q
        distract = _unique_distractors(rng, correct=ans, make_candidate=lambda: ans + rng.randint(-30, 30))
        opts = [ans] + distract
        rng.shuffle(opts)
        choices = {k: str(v) for k, v in zip(labels, opts)}
        answer = labels[opts.index(ans)]
        prompt = _mcq_prompt(f"【除法】{a} ÷ {b} = ?", choices)
        examples.append(
            Example(
                id=f"div_{i:04d}",
                suite="math_mcq",
                category="math_div",
                prompt=prompt,
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )

    for i in range(n_add):
        add_one(i)
    for i in range(n_sub):
        sub_one(i)
    for i in range(n_mul):
        mul_one(i)
    for i in range(n_div):
        div_one(i)

    rng.shuffle(examples)
    return examples


def build_sort_bench(*, seed: int, n: int, list_len: int = 8) -> List[Example]:
    import random

    rng = random.Random(seed)
    out: List[Example] = []
    for i in range(int(n)):
        values = [rng.randint(-999, 999) for _ in range(int(list_len))]
        answer = sorted(values)
        prompt = (
            "将以下整数从小到大排序，并用英文逗号分隔输出。\n\n"
            f"输入：{', '.join(str(x) for x in values)}\n\n"
            "只输出排序后的结果，不要输出其它内容。"
        )
        out.append(
            Example(
                id=f"sort_{i:04d}",
                suite="sort",
                category="sort_int",
                prompt=prompt,
                answer=answer,
                answer_kind="int_list",
            )
        )
    return out


def build_json_bench(*, seed: int, n: int) -> List[Example]:
    import random

    rng = random.Random(seed)
    out: List[Example] = []
    for i in range(int(n)):
        a = rng.randint(-999, 999)
        b = rng.randint(-999, 999)
        answer = {"sum": a + b, "diff": a - b, "prod": a * b}
        prompt = (
            "给定两个整数 a 和 b，请只输出一个 JSON 对象（不要用代码块），包含三个键：sum, diff, prod。\n"
            "其中 sum=a+b, diff=a-b, prod=a*b。\n\n"
            f"a={a}\n"
            f"b={b}\n\n"
            "只输出 JSON，不要输出其它内容。"
        )
        out.append(
            Example(
                id=f"json_{i:04d}",
                suite="json",
                category="json_math",
                prompt=prompt,
                answer=answer,
                answer_kind="json",
            )
        )
    return out


def build_copy_bench(*, seed: int, n: int, length: int = 24) -> List[Example]:
    import random
    import string

    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + "_"
    out: List[Example] = []
    for i in range(int(n)):
        s = "".join(rng.choice(alphabet) for _ in range(int(length)))
        prompt = f"请原样复制下面这段字符串（完全一致），只输出该字符串，不要输出其它内容：\n\n{s}"
        out.append(
            Example(
                id=f"copy_{i:04d}",
                suite="copy",
                category="copy_exact",
                prompt=prompt,
                answer=s,
                answer_kind="string",
            )
        )
    return out


def build_qa_bench(*, seed: int, n: int) -> List[Example]:
    import random

    rng = random.Random(seed)
    labels = ["A", "B", "C", "D"]

    names_pool = [
        "小明",
        "小红",
        "小刚",
        "小丽",
        "阿强",
        "阿珍",
        "小宇",
        "小雨",
        "小鹏",
        "小雪",
        "小哲",
        "小雅",
        "小辰",
        "小然",
        "小凡",
        "小安",
        "小禾",
        "小北",
        "小南",
        "小东",
        "小西",
    ]
    cities = ["北京", "上海", "广州", "深圳", "成都", "杭州", "南京", "苏州", "武汉", "西安", "重庆", "长沙"]
    hobbies = ["篮球", "羽毛球", "阅读", "摄影", "跑步", "绘画", "写作", "音乐", "旅行", "编程", "做饭", "围棋"]
    pets = ["猫", "狗", "兔子", "鹦鹉", "仓鼠", "乌龟", "金鱼", "蜥蜴"]

    out: List[Example] = []
    for i in range(int(n)):
        persons = rng.sample(names_pool, 4)
        person_city = {p: c for p, c in zip(persons, rng.sample(cities, 4))}
        person_hobby = {p: h for p, h in zip(persons, rng.sample(hobbies, 4))}
        person_pet = {p: t for p, t in zip(persons, rng.sample(pets, 4))}

        passage_lines = []
        for p in persons:
            passage_lines.append(f"{p}住在{person_city[p]}，喜欢{person_hobby[p]}，养{person_pet[p]}。")
        passage = "\n".join(passage_lines)

        qtype = rng.choice(
            [
                "who_city",
                "who_hobby",
                "who_pet",
                "city_of",
                "hobby_of",
                "pet_of",
            ]
        )

        if qtype == "who_city":
            city = rng.choice([person_city[p] for p in persons])
            correct = [p for p in persons if person_city[p] == city][0]
            opts = list(persons)
            rng.shuffle(opts)
            choices = {k: v for k, v in zip(labels, opts)}
            answer = labels[opts.index(correct)]
            question = f"阅读短文并回答问题：\n\n{passage}\n\n问题：谁住在{city}？"
            out.append(
                Example(
                    id=f"qa_{i:04d}",
                    suite="qa",
                    category="qa_who_city",
                    prompt=_mcq_prompt(question, choices),
                    answer=answer,
                    answer_kind="choice",
                    choices=choices,
                )
            )
            continue

        if qtype == "who_hobby":
            hobby = rng.choice([person_hobby[p] for p in persons])
            correct = [p for p in persons if person_hobby[p] == hobby][0]
            opts = list(persons)
            rng.shuffle(opts)
            choices = {k: v for k, v in zip(labels, opts)}
            answer = labels[opts.index(correct)]
            question = f"阅读短文并回答问题：\n\n{passage}\n\n问题：谁喜欢{hobby}？"
            out.append(
                Example(
                    id=f"qa_{i:04d}",
                    suite="qa",
                    category="qa_who_hobby",
                    prompt=_mcq_prompt(question, choices),
                    answer=answer,
                    answer_kind="choice",
                    choices=choices,
                )
            )
            continue

        if qtype == "who_pet":
            pet = rng.choice([person_pet[p] for p in persons])
            correct = [p for p in persons if person_pet[p] == pet][0]
            opts = list(persons)
            rng.shuffle(opts)
            choices = {k: v for k, v in zip(labels, opts)}
            answer = labels[opts.index(correct)]
            question = f"阅读短文并回答问题：\n\n{passage}\n\n问题：谁养{pet}？"
            out.append(
                Example(
                    id=f"qa_{i:04d}",
                    suite="qa",
                    category="qa_who_pet",
                    prompt=_mcq_prompt(question, choices),
                    answer=answer,
                    answer_kind="choice",
                    choices=choices,
                )
            )
            continue

        person = rng.choice(persons)
        if qtype == "city_of":
            correct = person_city[person]
            opts = [person_city[p] for p in persons]
            rng.shuffle(opts)
            choices = {k: v for k, v in zip(labels, opts)}
            answer = labels[opts.index(correct)]
            question = f"阅读短文并回答问题：\n\n{passage}\n\n问题：{person}住在哪个城市？"
            out.append(
                Example(
                    id=f"qa_{i:04d}",
                    suite="qa",
                    category="qa_city_of",
                    prompt=_mcq_prompt(question, choices),
                    answer=answer,
                    answer_kind="choice",
                    choices=choices,
                )
            )
            continue

        if qtype == "hobby_of":
            correct = person_hobby[person]
            opts = [person_hobby[p] for p in persons]
            rng.shuffle(opts)
            choices = {k: v for k, v in zip(labels, opts)}
            answer = labels[opts.index(correct)]
            question = f"阅读短文并回答问题：\n\n{passage}\n\n问题：{person}喜欢什么？"
            out.append(
                Example(
                    id=f"qa_{i:04d}",
                    suite="qa",
                    category="qa_hobby_of",
                    prompt=_mcq_prompt(question, choices),
                    answer=answer,
                    answer_kind="choice",
                    choices=choices,
                )
            )
            continue

        correct = person_pet[person]
        opts = [person_pet[p] for p in persons]
        rng.shuffle(opts)
        choices = {k: v for k, v in zip(labels, opts)}
        answer = labels[opts.index(correct)]
        question = f"阅读短文并回答问题：\n\n{passage}\n\n问题：{person}养什么宠物？"
        out.append(
            Example(
                id=f"qa_{i:04d}",
                suite="qa",
                category="qa_pet_of",
                prompt=_mcq_prompt(question, choices),
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )
    return out


def build_logic_bench(*, seed: int, n: int) -> List[Example]:
    import random

    rng = random.Random(seed)
    labels = ["A", "B", "C", "D"]

    names_pool = [
        "小明",
        "小红",
        "小刚",
        "小丽",
        "阿强",
        "阿珍",
        "小宇",
        "小雨",
        "小鹏",
        "小雪",
        "小哲",
        "小雅",
        "小辰",
        "小然",
    ]

    out: List[Example] = []
    n = int(n)
    n_bool = n // 2
    n_order = n - n_bool

    def add_bool(i: int) -> None:
        p = bool(rng.getrandbits(1))
        q = bool(rng.getrandbits(1))
        r = bool(rng.getrandbits(1))
        templates: List[Tuple[str, Any]] = [
            ("(p and q) or (not r)", lambda p, q, r: (p and q) or (not r)),
            ("(p or q) and (q or r)", lambda p, q, r: (p or q) and (q or r)),
            ("(not p) or (q and r)", lambda p, q, r: (not p) or (q and r)),
            ("(p == q) != r", lambda p, q, r: (p == q) != r),
            ("(p != q) and (q != r)", lambda p, q, r: (p != q) and (q != r)),
            ("(p and (not q)) or r", lambda p, q, r: (p and (not q)) or r),
        ]
        expr, fn = rng.choice(templates)
        value = bool(fn(p, q, r))

        # MCQ options: True/False + two distractors.
        choices_vals = ["True", "False", "无法确定", "以上都不对"]
        correct_val = "True" if value else "False"
        choices = {k: v for k, v in zip(labels, choices_vals)}
        answer = "A" if correct_val == "True" else "B"

        question = (
            "请根据给定的布尔变量取值，计算表达式的结果。\n\n"
            f"p={p}, q={q}, r={r}\n"
            f"表达式：{expr}\n\n"
            "结果是什么？"
        )
        out.append(
            Example(
                id=f"logic_bool_{i:04d}",
                suite="logic",
                category="logic_bool",
                prompt=_mcq_prompt(question, choices),
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )

    def add_order(i: int) -> None:
        persons = rng.sample(names_pool, 4)
        order = list(persons)
        rng.shuffle(order)
        p1, p2, p3, p4 = order
        stmts = [
            f"1) {p1}在{p2}的左边。",
            f"2) {p2}在{p3}的左边。",
            f"3) {p3}在{p4}的左边。",
        ]
        ask = rng.choice(["leftmost", "rightmost", "second", "third"])
        if ask == "leftmost":
            correct = p1
            question = "谁在最左边？"
            cat = "logic_order_leftmost"
        elif ask == "rightmost":
            correct = p4
            question = "谁在最右边？"
            cat = "logic_order_rightmost"
        elif ask == "second":
            correct = p2
            question = "谁在从左数第二个位置？"
            cat = "logic_order_second"
        else:
            correct = p3
            question = "谁在从左数第三个位置？"
            cat = "logic_order_third"

        opts = list(persons)
        rng.shuffle(opts)
        choices = {k: v for k, v in zip(labels, opts)}
        answer = labels[opts.index(correct)]
        body = (
            "有四个人站成一排，从左到右有四个位置。\n"
            "已知：\n" + "\n".join(stmts) + f"\n\n问题：{question}"
        )
        out.append(
            Example(
                id=f"logic_order_{i:04d}",
                suite="logic",
                category=cat,
                prompt=_mcq_prompt(body, choices),
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )

    for i in range(n_bool):
        add_bool(i)
    for i in range(n_order):
        add_order(i)

    rng.shuffle(out)
    return out


def build_knowledge_bench(*, seed: int, n: int) -> List[Example]:
    import random

    rng = random.Random(seed)
    labels = ["A", "B", "C", "D"]
    items: List[Tuple[str, str, str, List[str]]] = [
        ("knowledge_geo", "中国的首都是？", "北京", ["上海", "广州", "深圳"]),
        ("knowledge_geo", "日本的首都是？", "东京", ["大阪", "京都", "名古屋"]),
        ("knowledge_geo", "法国的首都是？", "巴黎", ["伦敦", "柏林", "罗马"]),
        ("knowledge_geo", "美国的首都是？", "华盛顿", ["纽约", "洛杉矶", "芝加哥"]),
        ("knowledge_geo", "俄罗斯的首都是？", "莫斯科", ["圣彼得堡", "基辅", "明斯克"]),
        ("knowledge_geo", "世界上面积最大的海洋是？", "太平洋", ["大西洋", "印度洋", "北冰洋"]),
        ("knowledge_geo", "世界最高峰是？", "珠穆朗玛峰", ["乔戈里峰", "干城章嘉峰", "洛子峰"]),
        ("knowledge_geo", "撒哈拉沙漠位于哪个大洲？", "非洲", ["亚洲", "欧洲", "南美洲"]),
        ("knowledge_geo", "澳大利亚所在的大洲通常称为？", "大洋洲", ["欧洲", "南极洲", "非洲"]),
        ("knowledge_science", "水的化学式是？", "H2O", ["CO2", "NaCl", "O2"]),
        ("knowledge_science", "二氧化碳的化学式是？", "CO2", ["CO", "H2O", "CH4"]),
        ("knowledge_science", "食盐（氯化钠）的化学式是？", "NaCl", ["KCl", "Na2CO3", "HCl"]),
        ("knowledge_science", "太阳系中最大的行星是？", "木星", ["地球", "火星", "金星"]),
        ("knowledge_science", "地球的天然卫星是？", "月球", ["太阳", "火星", "金星"]),
        ("knowledge_science", "人体最大的器官是？", "皮肤", ["心脏", "肝脏", "大脑"]),
        ("knowledge_science", "常见的光合作用主要发生在植物的哪个结构？", "叶绿体", ["细胞核", "线粒体", "细胞壁"]),
        ("knowledge_science", "声音在空气中传播需要？", "介质", ["真空", "重力", "磁场"]),
        ("knowledge_history", "中华人民共和国成立于哪一年？", "1949年", ["1911年", "1937年", "1978年"]),
        ("knowledge_history", "中国古代四大发明之一“造纸术”的改进者常认为是？", "蔡伦", ["张衡", "祖冲之", "毕昇"]),
        ("knowledge_history", "《史记》的作者是？", "司马迁", ["司马光", "班固", "陈寿"]),
        ("knowledge_culture", "《红楼梦》的作者是？", "曹雪芹", ["鲁迅", "吴承恩", "施耐庵"]),
        ("knowledge_culture", "《西游记》的作者通常认为是？", "吴承恩", ["曹雪芹", "罗贯中", "施耐庵"]),
        ("knowledge_culture", "《三国演义》的作者通常认为是？", "罗贯中", ["吴承恩", "施耐庵", "曹雪芹"]),
        ("knowledge_culture", "《水浒传》的作者通常认为是？", "施耐庵", ["罗贯中", "吴承恩", "曹雪芹"]),
        ("knowledge_math", "圆周率π的近似值通常写作？", "3.14", ["2.71", "1.62", "0.58"]),
        ("knowledge_math", "一个三角形的内角和是？", "180°", ["90°", "360°", "270°"]),
        ("knowledge_cs", "HTTP 的默认端口通常是？", "80", ["22", "443", "3306"]),
        ("knowledge_cs", "HTTPS 的默认端口通常是？", "443", ["80", "21", "25"]),
        ("knowledge_cs", "JSON 是什么的缩写？", "JavaScript Object Notation", ["Java Source Object Name", "Joint Service Orchestration Network", "Java Syntax Object Node"]),
        ("knowledge_general", "一打（dozen）等于多少？", "12", ["10", "11", "20"]),
        ("knowledge_general", "“公斤”是质量单位，其符号是？", "kg", ["g", "lb", "m"]),
    ]

    out: List[Example] = []
    pool = list(items)
    rng.shuffle(pool)
    n = int(n)
    for i in range(n):
        cat, q, correct_text, distractors = pool[i % len(pool)]
        opts = [correct_text] + list(distractors)
        rng.shuffle(opts)
        choices = {k: v for k, v in zip(labels, opts)}
        answer = labels[opts.index(correct_text)]
        prompt = _mcq_prompt(q, choices)
        out.append(
            Example(
                id=f"knowledge_{i:04d}",
                suite="knowledge",
                category=cat,
                prompt=prompt,
                answer=answer,
                answer_kind="choice",
                choices=choices,
            )
        )
    return out


def _extract_int(text: str) -> Optional[int]:
    m = INT_RE.findall(text or "")
    if not m:
        return None
    try:
        return int(m[-1])
    except ValueError:
        return None


def _extract_int_list(text: str) -> Optional[List[int]]:
    m = INT_RE.findall(text or "")
    if not m:
        return None
    out: List[int] = []
    for tok in m:
        try:
            out.append(int(tok))
        except ValueError:
            return None
    return out


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if "```" in s:
        # Try to extract fenced JSON content.
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            if s.lstrip().startswith("json"):
                s = s.lstrip()[4:]
            s = s.strip()

    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    blob = s[start : end + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _normalize_string(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip()


def score_example(ex: Example, text: str) -> Tuple[bool, bool]:
    s = (text or "").strip()
    if ex.answer_kind == "choice":
        pred = extract_choice(s)
        if pred is None and ex.choices:
            num = _extract_int(s)
            if num is not None:
                for k, v in ex.choices.items():
                    if v == str(num):
                        pred = k
                        break
        if pred is None and ex.choices:
            norm_out = _normalize_string(s)
            exact = [k for k, v in ex.choices.items() if _normalize_string(v) == norm_out]
            if len(exact) == 1:
                pred = exact[0]
            else:
                contains = [k for k, v in ex.choices.items() if _normalize_string(v) and _normalize_string(v) in norm_out]
                if len(contains) == 1:
                    pred = contains[0]
        valid = pred is not None
        return valid, bool(valid and pred == ex.answer)

    if ex.answer_kind == "int":
        pred = _extract_int(s)
        valid = pred is not None
        return valid, bool(valid and int(pred) == int(ex.answer))

    if ex.answer_kind == "int_list":
        pred_all = _extract_int_list(s)
        expected = list(ex.answer)
        if pred_all is None:
            return False, False
        if len(pred_all) < len(expected):
            return False, False
        pred = pred_all[-len(expected) :] if expected else []
        return True, bool(pred == expected)

    if ex.answer_kind == "json":
        pred = _extract_json(s)
        valid = pred is not None
        if not valid:
            return False, False
        expected = dict(ex.answer)
        for k, v in expected.items():
            if pred.get(k) != v:
                return True, False
        return True, True

    if ex.answer_kind == "string":
        pred = _normalize_string(s)
        valid = bool(pred)
        return valid, bool(pred == str(ex.answer))

    return False, False


@dataclass
class RunStats:
    name: str
    correct: int = 0
    invalid: int = 0
    total: int = 0
    gen_tokens: int = 0
    gen_time_s: float = 0.0
    wall_time_s: float = 0.0
    per_suite_total: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_suite_correct: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_suite_invalid: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_cat_total: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_cat_correct: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_cat_invalid: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(
        self,
        *,
        suite: str,
        category: str,
        correct: bool,
        valid: bool,
        gen_tokens: int,
        gen_time_s: float,
        wall_time_s: float,
    ) -> None:
        self.total += 1
        self.per_suite_total[suite] += 1
        self.per_cat_total[category] += 1
        if not valid:
            self.invalid += 1
            self.per_suite_invalid[suite] += 1
            self.per_cat_invalid[category] += 1
        if correct:
            self.correct += 1
            self.per_suite_correct[suite] += 1
            self.per_cat_correct[category] += 1
        self.gen_tokens += int(gen_tokens)
        self.gen_time_s += float(gen_time_s)
        self.wall_time_s += float(wall_time_s)

    def acc(self) -> float:
        return (self.correct / self.total) if self.total else 0.0

    def tok_s(self) -> float:
        return (self.gen_tokens / self.gen_time_s) if self.gen_time_s > 0 else 0.0

    def ex_s(self) -> float:
        return (self.total / self.wall_time_s) if self.wall_time_s > 0 else 0.0

    def avg_ms(self) -> float:
        return (1000.0 * self.wall_time_s / self.total) if self.total else 0.0


def _progress_line(name: str, stats: RunStats, done: int, total: int, started: float) -> str:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    return (
        f"[bench] {name} {done}/{total} "
        f"acc={stats.acc()*100:.1f}% invalid={stats.invalid} "
        f"tok/s={stats.tok_s():.0f} ex/s={rate:.2f} eta={eta:.0f}s"
    )


def _system_prompt_for(ex: Example) -> str:
    if ex.answer_kind == "choice":
        return "你是一个选择题解答器。请严格只输出最终答案的选项字母（A/B/C/D），不要输出其它内容。"
    if ex.answer_kind == "json":
        return "你是一个 JSON 生成器。请严格只输出 JSON 对象（不要代码块/解释/多余文本）。"
    if ex.answer_kind in ("int", "int_list"):
        return "你是一个严谨的计算助手。请严格按要求只输出结果，不要解释。"
    if ex.answer_kind == "string":
        return "你是一个复制器。请严格只输出目标字符串本身，不要输出其它内容。"
    return "你是一个严谨的助手。请严格按要求输出，不要解释。"


def run_mlx(
    *,
    checkpoint_dir: Path,
    tokenizer_path: str,
    tokenizer_type: str,
    examples: List[Example],
    max_new_tokens: int,
    seed: int,
    progress: bool,
    progress_every: int,
) -> RunStats:
    from .tokenizer_utils import load_tokenizer

    mx.random.seed(seed)

    cfg_dict = _load_config(checkpoint_dir)
    cfg = MiniLLMConfig(**cfg_dict).finalize()
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(str(checkpoint_dir / "model.safetensors"))
    model.eval()
    if int(cfg.lora_r) > 0:
        merge_lora(model)

    tokenizer = load_tokenizer(
        tokenizer_path,
        tokenizer_type=str(tokenizer_type),
    )

    stats = RunStats(name="mini_llm")
    started = time.perf_counter()
    total = len(examples)
    for i, ex in enumerate(examples, start=1):
        t_wall0 = time.perf_counter()

        system = _system_prompt_for(ex)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": ex.prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids: List[int] = tokenizer.encode(prompt_text, add_special_tokens=False)

        t_gen0 = time.perf_counter()
        out_ids = generate(
            model,
            input_ids=prompt_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            min_new_tokens=0,
            banned_token_ids=list(tokenizer.all_special_ids),
            temperature=0.0,
            top_p=1.0,
            max_seq_len=None,
        )
        t_gen1 = time.perf_counter()

        resp_ids = out_ids[len(prompt_ids) :]
        text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
        valid, correct = score_example(ex, text)

        t_wall1 = time.perf_counter()
        stats.add(
            suite=ex.suite,
            category=ex.category,
            correct=correct,
            valid=valid,
            gen_tokens=len(resp_ids),
            gen_time_s=t_gen1 - t_gen0,
            wall_time_s=t_wall1 - t_wall0,
        )

        if progress and (i % max(int(progress_every), 1) == 0 or i == total):
            print(_progress_line(stats.name, stats, i, total, started), flush=True)
    return stats


def ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    seed: int,
    think: bool,
) -> Tuple[str, Dict[str, Any]]:
    import requests

    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": bool(think),
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": int(max_new_tokens),
            "seed": int(seed),
        },
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    thinking = msg.get("thinking")
    text = content if isinstance(content, str) and content.strip() else ""
    if not text and isinstance(thinking, str) and thinking.strip():
        text = thinking
    if not text:
        text = data.get("response", "") or ""
    return str(text), data


def run_ollama(
    *,
    base_url: str,
    model: str,
    examples: List[Example],
    max_new_tokens: int,
    seed: int,
    think: bool,
    progress: bool,
    progress_every: int,
) -> RunStats:
    stats = RunStats(name=model)

    started = time.perf_counter()
    total = len(examples)
    for i, ex in enumerate(examples, start=1):
        t_wall0 = time.perf_counter()

        system = _system_prompt_for(ex)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": ex.prompt}]
        text, meta = ollama_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            seed=seed,
            think=think,
        )
        valid, correct = score_example(ex, str(text))

        eval_count = int(meta.get("eval_count") or 0)
        eval_duration_ns = int(meta.get("eval_duration") or 0)
        gen_time_s = (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
        t_wall1 = time.perf_counter()
        stats.add(
            suite=ex.suite,
            category=ex.category,
            correct=correct,
            valid=valid,
            gen_tokens=eval_count,
            gen_time_s=gen_time_s,
            wall_time_s=t_wall1 - t_wall0,
        )

        if progress and (i % max(int(progress_every), 1) == 0 or i == total):
            print(_progress_line(stats.name, stats, i, total, started), flush=True)

    return stats


def _line(char: str = "-", n: int = 60) -> str:
    return char * n


def _kv(k: str, v: str, *, w: int = 14) -> str:
    return f"{k:<{w}}: {v}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench: MiniLLM (MLX) vs local Ollama (qwen3:0.6b)")
    parser.add_argument("--out_dir", type=str, default="out/mlx")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["auto", "hf", "rustbpe"],
        default="auto",
        help="Tokenizer backend: auto (prefer RustBPE), hf, or rustbpe.",
    )

    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_model", type=str, default="qwen3:0.6b")
    parser.add_argument(
        "--ollama_think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Control Qwen3 thinking mode (default: disabled for stable MCQ scoring).",
    )

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        help="Comma-separated: all|math_mcq|qa|logic|knowledge|sort|json|copy (default: all).",
    )
    parser.add_argument("--n_add", type=int, default=50, help="math_mcq: add examples")
    parser.add_argument("--n_sub", type=int, default=50, help="math_mcq: sub examples")
    parser.add_argument("--n_mul", type=int, default=50, help="math_mcq: mul examples")
    parser.add_argument("--n_div", type=int, default=50, help="math_mcq: div examples")
    parser.add_argument("--n_qa", type=int, default=50, help="qa: examples")
    parser.add_argument("--n_logic", type=int, default=50, help="logic: examples")
    parser.add_argument("--n_knowledge", type=int, default=50, help="knowledge: examples")
    parser.add_argument("--n_sort", type=int, default=50, help="sort: examples")
    parser.add_argument("--sort_list_len", type=int, default=8, help="sort: list length")
    parser.add_argument("--n_json", type=int, default=50, help="json: examples")
    parser.add_argument("--n_copy", type=int, default=50, help="copy: examples")
    parser.add_argument("--copy_length", type=int, default=24, help="copy: string length")

    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_ollama", action="store_true")
    parser.add_argument("--no_mlx", action="store_true")
    args = parser.parse_args()

    from .tokenizer_utils import load_tokenizer

    suites = [s.strip() for s in str(args.suite).split(",") if s.strip()]
    if "all" in suites:
        suites = ["math_mcq", "qa", "logic", "knowledge", "sort", "json", "copy"]
    suites = list(dict.fromkeys(suites))

    examples: List[Example] = []
    if "math_mcq" in suites:
        examples.extend(
            build_math_mcq(
                seed=int(args.seed) + 0,
                n_add=int(args.n_add),
                n_sub=int(args.n_sub),
                n_mul=int(args.n_mul),
                n_div=int(args.n_div),
            )
        )
    if "qa" in suites:
        examples.extend(build_qa_bench(seed=int(args.seed) + 1, n=int(args.n_qa)))
    if "logic" in suites:
        examples.extend(build_logic_bench(seed=int(args.seed) + 2, n=int(args.n_logic)))
    if "knowledge" in suites:
        examples.extend(build_knowledge_bench(seed=int(args.seed) + 3, n=int(args.n_knowledge)))
    if "sort" in suites:
        examples.extend(
            build_sort_bench(seed=int(args.seed) + 4, n=int(args.n_sort), list_len=int(args.sort_list_len))
        )
    if "json" in suites:
        examples.extend(build_json_bench(seed=int(args.seed) + 5, n=int(args.n_json)))
    if "copy" in suites:
        examples.extend(build_copy_bench(seed=int(args.seed) + 6, n=int(args.n_copy), length=int(args.copy_length)))

    if not examples:
        raise ValueError(f"No examples selected (suite={args.suite})")

    import random

    random.Random(int(args.seed)).shuffle(examples)

    ckpt: Optional[Path] = None
    if not args.no_mlx:
        ckpt = _resolve_checkpoint(args.checkpoint, out_dir=args.out_dir)

    print(_line("=", 60))
    print("MiniLLM Bench (synthetic)  |  MLX vs Ollama")
    print(_line("=", 60))
    suite_counts: Dict[str, int] = {}
    for ex in examples:
        suite_counts[ex.suite] = suite_counts.get(ex.suite, 0) + 1
    suite_desc = ", ".join(f"{k}={v}" for k, v in sorted(suite_counts.items()))
    print(_kv("dataset", f"synthetic total={len(examples)} seed={args.seed} suites=[{suite_desc}]"))
    if ckpt is not None:
        print(_kv("mlx_ckpt", str(ckpt)))
    if not args.no_ollama:
        print(_kv("ollama", f"{args.ollama_url}  model={args.ollama_model}"))
    print(_kv("decode", f"temperature=0 max_new_tokens={args.max_new_tokens} ollama_think={bool(args.ollama_think)}"))
    print(_kv("progress", f"every={args.progress_every} enabled={not args.no_progress}"))
    print()

    runs: List[RunStats] = []
    if not args.no_mlx:
        assert ckpt is not None
        runs.append(
            run_mlx(
                checkpoint_dir=ckpt,
                tokenizer_path=args.tokenizer_path,
                tokenizer_type=str(args.tokenizer_type),
                examples=examples,
                max_new_tokens=int(args.max_new_tokens),
                seed=int(args.seed),
                progress=not bool(args.no_progress),
                progress_every=int(args.progress_every),
            )
        )
    if not args.no_ollama:
        runs.append(
            run_ollama(
                base_url=args.ollama_url,
                model=args.ollama_model,
                examples=examples,
                max_new_tokens=int(args.max_new_tokens),
                seed=int(args.seed),
                think=bool(args.ollama_think),
                progress=not bool(args.no_progress),
                progress_every=int(args.progress_every),
            )
        )

    def fmt_run(r: RunStats) -> str:
        return (
            f"acc={r.acc()*100:5.1f}% ({r.correct}/{r.total}) "
            f"invalid={r.invalid:3d} "
            f"gen_tok/s={r.tok_s():6.0f} "
            f"ex/s={r.ex_s():5.2f} avg_ms={r.avg_ms():6.1f}"
        )

    print("Score (overall)")
    print(_line("-", 60))
    for r in runs:
        print(_kv(r.name, fmt_run(r)))
    print()

    print("By-suite")
    print(_line("-", 60))
    suites_sorted = sorted({ex.suite for ex in examples})
    for suite in suites_sorted:
        row = [f"{suite:<10}"]
        for r in runs:
            tot = r.per_suite_total[suite]
            cor = r.per_suite_correct[suite]
            inv = r.per_suite_invalid[suite]
            acc = (cor / tot) if tot else 0.0
            row.append(f"{r.name}: {acc*100:5.1f}% ({cor}/{tot}) inv={inv}")
        print("  ".join(row))
    print()

    print("By-category")
    print(_line("-", 60))
    cats = sorted({ex.category for ex in examples})
    for cat in cats:
        row = [f"{cat:<12}"]
        for r in runs:
            tot = r.per_cat_total[cat]
            cor = r.per_cat_correct[cat]
            inv = r.per_cat_invalid[cat]
            acc = (cor / tot) if tot else 0.0
            row.append(f"{r.name}: {acc*100:5.1f}% ({cor}/{tot}) inv={inv}")
        print("  ".join(row))
    print(_line("=", 60))


if __name__ == "__main__":
    main()
