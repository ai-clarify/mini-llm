import os
import random
import re
import sys
from pathlib import Path
from threading import Thread

import torch
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="MiniLLM", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_assistant_content(content):
    if model_source == "API":
        if 'R1' not in api_model_name:
            return content
    elif model_source == "本地模型":
        if 'R1' not in MODEL_PATHS[selected_model][1]:
            return content
    else:
        return content

    if '<think>' in content and '</think>' in content:
        content = re.sub(r'(<think>)(.*?)(</think>)',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        content = re.sub(r'<think>(.*?)$',
                         r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        content = re.sub(r'(.*?)</think>',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    return content


import numpy as np


@st.cache_resource
def load_model_tokenizer(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Local model path not found: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.eval().to(device)
    return model, tokenizer


def _resolve_mlx_checkpoint(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file() and candidate.name.endswith(".safetensors"):
        candidate = candidate.parent
    if candidate.is_dir() and (candidate / "model.safetensors").is_file():
        return candidate

    # Try to resolve from an output root (out/mlx style).
    def _iter_step_dirs(base: Path) -> list[tuple[int, Path]]:
        if not base.exists() or not base.is_dir():
            return []
        out: list[tuple[int, Path]] = []
        for child in base.iterdir():
            if not child.is_dir() or not child.name.startswith("step_"):
                continue
            try:
                step = int(child.name.split("_", 1)[1])
            except ValueError:
                continue
            if (child / "model.safetensors").is_file() and (child / "config.json").is_file():
                out.append((step, child))
        out.sort(key=lambda x: x[0])
        return out

    roots = [
        candidate / "sft" / "checkpoints",
        candidate / "pretrain" / "checkpoints",
        candidate / "checkpoints",
    ]
    for root in roots:
        steps = _iter_step_dirs(root)
        if steps:
            return steps[-1][1]

    raise FileNotFoundError(f"MLX checkpoint not found under: {path}")


@st.cache_resource
def load_mlx_model_tokenizer(checkpoint_dir: str, tokenizer_path: str):
    from transformers import AutoTokenizer

    from mlx_train.cli.infer import load_config
    from mlx_train.models import MiniLLMForCausalLM

    ckpt = _resolve_mlx_checkpoint(checkpoint_dir)
    cfg = load_config(ckpt)
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(ckpt / "model.safetensors"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=os.path.isdir(tokenizer_path),
    )
    return model, tokenizer, cfg


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    if st.button("🗑", key=f"delete_{i}"):
                        st.session_state.messages.pop(i)
                        st.session_state.messages.pop(i - 1)
                        st.session_state.chat_messages.pop(i)
                        st.session_state.chat_messages.pop(i - 1)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: #ddd; border-radius: 10px; color: black;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


def delete_conversation(index):
    st.session_state.messages.pop(index)
    st.session_state.messages.pop(index - 1)
    st.session_state.chat_messages.pop(index)
    st.session_state.chat_messages.pop(index - 1)
    st.rerun()


st.sidebar.title("模型设定调整")

# st.sidebar.text("训练数据偏差，增加上下文记忆时\n多轮对话（较单轮）容易出现能力衰减")
st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
# st.session_state.history_chat_num = 0
st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192, step=1)
st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)

model_source = st.sidebar.radio("选择模型来源", ["本地模型", "MLX", "API"], index=1)

if model_source == "API":
    api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/v1")
    api_model_id = st.sidebar.text_input("Model ID", value="minillm")
    api_model_name = st.sidebar.text_input("Model Name", value="MiniLLM2")
    api_key = st.sidebar.text_input("API Key", value="none", type="password")
    slogan = f"Hi, I'm {api_model_name}"
elif model_source == "MLX":
    mlx_checkpoint = st.sidebar.text_input("MLX Checkpoint/Out Dir", value="out/mlx")
    mlx_tokenizer_path = st.sidebar.text_input("Tokenizer Path", value="./model")
    slogan = "Hi, I'm MLX"
else:
    MODEL_PATHS = {
        "MiniLLM2-R1 (0.1B)": ["../MiniLLM2-R1", "MiniLLM2-R1"],
        "MiniLLM2-Small-R1 (0.02B)": ["../MiniLLM2-Small-R1", "MiniLLM2-Small-R1"],
        "MiniLLM2 (0.1B)": ["../MiniLLM2", "MiniLLM2"],
        "MiniLLM2-MoE (0.15B)": ["../MiniLLM2-MoE", "MiniLLM2-MoE"],
        "MiniLLM2-Small (0.02B)": ["../MiniLLM2-Small", "MiniLLM2-Small"]
    }

    selected_model = st.sidebar.selectbox('Models', list(MODEL_PATHS.keys()), index=2)  # 默认选择 MiniLLM2
    model_path = MODEL_PATHS[selected_model][0]
    slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"

image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniLLM/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 45px; height: 45px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    if model_source == "本地模型":
        try:
            model, tokenizer = load_model_tokenizer(model_path)
        except Exception as e:
            st.error(f"本地模型加载失败: {e}")
            model, tokenizer = None, None
    elif model_source == "MLX":
        try:
            model, tokenizer, _ = load_mlx_model_tokenizer(mlx_checkpoint, mlx_tokenizer_path)
        except Exception as e:
            st.error(f"MLX 模型加载失败: {e}")
            model, tokenizer = None, None
    else:
        model, tokenizer = None, None

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder="给 MiniLLM 发送消息")

    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            if model_source == "API":
                try:
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=api_key,
                        base_url=api_url
                    )
                    history_num = st.session_state.history_chat_num + 1  # +1 是为了包含当前的用户消息
                    conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]
                    answer = ""
                    response = client.chat.completions.create(
                        model=api_model_id,
                        messages=conversation_history,
                        stream=True,
                        temperature=st.session_state.temperature
                    )

                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        answer += content
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

                except Exception as e:
                    answer = f"API调用出错: {str(e)}"
                    placeholder.markdown(answer, unsafe_allow_html=True)
            elif model_source == "MLX":
                if model is None or tokenizer is None:
                    raise RuntimeError("MLX model/tokenizer is not loaded")

                import mlx.core as mx

                from mlx_train.cli.infer import generate as mlx_generate

                random_seed = random.randint(0, 2 ** 32 - 1)
                mx.random.seed(random_seed)

                st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[
                    -(st.session_state.history_chat_num + 1):]
                new_prompt = tokenizer.apply_chat_template(
                    st.session_state.chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
                out_ids = mlx_generate(
                    model,
                    input_ids=prompt_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=st.session_state.max_new_tokens,
                    min_new_tokens=0,
                    banned_token_ids=list(tokenizer.all_special_ids),
                    temperature=st.session_state.temperature,
                    top_p=0.85,
                    max_seq_len=None,
                )
                answer = tokenizer.decode(out_ids[len(prompt_ids):], skip_special_tokens=True)
                placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            else:
                if model is None or tokenizer is None:
                    raise RuntimeError("Local model/tokenizer is not loaded")

                random_seed = random.randint(0, 2 ** 32 - 1)
                setup_seed(random_seed)

                st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[
                                                                 -(st.session_state.history_chat_num + 1):]
                new_prompt = tokenizer.apply_chat_template(
                    st.session_state.chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    new_prompt,
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
                    "num_return_sequences": 1,
                    "do_sample": True,
                    "attention_mask": inputs.attention_mask,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "temperature": st.session_state.temperature,
                    "top_p": 0.85,
                    "streamer": streamer,
                }

                Thread(target=model.generate, kwargs=generation_kwargs).start()

                answer = ""
                for new_text in streamer:
                    answer += new_text
                    placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            with st.empty():
                if st.button("×", key=f"delete_{len(messages) - 1}"):
                    st.session_state.messages = st.session_state.messages[:-2]
                    st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                    st.rerun()


if __name__ == "__main__":
    main()
