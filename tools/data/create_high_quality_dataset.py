#!/usr/bin/env python3
"""
创建高质量数据集：
- 移除重复数据
- 移除空内容
- 移除极短内容
- 只保留高质量的对话数据
"""
import json
import hashlib
from pathlib import Path

def clean_dataset(input_file, output_file):
    """清理数据集，生成高质量数据"""

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("=" * 80)

    seen_hashes = set()
    total_count = 0
    removed_duplicate = 0
    removed_empty = 0
    removed_short = 0
    removed_invalid = 0
    kept_count = 0

    # 创建输出目录
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            if line_num % 100000 == 0:
                print(f"处理了 {line_num:,} 条记录，保留 {kept_count:,} 条...")

            total_count += 1

            try:
                data = json.loads(line.strip())

                # 检查格式
                if 'conversations' not in data or not isinstance(data['conversations'], list):
                    removed_invalid += 1
                    continue

                conversations = data['conversations']

                # 提取用户和助手的内容
                user_content = ""
                assistant_content = ""

                for conv in conversations:
                    if conv.get('role') == 'user':
                        user_content = conv.get('content', '').strip()
                    elif conv.get('role') == 'assistant':
                        assistant_content = conv.get('content', '').strip()

                # 过滤规则 1: 移除空内容
                if not user_content or not assistant_content:
                    removed_empty += 1
                    continue

                # 过滤规则 2: 移除极短内容
                # 用户消息至少 5 个字符，助手回复至少 10 个字符
                if len(user_content) < 5 or len(assistant_content) < 10:
                    removed_short += 1
                    continue

                # 过滤规则 3: 移除重复数据
                content_hash = hashlib.md5(
                    (user_content + assistant_content).encode('utf-8')
                ).hexdigest()

                if content_hash in seen_hashes:
                    removed_duplicate += 1
                    continue

                seen_hashes.add(content_hash)

                # 保留这条数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept_count += 1

            except json.JSONDecodeError:
                removed_invalid += 1
            except Exception as e:
                print(f"警告：第 {line_num} 行处理失败: {str(e)}")
                removed_invalid += 1

    # 打印统计信息
    print("\n" + "=" * 80)
    print("清理完成！")
    print("=" * 80)
    print(f"\n总记录数: {total_count:,}")
    print(f"\n移除的记录:")
    print(f"  重复数据: {removed_duplicate:,} ({removed_duplicate/total_count*100:.2f}%)")
    print(f"  空内容: {removed_empty:,} ({removed_empty/total_count*100:.2f}%)")
    print(f"  极短内容: {removed_short:,} ({removed_short/total_count*100:.2f}%)")
    print(f"  格式错误: {removed_invalid:,} ({removed_invalid/total_count*100:.2f}%)")
    print(f"  总计移除: {total_count - kept_count:,} ({(total_count - kept_count)/total_count*100:.2f}%)")
    print(f"\n保留的记录: {kept_count:,} ({kept_count/total_count*100:.2f}%)")

    return kept_count


if __name__ == '__main__':
    input_path = 'data/final/sft_mini_512.cleaned.jsonl'
    output_path = 'data/final/sft_high_quality.jsonl'

    clean_dataset(input_path, output_path)
