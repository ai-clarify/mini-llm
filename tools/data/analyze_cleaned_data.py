#!/usr/bin/env python3
"""
分析 cleaned.jsonl 文件的数据质量和潜在优化点
"""
import json
import hashlib
from collections import defaultdict, Counter
import random

def analyze_data(filepath, sample_size=10000):
    """分析数据文件"""

    print(f"正在分析: {filepath}")
    print("=" * 80)

    # 统计信息
    total_count = 0
    user_lengths = []
    assistant_lengths = []
    conversation_counts = []
    duplicates = defaultdict(list)
    empty_user = 0
    empty_assistant = 0
    very_short_user = 0  # < 5 chars
    very_short_assistant = 0  # < 10 chars
    very_long_user = 0  # > 1000 chars
    very_long_assistant = 0  # > 5000 chars
    format_errors = 0

    # 采样数据用于详细检查
    samples = []

    # 第一遍：收集统计信息
    print("\n第一遍扫描：收集统计信息...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"  处理了 {line_num:,} 条记录...")

            try:
                data = json.loads(line.strip())
                total_count += 1

                # 检查格式
                if 'conversations' not in data:
                    format_errors += 1
                    continue

                conversations = data['conversations']
                conversation_counts.append(len(conversations))

                # 提取用户和助手的内容
                user_content = ""
                assistant_content = ""

                for conv in conversations:
                    if conv.get('role') == 'user':
                        user_content = conv.get('content', '')
                    elif conv.get('role') == 'assistant':
                        assistant_content = conv.get('content', '')

                # 统计长度
                user_len = len(user_content)
                assistant_len = len(assistant_content)

                user_lengths.append(user_len)
                assistant_lengths.append(assistant_len)

                # 检查空内容
                if user_len == 0:
                    empty_user += 1
                if assistant_len == 0:
                    empty_assistant += 1

                # 检查极短内容
                if user_len > 0 and user_len < 5:
                    very_short_user += 1
                if assistant_len > 0 and assistant_len < 10:
                    very_short_assistant += 1

                # 检查极长内容
                if user_len > 1000:
                    very_long_user += 1
                if assistant_len > 5000:
                    very_long_assistant += 1

                # 检测重复（使用内容的哈希）
                content_hash = hashlib.md5(
                    (user_content + assistant_content).encode('utf-8')
                ).hexdigest()
                duplicates[content_hash].append(line_num)

                # 采样
                if len(samples) < sample_size and random.random() < (sample_size / 1200000):
                    samples.append({
                        'line_num': line_num,
                        'user': user_content[:200],
                        'assistant': assistant_content[:200],
                        'user_len': user_len,
                        'assistant_len': assistant_len
                    })

            except json.JSONDecodeError:
                format_errors += 1
            except Exception as e:
                print(f"  警告：第 {line_num} 行出现错误: {str(e)}")

    # 统计重复数据
    duplicate_groups = {k: v for k, v in duplicates.items() if len(v) > 1}
    duplicate_count = sum(len(v) - 1 for v in duplicate_groups.values())

    # 打印统计结果
    print("\n" + "=" * 80)
    print("统计结果:")
    print("=" * 80)
    print(f"\n总记录数: {total_count:,}")
    print(f"格式错误: {format_errors:,}")

    print(f"\n用户消息长度:")
    print(f"  平均: {sum(user_lengths) / len(user_lengths):.1f} 字符")
    print(f"  中位数: {sorted(user_lengths)[len(user_lengths)//2]:.1f} 字符")
    print(f"  最小: {min(user_lengths):,} 字符")
    print(f"  最大: {max(user_lengths):,} 字符")
    print(f"  空内容: {empty_user:,} ({empty_user/total_count*100:.2f}%)")
    print(f"  极短内容 (<5字符): {very_short_user:,} ({very_short_user/total_count*100:.2f}%)")
    print(f"  极长内容 (>1000字符): {very_long_user:,} ({very_long_user/total_count*100:.2f}%)")

    print(f"\n助手回复长度:")
    print(f"  平均: {sum(assistant_lengths) / len(assistant_lengths):.1f} 字符")
    print(f"  中位数: {sorted(assistant_lengths)[len(assistant_lengths)//2]:.1f} 字符")
    print(f"  最小: {min(assistant_lengths):,} 字符")
    print(f"  最大: {max(assistant_lengths):,} 字符")
    print(f"  空内容: {empty_assistant:,} ({empty_assistant/total_count*100:.2f}%)")
    print(f"  极短内容 (<10字符): {very_short_assistant:,} ({very_short_assistant/total_count*100:.2f}%)")
    print(f"  极长内容 (>5000字符): {very_long_assistant:,} ({very_long_assistant/total_count*100:.2f}%)")

    print(f"\n对话轮次:")
    conv_counter = Counter(conversation_counts)
    for count in sorted(conv_counter.keys())[:10]:
        print(f"  {count} 轮: {conv_counter[count]:,} ({conv_counter[count]/total_count*100:.2f}%)")

    print(f"\n重复数据:")
    print(f"  重复组数: {len(duplicate_groups):,}")
    print(f"  重复记录数: {duplicate_count:,} ({duplicate_count/total_count*100:.2f}%)")

    if len(duplicate_groups) > 0:
        print(f"\n  最多重复的前 5 组:")
        sorted_dups = sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (hash_val, lines) in enumerate(sorted_dups[:5], 1):
            print(f"    {i}. 重复 {len(lines)} 次，行号: {lines[:5]}{'...' if len(lines) > 5 else ''}")

    # 展示一些样本
    print(f"\n随机样本 (10条):")
    print("=" * 80)
    for i, sample in enumerate(random.sample(samples, min(10, len(samples))), 1):
        print(f"\n样本 {i} (行 {sample['line_num']}):")
        print(f"  用户 ({sample['user_len']} 字符): {sample['user'][:100]}...")
        print(f"  助手 ({sample['assistant_len']} 字符): {sample['assistant'][:100]}...")

    # 分析潜在优化点
    print("\n" + "=" * 80)
    print("潜在优化建议:")
    print("=" * 80)

    issues = []

    if duplicate_count > 0:
        issues.append(f"1. 去重：发现 {duplicate_count:,} 条重复记录 ({duplicate_count/total_count*100:.2f}%)")

    if empty_user > 0 or empty_assistant > 0:
        issues.append(f"2. 删除空内容：{empty_user:,} 条空用户消息，{empty_assistant:,} 条空助手回复")

    if very_short_user > 0 or very_short_assistant > 0:
        issues.append(f"3. 过滤极短内容：{very_short_user:,} 条极短用户消息 (<5字符)，{very_short_assistant:,} 条极短助手回复 (<10字符)")

    if very_long_assistant > 0:
        issues.append(f"4. 检查极长回复：{very_long_assistant:,} 条极长助手回复 (>5000字符)，可能包含重复或低质量内容")

    if format_errors > 0:
        issues.append(f"5. 修复格式错误：{format_errors:,} 条格式错误的记录")

    if issues:
        for issue in issues:
            print(f"\n{issue}")
    else:
        print("\n未发现明显的数据质量问题！数据已经很干净了。")

    # 估算优化后的数据量
    potential_removal = duplicate_count + empty_user + empty_assistant + very_short_user + very_short_assistant + format_errors
    potential_removal = min(potential_removal, total_count)  # 避免重复计数

    print(f"\n如果应用所有优化，预计可以移除约 {potential_removal:,} 条记录")
    print(f"优化后预计剩余: {total_count - potential_removal:,} 条记录 ({(total_count - potential_removal)/total_count*100:.2f}%)")


if __name__ == '__main__':
    analyze_data('data/final/sft_mini_512.cleaned.jsonl', sample_size=1000)
