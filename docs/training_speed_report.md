# MiniLLM 训练加速系统性调研报告

日期: 2026-01-23

## 目标与评价指标
- 目标: 在不显著降低质量的前提下，提高训练吞吐 (tokens/s)、降低单位损失/困惑度的时间与成本。
- 关键指标: 吞吐 (tok/s)、MFU (模型 FLOPs 利用率)、内存占用、训练稳定性 (loss 曲线)、最终质量 (eval)。

## 1) 数据维度 (质量 + 去重 + 采样)
**为何重要**: 数据质量直接影响收敛速度和所需训练步数，重复数据会导致记忆化并降低样本效率。

- **去重/近重复**: 公开研究表明，去重能显著减少模型“记忆化”并减少达到同等/更好精度所需的训练步数。可作为首要降本提速手段。 [1]
- **高质量过滤流水线**: CCNet 等工作展示了“语言识别 + 去重 + 质量过滤 (对齐高质量语料)”的系统化数据清洗流程，可显著提升语料质量。 [2]
- **清洗版 Common Crawl**: C4 是清洗后的 Common Crawl 语料，体现了“清洗 + 过滤”的基线工程路线。 [3]
- **更高级过滤**: 近期工作用 LLM 作为“高标准过滤器”，再训练轻量分类器进行大规模过滤，质量优于传统规则过滤，但需权衡成本。 [4]

**可落地动作**
- 数据去重 (doc 级 / 句子级 / n-gram 级) → 缩短训练步数。 [1]
- 质量过滤 (语言识别 + 规则过滤 + 统计/困惑度筛选) → 提升样本效率。 [2]

## 2) 模型架构维度 (计算瓶颈优化)
- **FlashAttention / FlashAttention-2**: IO-aware 注意力显著降低内存读写并提升训练吞吐，FlashAttention-2 在并行划分上进一步提速。 [5][6]
- **GQA/MQA 降 KV 开销**: MQA 通过共享 K/V head 降低解码带宽开销；GQA 在速度与质量间折中，并可从多头模型“少量计算”迁移得到。 [7][8]
- **MoE (Switch Transformers)**: 稀疏激活在相近计算预算下扩大模型容量，并报告显著预训练加速。 [9]
- **算力最优规模 (Chinchilla)**: 模型大小与训练 token 数应同比例增长，避免“欠训练”导致的低效。 [10]

## 3) 优化器与数值精度
- **混合精度训练**: FP16/BF16 + FP32 master weights 可显著降低内存并提升吞吐。 [11][12]
- **Adafactor**: 用低秩统计量替代完整二阶矩，显著降低优化器状态内存。 [13]
- **Lion**: 仅保留动量的一阶状态，内存开销更低，且在多任务中表现具竞争力。 [14]

## 4) 并行化与系统层面
- **Activation Checkpointing**: 以额外前向计算换显存，可提升 batch/序列长度上限。 [15]
- **ZeRO (分片优化器/梯度/参数)**: 显著降低冗余内存并提升可训练规模。 [16]
- **Tensor Parallel / Pipeline Parallel**: Megatron-LM (层内并行) 与 GPipe (流水线) 是大模型训练的经典路线。 [17][18]
- **Sequence Parallel**: 通过切分序列维度突破长序列训练内存瓶颈。 [19]

## 5) 建议的系统化加速路线 (从低风险到高收益)
1. **数据去重与质量过滤优先**: 以同等质量更少 token 达到收敛。 [1][2]
2. **混合精度 + 优化器内存优化**: BF16/FP16 + 低状态优化器。 [11][12][13]
3. **高效注意力核**: FlashAttention/FlashAttention-2。 [5][6]
4. **并行化策略组合**: ZeRO + Tensor/Pipeline/Sequence Parallel。 [16][17][18][19]
5. **架构层面改造**: GQA/MQA、MoE、或更激进的稀疏/低秩结构。 [7][8][9]

## 6) 风险与权衡
- **质量 vs 速度**: 去重/过滤可能引入分布偏移，需要回归测试。 [1][2]
- **吞吐 vs 稳定性**: 混合精度/低状态优化器需调学习率与稳定性策略。 [11][12][13][14]
- **复杂度成本**: 并行化与 MoE 需要通信/系统复杂度投入。 [9][16][17][18]

## 7) MiniLLM 具体落地建议 (结合本仓库)
- 数据侧: 对 `dataset/minimind/*.jsonl` 做近重复去重与质量过滤，优先落地。 [1][2]
- 训练侧: 统一使用 BF16 或 FP16 混合精度，并切换低内存优化器做对比基线。 [11][12][13][14]
- 算子侧: MLX 侧关注注意力实现与 cache 行为，优先验证 FlashAttention 类改进的收益。 [5][6]
- 并行侧: 若扩到多卡/集群，优先 ZeRO + pipeline/tensor parallel，再考虑 sequence parallel。 [16][17][18][19]

### MLX / Apple GPU 约束下的“负载优先”优化清单
这些建议主要通过减少 CPU 数据管线开销、降低 padding 与同步开销来提升 GPU 利用率：
- **预分词/缓存**: 使用 `python -m mlx_train.pretokenize` 生成 `ids`；或在训练时启用 `--cache_tokenized`（内存换速度）。  
- **数据预取**: 启用 `--prefetch_batches N`（本次新增）后台线程预取 micro-batch，减少 GPU 等待数据。  
- **Bucketing 降 padding**: 使用 `--bucket_sizes 256,512` 等设置，将 batch padding 降到最小桶。  
- **bin2d 固定长度**: `--data_format bin2d --bin_cache memory` 直接喂固定长度 ids，减少 Python 端 padding/切片开销。  
- **细粒度 trace**: `--trace_timing_out out/trace_timing` 可输出逐层/逐阶段 timing JSON，定位热点（trace steps 会禁用 compile 以便准确计时）。  
- **SFT 稀疏损失**: 对话数据常见“仅 assistant token 计 loss”，启用 `--sparse_loss` 可显著减少无效计算。  
- **显存上限自动化**: `--memory_limit_auto_ratio 0.9 --memory_limit_auto_mode available` 用“可用内存”比例自动限额，防止峰值炸机。  
- **短跑禁用保存**: `--save_interval 0`（仅用于测速）避免 checkpoint I/O 干扰吞吐统计。  

### 本地 tiny 基准（time-first, 2026-01-23）
以 `preset=tiny` 为基准，`seq_len=512, accum=1, max_steps=50`：
- **bf16 基线**: 61k tok/s。  
- **float16**: 59k tok/s（略慢）。  
- **bf16 + sparse_loss**: 75k tok/s（约 +22%）。  
- **bf16 + sparse_loss + label_bucket_sizes=64,128,256,512 + prefetch=4 + paired_heads**: 83.9k tok/s。  
- **bin2d + memory + shuffle=512 + batch=72**: 87.0k tok/s（当前“有 shuffle”的最快组合，需 ~14GB 限额）。  
- **bin2d + memory + shuffle=512 + batch=64**: 83.9k tok/s。  
- **bin2d + memory + shuffle=0**: 80.8k tok/s（速度快但不推荐用于正式训练）。  
- **bin2d + memory + shuffle=1024/2048**: 78.9k / 81.0k tok/s（略慢）。  
- **bin2d + memory + shuffle=4096**: 57.9k tok/s（本机一次性测试异常偏慢）。  
-> 预分词 `sft_mini_512.ids.jsonl`（约 1.6G）在当前设置下未提升吞吐（约 69.8k tok/s），推测 I/O + JSON 解析开销抵消了 tokenizer 省时。

### 质量快检（仅作方向判断）
在相同数据与步数下，对 `paired_heads` 做 eval（shuffle=2048）：
- 100 batches：baseline avg_loss=6.2759，paired_heads avg_loss=5.9880  
- 1000 batches：baseline avg_loss=6.5693，paired_heads avg_loss=6.3673  
结论：短跑评估未见质量下降（甚至略优），但仍需更长训练/更正式评测确认。

### 额外方案探索（均未超越当前 best）
在 paired_heads + shuffle=2048 的质量约束下测试：
- `--hidden_act relu2`、`--qk_norm`、`--batch_size 72`、`--value_mix 0.1`、`--embed_skip_scale 0.1` 均较慢（详见 `docs/training_speed_experiments.md`）。

### 新一轮参数搜索（>=6 组）
在 paired_heads + sparse_loss 基线上测试：
- `prefetch=2/6/8` 均低于 `prefetch=4`
- `bucket_sizes=256,384,512` 明显变慢
- `label_bucket_sizes=32,64,128,256,512` 未优于默认
- bin2d + `shuffle_buffer=0/512/1024/2048/4096` 测试中，**512 最优**，4096 异常偏慢
- `optim_state_dtype=param`（b72）略慢
- `optimizer=lion`（b72）明显变慢
- 小步 batch 扫描：b68/b76 均低于 b72；b72 + shuffle=768/prefetch=3/5 仍低于 b72+shuffle=512
- `optimizer=adafactor`（b72）显著变慢
- `optimizer=muon` 修复 compile 兼容后仍极慢（5-step 测试 >40s 无输出，已中止）

当前推荐（质量优先）：`data_format=bin2d` + `bin_cache=memory` + `shuffle_buffer=512` + `prefetch=4` + `paired_heads` + `batch_size=72`（需 ~14GB 限额）。

### 二进制数据格式（packbin）探索
将 JSONL 打包为 `.ids.bin/.idx`（可选 label positions）：
- bin + labels + mmap：~69.1k tok/s（慢）
- bin + labels + memory：~75.4k tok/s（慢）
- bin no‑label + mmap：~76.3k tok/s（慢）
- bin no‑label + memory：~81.3k tok/s（≈基线）
结论：当前实现下，二进制格式**未显著快于**原始 JSONL + shuffle=2048；仅 “no‑label + memory” 能追平基线。

### bin2d 固定长度格式（更激进）
在 paired_heads + sparse_loss + prefetch=4 下测试：
- bin2d + memory + shuffle=512 + batch=72：~87.0k tok/s（当前最快且带 shuffle，需 ~14GB）
- bin2d + memory + shuffle=512 + batch=80：~86.3k tok/s（略低于 b72）
- bin2d + memory + shuffle=512 + batch=64：~83.9k tok/s
- bin2d + memory + shuffle=0：~80.8k tok/s（快但无 shuffle）
- bin2d + memory + shuffle=1024：~78.9k tok/s
- bin2d + memory + shuffle=2048：~81.0k tok/s
- bin2d + memory + shuffle=4096：~57.9k tok/s（单次异常偏慢）
- bin2d + mmap：~80.6k tok/s（略慢）
- bin2d + memory + no compile：~66.2k tok/s（明显变慢）
结论：**bin2d + memory + shuffle=512** 是目前最优组合之一，但占用更多 RAM 与磁盘。
按 87.0k tok/s 估算，`sft_mini_512` 每个 epoch 约 **1.98 小时**。

### 端到端 trace（tiny, 单步）
以 `bin2d + shuffle=512 + paired_heads` 为例，`--trace_timing_out` 单步采样：
- 单步总耗时 ~484ms，其中 `opt` ~475ms，`fwd_bwd` ~460ms（优化器与反向是主耗时）。  
- 最慢的逐层事件集中在 attention（layer.1/2/3 attn）。  
- 峰值显存 ~3.5GiB，active ~3.1GiB（小模型下内存充裕）。  
结论：优化方向以 **降低优化器更新成本** 与 **进一步提升 attention kernel** 为优先。

### GPU 占用节流（可用性优先）
在 `--step_sleep_ms=3` 下进行节流试验，GPU 不再打满，tok/s 下降到 73k–75k：
- b64 p2 sleep3: ~73.3k tok/s
- b60 p2 sleep3: ~73.8k tok/s
- b60 p1 sleep3: ~75.0k tok/s

进一步尝试“稀疏睡眠”（每 N 步 sleep 一次）以靠近 87k：
- b72 sleep0.5ms (每步): ~77.8k tok/s
- b72 sleep1ms /2步: ~75.8k tok/s
- b72 sleep2ms /8步: ~79.4k tok/s
- b72 sleep5ms /10步: ~79.2k tok/s
- b72 sleep0.2ms /8步: ~79.6k tok/s
- b72 sleep0.05ms /16步: **~85.9k tok/s**
不节流对比（当前空闲较多的状态）：
- b72 nosleep: ~83.4k tok/s
- b80 nosleep: ~83.9k tok/s
- b84 nosleep: ~71.0k tok/s
基线复测（b72/p4/s512）：~80.7k tok/s（复测2次约 ~83.4k，说明系统/调度噪声影响较大）。
继续按调研建议调 `prefetch/shuffle`：
- b72 p6 s512：~85.8k tok/s（显著回升）
- b72 p4 s768：~85.7k tok/s
- b72 p6 s768：~85.4k tok/s
- b72 p5 s512：~85.2k tok/s
- b72 p7 s512：~84.8k tok/s
- b72 p8 s512：~84.3k tok/s
- b72 p6 s1024：~80.4k tok/s（变慢）
- b72 p6 s640：~75.5k tok/s（明显变慢）
编译/内核路径验证：
- 关闭 compile：~67.8k tok/s（明显变慢）
- 关闭 compile + compile_optimizer：~71.7k tok/s（仍明显变慢）
- 关闭 metal_kernels：~77.3k tok/s（变慢）
结论：当前系统状态下，**极轻度节流（0.05ms/16步）反而快于不节流**；对不节流场景，提升最快的路径是调大 `prefetch` 或 `shuffle`，且必须保持 compile + compile_optimizer + metal_kernels。

优化器状态 dtype 验证：
- `optim_state_dtype=param`：~82.2k tok/s（比 float32 慢）
结论：**param dtype 不利于吞吐**，维持 float32。

## 9) 进一步调研：长序列训练与注意力系统优化（2023-2025）
- **FlashAttention-3** 在 H100 上通过异步流水与 FP8 低精度等技术把注意力利用率显著拉升，报告 1.5-2.0x 速度提升与高 TFLOPs。[25]
- **Ring Attention** 用 blockwise attention + 通信重叠把序列拆分到多设备，可处理更长上下文。[26]
- **Striped Attention** 在因果场景改造 Ring Attention，修复负载不均，报告最高 1.45x 吞吐提升。[27]
- **DistFlashAttn** 结合负载均衡、KV 通信重叠与“重计算友好”checkpointing，在长序列训练上取得显著加速。[28]
- **DeepSpeed-Ulysses** 以 all-to-all 方式做序列并行，通信量随设备扩展保持常数；论文报告 2.5x 速度 + 4x 更长序列。[29]
- **LoongTrain** 提出 head-context 2D 并行与 Double-Ring-Attention，报告 MFU 最多提升 2.88x。[30]
- **ZeRO** 通过分片优化器/梯度/参数降低显存冗余，提升可训练规模。[31][32]
- **FSDP** 在 PyTorch 中提供参数/梯度/优化器状态的全分片能力，提升内存效率与规模上限。[33]
- **Activation checkpointing** 用“重算换显存”以降低激活峰值，适合提升 batch 或 seq_len。[34]
- **xFormers memory-efficient attention** 提供多种内核实现以降低注意力内存与提升速度（CUDA 环境）。[35]

### 1000-step 训练质量对比（shuffle=4096）
- baseline avg_loss=3.8709 (1000-batch eval)
- paired_heads avg_loss=3.8759 (1000-batch eval)
结论：1000-step 下两者质量接近，paired_heads 未见明显退化。
- `--grad_clip 0` + `--sparse_loss` + compile 会触发 MLX 运行时错误（unordered_map::at），暂不推荐。

## 8) Modded‑NanoGPT 速度赛方案要点（整理）
> 来源：KellerJordan/modded‑nanogpt README（速度赛 8×H100 目标 3.28 loss 的组合优化）[24]

**目标与约束**
- 目标：8×H100 上在 FineWeb 验证集达到 3.28 loss；<100 秒 / <500M tokens 的 speedrun 条件。[24]

**架构与损失侧**
- Rotary、QK‑Norm、ReLU²。[24]
- 投影层零初始化（muP‑like）。[24]
- 残差增强：embedding→各层 skip、以及特定 block 间 skip。[24]
- 注意力 value 中混入额外 embeddings，并配套 gating。[24]
- 允许在预测前“回退”部分层贡献（back‑out）。[24]
- Smear（1‑token look‑back）、bigram hash embedding。[24]
- Paired head attention、partial key offset。[24]
- Multi‑token prediction (MTP)。[24]

**注意力/精度与核优化**
- Head matmul 用 FP8；logits softcap 与不对称 rescale。[24]
- Flash Attention 3 + long‑short sliding window（含 YaRN warmup）。[24]
- Sparse attention gate。[24]

**训练技巧与优化器**
- Muon 优化器（含 Polar Express / NorMuon 变体）。[24]
- embedding 与 lm_head 两步累积更新。[24]
- LR 绑定的谨慎 weight decay，残差流指数衰减。[24]
- Batch size schedule。[24]
- 训练中后期解绑 embed 与 lm_head。[24]

**数据与输入侧**
- batch 起始对齐 EOS + 限制最大文档长度。[24]

**系统层**
- README 指出还有大量系统级优化（不逐条列举）。[24]

### 对 MiniLLM/MLX 的可迁移性判断（面向时间优先）
- **较易落地（硬件依赖低）**：QK‑Norm、ReLU²、logit softcap、zero‑init residual、skip/value‑mix/gates、back‑out、smear、bigram hash、partial key offset、paired heads、MTP、EOS 对齐/最大文档长度、batch size schedule。[24]
- **强硬件依赖（MLX 上难复刻或收益小）**：FP8 head matmul、FlashAttention3（特定窗口模式）。[24]
- **优化器依赖（需单独验证）**：Muon 及其变体。[24]
- **稀疏损失 (SFT)**: 若 loss_mask 稀疏，启用 `--sparse_loss` + `--label_bucket_sizes` 可减少无效计算。  
- **编译/内核**: 保持 `--compile` / `--compile_optimizer` / `--metal_kernels` 打开，减少 Python/框架开销。  

## 8) nanochat / modded-nanogpt 速度技巧 (可迁移与不可迁移)
**nanochat (Karpathy)** 重点在数据管线效率：无限流式 dataloader、并行 tokenization + batch tokenize、pinned memory + non_blocking 传输、DDP 分片读取减少竞争等。 [20][21][22][23]

**modded-nanogpt (KellerJordan)** 是 H100 speedrun 取向：ReLU^2、QK-Norm、投影零初始化、额外 skip connections、attention value 混入 embedding、logit softcap、FP8 head matmul、FlashAttention-3 的长短滑窗等一组组合拳。 [24]

**MLX 迁移取舍**:
- 可直接迁移: ReLU^2、QK-Norm、logit softcap、投影零初始化、value-mix、EOS 对齐/长度上限。
- 受限/难迁移: FP8、FlashAttention-3 及其 CUDA kernel 依赖。

## 10) 本轮实验前调研（2026-01-23）
目标：在 **不节流** 前提下回到或超过 87k tok/s；先定位最可能影响吞吐的两类因素：
- **数据管线抖动**：bin2d 固定长度下，shuffle/prefetch 改变供数方差，历史记录显示 `shuffle=512` 最稳，但系统空闲度变化时结果有波动。
- **编译与内核路径**：`mx.compile`/`compile_optimizer` 与 `metal_kernels` 的组合决定了核心算子是否走 fused path，理论上应显著影响 tok/s。
基于以上调研，实验顺序：先重测 **基线 b72/p4/s512**，再扫 `shuffle/prefetch` 组合，最后验证编译/内核开关的影响。

## 11) 本轮调研（2026-01-23 追加）
目标：尝试向 160k tok/s 靠近，但保持训练形态不变（不改模型规模）。
调研结论：
- **吞吐上限与 FLOPs/Token 近似线性**：在 GPU 满载情况下，tok/s 难以成倍提升，除非降低每 token 计算量或引入更高效内核。
- **dtype 影响内核路径**：float16 可能触发更快的内核实现，但在 MLX/Apple GPU 上不保证优于 bfloat16。
- **序列长度对效率有非线性影响**：更长序列可能提高 kernel 利用率（减少 launch overhead），但注意力复杂度随长度增长，吞吐不一定上升。
- **accum_steps 降低优化器调用频率**：有机会稍微抬升 tok/s，但效果通常有限。
因此，本轮实验围绕：`float16`、`accum_steps=2`、`seq_len=256/384` 三方向进行验证（把 seq_len 作为“速度上限”探测）。

实验结果：
- float16（b72/p6/s512）：~74.1k tok/s（比 bfloat16 慢）
- accum_steps=2（b72/p6/s512）：~80.8k tok/s（变慢）
- seq_len=256（jsonl ids, b144）：~82.7k tok/s（未提升）
- seq_len=384（jsonl ids, b96）：~84.6k tok/s（略低于最优）
结论：当前 MLX/Apple GPU 上 **降低 seq_len 与改 float16 并不能拉升吞吐**；最大收益仍来自 bin2d + prefetch/shuffle 的数据管线组合。

### modded-nanogpt README 逐条清单（2026-01-23）
以下为 README 中列出的“速度赛”组合拳（原文顺序），便于后续逐项对照与落地规划：
- Modernized architecture: Rotary embeddings、QK-Norm、ReLU^2
- Muon optimizer (及其相关实现/变体)
- FP8 matmul for head + asym rescale + logit softcap
- Projection zero-init (muP-like)
- Skip connections: embedding -> every block；block 3 -> 6
- Extra embeddings mixed into attention values (Zhou et al. 2024)
- FlashAttention-3 (long-short sliding window + YaRN warmup)
- Align training batch starts with EoS + max document length
- Accumulate gradients for 2 steps for embedding + lm_head
- “Back out” contributions from first 2/3 layers before prediction
- Polar Express (Muon 相关)
- Smear module (1-token lookback)
- Sparse attention gate
- NorMuon
- Cautious weight decay (schedule tied to LR)
- Exponential decay of residual stream
- Batch size schedule
- Partial key offset
- Multi-token prediction (MTP)
- Untie embed & lm_head at 2/3 training
- Additional gating on value embeddings + skip connection
- Paired head attention
- Bigram hash embedding
- 以及更多系统级优化

---

## References
[1] https://aclanthology.org/2022.acl-long.577/
[2] https://arxiv.org/abs/1911.00359
[3] https://www.tensorflow.org/datasets/catalog/c4
[4] https://arxiv.org/abs/2410.02755
[5] https://arxiv.org/abs/2205.14135
[6] https://arxiv.org/abs/2307.08691
[7] https://arxiv.org/abs/1911.02150
[8] https://arxiv.org/abs/2305.13245
[9] https://arxiv.org/abs/2101.03961
[10] https://arxiv.org/abs/2203.15556
[11] https://arxiv.org/abs/1710.03740
[12] https://arxiv.org/abs/1905.12322
[13] https://arxiv.org/abs/1804.04235
[14] https://arxiv.org/abs/2302.06675
[15] https://arxiv.org/abs/1604.06174
[16] https://arxiv.org/abs/1910.02054
[17] https://arxiv.org/abs/1909.08053
[18] https://arxiv.org/abs/1811.06965
[19] https://arxiv.org/abs/2105.13120
[20] https://deepwiki.com/karpathy/nanochat/4.3-data-loading
[21] https://deepwiki.com/karpathy/nanochat/4.2-data-loading-and-efficiency
[22] https://deepwiki.com/karpathy/nanochat/4.3-training-loop-and-features
[23] https://github.com/karpathy/nanochat
[24] https://github.com/KellerJordan/modded-nanogpt
[25] https://arxiv.org/abs/2407.08608
[26] https://arxiv.org/abs/2310.01889
[27] https://arxiv.org/abs/2311.09431
[28] https://arxiv.org/abs/2310.03294
[29] https://arxiv.org/abs/2309.14509
[30] https://arxiv.org/abs/2406.18485
[31] https://arxiv.org/abs/1910.02054
[32] https://docs.pytorch.org/docs/stable/fsdp.html
[33] https://docs.pytorch.org/docs/stable/checkpoint
[34] https://facebookresearch.github.io/xformers/components/ops.html
