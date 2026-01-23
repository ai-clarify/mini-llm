# Training Speed Experiments (MLX)

Unless noted, runs are on sft_mini_512.jsonl, seq_len=512, bf16, prefetch=2.

## Tiny time-extreme (2026-01-23)
Preset=tiny, seq_len=512, batch=64, accum=1, prefetch=2, shuffle_buffer=0, max_steps=50, log_interval=50, memory_limit=12GiB.

| run | dtype | extra | tok/s | notes |
|---|---|---|---:|---|
| tiny_baseline | bf16 | - | 61,279 | fastest baseline |
| tiny_f16 | f16 | - | 59,202 | slower than bf16 |
| tiny_sparse | bf16 | `--sparse_loss` | 75,048 | +22% vs baseline |
| tiny_sparse_lbl64_512 | bf16 | `--sparse_loss --label_bucket_sizes 64,128,256,512` | 76,707 | best so far |
| tiny_sparse_lbl32_256 | bf16 | `--sparse_loss --label_bucket_sizes 32,64,128,256` | 76,293 | slightly slower |
| tiny_sparse_lbl64_512_p4 | bf16 | `--sparse_loss --label_bucket_sizes 64,128,256,512 --prefetch_batches 4` | 77,814 | new best |
| tiny_sparse_lbl64_512_p8 | bf16 | `--sparse_loss --label_bucket_sizes 64,128,256,512 --prefetch_batches 8` | 77,635 | slightly below p4 |
| tiny_sparse_lbl64_512_p4_paired | bf16 | `--sparse_loss --label_bucket_sizes 64,128,256,512 --prefetch_batches 4 --paired_heads` | 83,911 | best (arch change) |
| tiny_sparse_lbl64_512_p4_shuffle512 | bf16 | `--sparse_loss --label_bucket_sizes 64,128,256,512 --prefetch_batches 4 --shuffle_buffer 512` | 73,844 | quality-preserving shuffle |
| tiny_sparse_lbl64_512_p4_shuffle2048 | bf16 | `--sparse_loss --label_bucket_sizes 64,128,256,512 --prefetch_batches 4 --shuffle_buffer 2048` | 74,992 | quality-preserving shuffle |
| tiny_sparse_lbl64_512_p4_shuffle2048_ids | bf16 | pretokenized ids file + `--shuffle_buffer 2048` | 69,765 | slower (I/O larger) |
| tiny_sparse_lbl64_512_p4_shuffle2048_paired | bf16 | `--paired_heads` + quality-preserving shuffle | 81,111 | best (quality-checked) |
| tiny_sparse_lbl64_512_p4_shuffle2048_paired_relu2 | bf16 | `--paired_heads --hidden_act relu2` | 77,430 | slower |
| tiny_sparse_lbl64_512_p4_shuffle2048_paired_qknorm | bf16 | `--paired_heads --qk_norm` | 72,723 | slower |
| tiny_sparse_lbl64_512_p4_shuffle2048_paired_bs72 | bf16 | `--paired_heads --batch_size 72` | 73,499 | slower |
| tiny_sparse_lbl64_512_p4_shuffle2048_paired_vmix | bf16 | `--paired_heads --value_mix 0.1` | 76,404 | slower |
| tiny_sparse_lbl64_512_p4_shuffle2048_paired_embedskip | bf16 | `--paired_heads --embed_skip_scale 0.1` | 78,866 | slower |
| tiny_sparse_paired_bin_labels | bf16 | bin+labels (mmap) | 69,059 | slower |
| tiny_sparse_paired_bin_labels_mem | bf16 | bin+labels (memory) | 75,410 | slower |
| tiny_sparse_paired_bin_nolabel | bf16 | bin no-label (mmap) | 76,302 | slower |
| tiny_sparse_paired_bin_nolabel_mem | bf16 | bin no‑label (memory) | 81,311 | ~= best |
| tiny_sparse_noclip | bf16 | `--sparse_loss --grad_clip 0` | - | compile error (unordered_map::at) |
| paired_prefetch2 | bf16 | `--paired_heads --prefetch_batches 2` | 78,449 | slower |
| paired_prefetch6 | bf16 | `--paired_heads --prefetch_batches 6` | 79,637 | slower |
| paired_prefetch8 | bf16 | `--paired_heads --prefetch_batches 8` | 79,134 | slower |
| paired_bucket256_384_512 | bf16 | `--paired_heads --bucket_sizes 256,384,512` | 72,152 | slower |
| paired_label32_512 | bf16 | `--paired_heads --label_bucket_sizes 32,64,128,256,512` | 78,460 | slower |
| paired_shuffle1024 | bf16 | `--paired_heads --shuffle_buffer 1024` | 78,723 | slower |
| paired_shuffle4096 | bf16 | `--paired_heads --shuffle_buffer 4096` | 82,287 | slightly faster |
| bin2d_mmap | bf16 | bin2d + mmap | 80,645 | slower |
| bin2d_mem | bf16 | bin2d + memory | 83,738 | faster than jsonl |
| bin2d_mem_shuffle0 | bf16 | bin2d + memory + shuffle=0 | 80,844 | fastest (no shuffle) |
| bin2d_mem_shuffle512 | bf16 | bin2d + memory + shuffle=512 | 83,893 | best w/ shuffle |
| bin2d_mem_shuffle512_b72 | bf16 | bin2d + memory + shuffle=512 + batch=72 | 87,038 | new best (needs 14GB) |
| bin2d_mem_shuffle512_b80 | bf16 | bin2d + memory + shuffle=512 + batch=80 | 86,304 | slightly below b72 |
| bin2d_mem_shuffle512_b72_param | bf16 | b72 + optim_state_dtype=param | 85,699 | slower than default |
| bin2d_mem_shuffle512_b72_lion | bf16 | b72 + optimizer=lion | 79,541 | slower |
| bin2d_mem_shuffle512_b68 | bf16 | b68 + shuffle=512 | 77,454 | slower |
| bin2d_mem_shuffle512_b70 | bf16 | b70 + shuffle=512 | 77,598 | slower |
| bin2d_mem_shuffle512_b76 | bf16 | b76 + shuffle=512 | 79,894 | slower |
| bin2d_mem_shuffle512_b74 | bf16 | b74 + shuffle=512 | 78,497 | slower |
| bin2d_mem_shuffle512_b72_p3 | bf16 | b72 + prefetch=3 | 80,253 | slower |
| bin2d_mem_shuffle512_b72_p5 | bf16 | b72 + prefetch=5 | 79,339 | slower |
| bin2d_mem_shuffle512_b72_s256 | bf16 | b72 + shuffle=256 | 80,264 | slower |
| bin2d_mem_shuffle512_b72_s768 | bf16 | b72 + shuffle=768 | 82,423 | slower than s512 |
| bin2d_mem_shuffle384_b72 | bf16 | b72 + shuffle=384 | 79,081 | slower |
| bin2d_mem_shuffle640_b72 | bf16 | b72 + shuffle=640 | 77,095 | slower |
| bin2d_mem_shuffle512_b72_adafactor | bf16 | b72 + optimizer=adafactor | 65,255 | much slower |
| bin2d_mem_shuffle1024 | bf16 | bin2d + memory + shuffle=1024 | 78,870 | slower |
| bin2d_mem_shuffle2048 | bf16 | bin2d + memory + shuffle=2048 | 81,024 | slower |
| bin2d_mem_shuffle4096 | bf16 | bin2d + memory + shuffle=4096 | 57,929 | outlier slow |
| bin2d_mem_shuffle512_p2 | bf16 | bin2d + memory + shuffle=512 + prefetch=2 | 79,730 | slower |
| bin2d_mem_no_compile | bf16 | bin2d + memory + no compile | 66,227 | slower |

Epoch estimate (samples=1,214,724, tokens/epoch=621,938,688):
- 61,279 tok/s → ~2.82 h / epoch
- 75,048 tok/s → ~2.30 h / epoch
- 76,707 tok/s → ~2.25 h / epoch
- 77,814 tok/s → ~2.22 h / epoch
- 83,893 tok/s → ~2.06 h / epoch
- 87,038 tok/s → ~1.98 h / epoch

## Quality sanity check (sft_mini_512, seq_len=512, batch=8)
- 100 batches: baseline avg_loss=6.2759, paired_heads avg_loss=5.9880
- 1000 batches: baseline avg_loss=6.5693, paired_heads avg_loss=6.3673
- 1000-step train + 1000-batch eval (shuffle=4096): baseline avg_loss=3.8709, paired_heads avg_loss=3.8759

| kind | batch | accum | tok/s avg | step_ms avg | mem_peak MiB | mem_limit MiB | ok |
|---|---:|---:|---:|---:|---:|---:|:--:|
| grid_acc1 | 18 | 1 | 11835 | 618.5 | 6996 | - | True |
| grid_acc1 | 20 | 1 | 11840 | 695.4 | 7687 | - | True |
| grid_acc1 | 22 | 1 | 12682 | 712.9 | 8239 | - | True |
| grid_acc1 | 24 | 1 | 12871 | 771.8 | 8949 | - | True |
| grid_acc1 | 26 | 1 | 12612 | 848.3 | 9661 | - | True |
| grid_acc1 | 28 | 1 | 12760 | 893.4 | 10176 | - | True |
| grid_acc1 | 30 | 1 | 12904 | 957.0 | 10873 | - | True |
| grid_acc1 | 32 | 1 | 12615 | 1032.5 | 11570 | - | True |
| grid_acc1 | 34 | 1 | 12819 | 1093.5 | 12174 | - | True |
| grid_acc2 | 18 | 2 | 12846 | 1143.3 | 12547 | - | True |
| grid_acc2 | 20 | 2 | 13016 | 1268.6 | 13887 | - | True |
| grid_acc2 | 22 | 2 | 13085 | 1391.4 | 15174 | - | True |
| grid_acc2 | 24 | 2 | 12206 | 1750.7 | 16450 | - | True |
| grid_acc2 | 26 | 2 | 12926 | 1726.4 | 17644 | - | True |
| grid_acc2 | 28 | 2 | 12418 | 1978.9 | 18970 | - | True |
| memlimit_12g | 16 | 1 | 12288 | 522.4 | 6477 | 12000 | True |
| memlimit_12g | 24 | 1 | 12816 | 770.7 | 8949 | 12000 | True |
| memlimit_12g | 32 | 1 | 11692 | 1170.7 | 11570 | 12000 | True |
| memlimit_12g | 40 | 1 | 11437 | 1610.9 | 12986 | 12000 | True |
| memlimit_12g | 48 | 1 | 11092 | 1944.2 | 15502 | 12000 | True |
| long_50 | 24 | 2 | 14858 | - | 16450 | - | True |
| long_50 | 32 | 1 | 13781 | - | 11570 | 12000 | True |


## Long Runs (time-focused)
| name | batch | accum | steps | tok/s avg (>=40) | mem_peak MiB | mem_limit MiB | elapsed s | ok |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| long_200_b24_a2 | 24 | 2 | 200 | 15422 | 16450 | - | 316.1 | True |
| long_200_b32_a1_12g | 32 | 1 | 200 | 15331 | 11570 | 12000 | 227.2 | True |
| long_100_b30_a1 | 30 | 1 | 100 | 15143 | 10873 | - | 103.3 | True |
| long_100_b22_a2 | 22 | 2 | 100 | 16197 | 15174 | - | 143.7 | True |

## Epoch Time Estimates (sft_mini_512.jsonl)
Assumes `samples=1,214,724` and `seq_len=512`, so tokens/epoch = 621,938,688.

| batch | accum | tok/s avg | mem_peak MiB | est hours / epoch |
|---:|---:|---:|---:|---:|
| 24 | 2 | 15,422 | 16,450 | 11.20 |
| 32 | 1 | 15,331 | 11,570 | 11.27 |
| 30 | 1 | 15,143 | 10,873 | 11.41 |
| 22 | 2 | 16,197 | 15,174 | 10.67 |

## Final Short-Run Picks (time-first, 4-step)
Measured without `--profile_timing` (lower overhead), `max_steps=4`, skip first step for tok/s avg.

| batch | accum | prefetch | shuffle_buffer | tok/s avg | mem_peak MiB | notes |
|---:|---:|---:|---:|---:|---:|---|
| 22 | 2 | 4 | 0 | 13,439 | 15,174 | fastest (no shuffle) |
| 22 | 2 | 4 | 512 | 13,358 | 15,174 | near-best, with shuffle |
| 32 | 1 | 0 | 512 | 12,930 | 11,570 | 12GiB-safe |


## Prefetch + Shuffle Buffer Tuning (short runs)
| name | batch | accum | prefetch | shuffle_buffer | tok/s avg | mem_peak MiB | ok |
|---|---:|---:|---:|---:|---:|---:|:--:|
| b24_a2 | 24 | 2 | 0 | 0 | 7584 | 16450 | True |
| b24_a2 | 24 | 2 | 0 | 512 | 13120 | 16450 | True |
| b24_a2 | 24 | 2 | 0 | 2048 | 13479 | 16450 | True |
| b24_a2 | 24 | 2 | 2 | 0 | 13330 | 16450 | True |
| b24_a2 | 24 | 2 | 2 | 512 | 13221 | 16450 | True |
| b24_a2 | 24 | 2 | 2 | 2048 | 13419 | 16450 | True |
| b24_a2 | 24 | 2 | 4 | 0 | 13718 | 16450 | True |
| b24_a2 | 24 | 2 | 4 | 512 | 13409 | 16450 | True |
| b24_a2 | 24 | 2 | 4 | 2048 | 12987 | 16450 | True |
| b32_a1 | 32 | 1 | 0 | 0 | 13063 | 11570 | True |
| b32_a1 | 32 | 1 | 0 | 512 | 13688 | 11570 | True |
| b32_a1 | 32 | 1 | 0 | 2048 | 13267 | 11570 | True |
| b32_a1 | 32 | 1 | 2 | 0 | 13369 | 11570 | True |
| b32_a1 | 32 | 1 | 2 | 512 | 13303 | 11602 | True |
| b32_a1 | 32 | 1 | 2 | 2048 | 13518 | 11570 | True |
| b32_a1 | 32 | 1 | 4 | 0 | 13469 | 11570 | True |
| b32_a1 | 32 | 1 | 4 | 512 | 13230 | 11570 | True |
| b32_a1 | 32 | 1 | 4 | 2048 | 13086 | 11570 | True |


## Fine Grid (acc=2 around bs 20-24, acc=1 around bs 28-32)
| kind | batch | accum | prefetch | shuffle_buffer | tok/s avg | mem_peak MiB | ok |
|---|---:|---:|---:|---:|---:|---:|:--:|
| tune_gridC_acc2 | 20 | 2 | 2 | 0 | 11546 | 13887 | True |
| tune_gridC_acc2 | 20 | 2 | 2 | 512 | 11660 | 13887 | True |
| tune_gridC_acc2 | 20 | 2 | 2 | 2048 | 11574 | 13887 | True |
| tune_gridC_acc2 | 20 | 2 | 4 | 0 | 11642 | 13887 | True |
| tune_gridC_acc2 | 20 | 2 | 4 | 512 | 11421 | 13887 | True |
| tune_gridC_acc2 | 20 | 2 | 4 | 2048 | 11951 | 13887 | True |
| tune_gridC_acc2 | 22 | 2 | 2 | 0 | 11539 | 15174 | True |
| tune_gridC_acc2 | 22 | 2 | 2 | 512 | 11603 | 15174 | True |
| tune_gridC_acc2 | 22 | 2 | 2 | 2048 | 11680 | 15174 | True |
| tune_gridC_acc2 | 22 | 2 | 4 | 0 | 12448 | 15174 | True |
| tune_gridC_acc2 | 22 | 2 | 4 | 512 | 13153 | 15174 | True |
| tune_gridC_acc2 | 22 | 2 | 4 | 2048 | 13134 | 15174 | True |
| tune_gridC_acc2 | 24 | 2 | 2 | 0 | 12501 | 16450 | True |
| tune_gridC_acc2 | 24 | 2 | 2 | 512 | 13673 | 16450 | True |
| tune_gridC_acc2 | 24 | 2 | 2 | 2048 | 13828 | 16450 | True |
| tune_gridC_acc2 | 24 | 2 | 4 | 0 | 13847 | 16450 | True |
| tune_gridC_acc2 | 24 | 2 | 4 | 512 | 13870 | 16450 | True |
| tune_gridC_acc2 | 24 | 2 | 4 | 2048 | 13925 | 16450 | True |
| tune_gridD_acc1 | 28 | 1 | 0 | 0 | 13459 | 10176 | True |
| tune_gridD_acc1 | 28 | 1 | 0 | 512 | 13634 | 10176 | True |
| tune_gridD_acc1 | 28 | 1 | 2 | 0 | 13783 | 10176 | True |
| tune_gridD_acc1 | 28 | 1 | 2 | 512 | 13708 | 10176 | True |
| tune_gridD_acc1 | 30 | 1 | 0 | 0 | 13753 | 10873 | True |
| tune_gridD_acc1 | 30 | 1 | 0 | 512 | 13658 | 10873 | True |
| tune_gridD_acc1 | 30 | 1 | 2 | 0 | 13824 | 10873 | True |
| tune_gridD_acc1 | 30 | 1 | 2 | 512 | 13668 | 10873 | True |
| tune_gridD_acc1 | 32 | 1 | 0 | 0 | 13564 | 11570 | True |
| tune_gridD_acc1 | 32 | 1 | 0 | 512 | 13605 | 11570 | True |
| tune_gridD_acc1 | 32 | 1 | 2 | 0 | 13673 | 11570 | True |
| tune_gridD_acc1 | 32 | 1 | 2 | 512 | 13318 | 11570 | True |

## Smoke Runs (2026-01-22)
- Train (init_from step_00151841, preset custom, bs=24 acc=2, prefetch=4, shuffle=2048, steps=4): tok/s 7363 → 6261, mem_peak ≈ 20444 MiB.
- Train baseline_silu (init_from step_00151841, bs=24 acc=2, prefetch=4, shuffle=2048, steps=6): tok/s step2-6 avg ≈ 5,365 (5018 → 6247).
- Bench (speculator/infer/mlx/bench.py): baseline 398.77 tok/s, spec_len=2 396.41 tok/s (0.99x), acceptance 0.54.

## GPU Throttle Runs (2026-01-23)
| kind | batch | accum | prefetch | shuffle_buffer | tok/s avg | mem_limit MiB | step_sleep_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bin2d_mem_shuffle512_b64_p2_sleep3 | 64 | 1 | 2 | 512 | 73320 | 12000 | 3 |
| bin2d_mem_shuffle512_b60_p2_sleep3 | 60 | 1 | 2 | 512 | 73795 | 12000 | 3 |
| bin2d_mem_shuffle512_b60_p1_sleep3 | 60 | 1 | 1 | 512 | 75033 | 12000 | 3 |
| bin2d_mem_shuffle512_b72_sleep0p5 | 72 | 1 | 4 | 512 | 77786 | 14000 | 0.5 |
| bin2d_mem_shuffle512_b72_sleep1_e2 | 72 | 1 | 4 | 512 | 75754 | 14000 | 1/2 |
| bin2d_mem_shuffle512_b72_sleep2_e8 | 72 | 1 | 4 | 512 | 79446 | 14000 | 2/8 |
| bin2d_mem_shuffle512_b72_sleep5_e10 | 72 | 1 | 4 | 512 | 79236 | 14000 | 5/10 |
| bin2d_mem_shuffle512_b72_sleep0p2_e8 | 72 | 1 | 4 | 512 | 79588 | 14000 | 0.2/8 |
| bin2d_mem_shuffle512_b72_sleep0p05_e16 | 72 | 1 | 4 | 512 | 85884 | 14000 | 0.05/16 |
| bin2d_mem_shuffle512_b72_nosleep | 72 | 1 | 4 | 512 | 83390 | 14000 | 0 |
| bin2d_mem_shuffle512_b80_nosleep | 80 | 1 | 4 | 512 | 83863 | 15000 | 0 |
| bin2d_mem_shuffle512_b84_nosleep | 84 | 1 | 4 | 512 | 70976 | 16000 | 0 |
| bin2d_mem_shuffle512_b72_nosleep_retest | 72 | 1 | 4 | 512 | 80723 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p6_s512 | 72 | 1 | 6 | 512 | 85811 | 14000 | 0 |
| bin2d_mem_shuffle768_b72_p4 | 72 | 1 | 4 | 768 | 85698 | 14000 | 0 |
| bin2d_mem_shuffle768_b72_p6 | 72 | 1 | 6 | 768 | 85416 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_nocompile | 72 | 1 | 4 | 512 | 67767 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_nocompile_noopt | 72 | 1 | 4 | 512 | 71678 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_nometa | 72 | 1 | 4 | 512 | 77334 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_nosleep_retest2 | 72 | 1 | 4 | 512 | 83419 | 14000 | 0 |
| bin2d_mem_shuffle1024_b72_p6 | 72 | 1 | 6 | 1024 | 80394 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p8_s512 | 72 | 1 | 8 | 512 | 84287 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p5_s512 | 72 | 1 | 5 | 512 | 85226 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p7_s512 | 72 | 1 | 7 | 512 | 84752 | 14000 | 0 |
| bin2d_mem_shuffle640_b72_p6 | 72 | 1 | 6 | 640 | 75480 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p6_s512_param | 72 | 1 | 6 | 512 | 82161 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p6_s512_vm0p1 | 72 | 1 | 6 | 512 | 73821 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p6_s512_vm0p05 | 72 | 1 | 6 | 512 | 78200 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p6_s512_f16 | 72 | 1 | 6 | 512 | 74121 | 14000 | 0 |
| jsonl_ids_b144_p6_s256 | 144 | 1 | 6 | 512 | 82729 | 14000 | 0 |
| jsonl_ids_b96_p6_s384 | 96 | 1 | 6 | 512 | 84608 | 14000 | 0 |
| bin2d_mem_shuffle512_b72_p6_s512_acc2 | 72 | 2 | 6 | 512 | 80841 | 14000 | 0 |
