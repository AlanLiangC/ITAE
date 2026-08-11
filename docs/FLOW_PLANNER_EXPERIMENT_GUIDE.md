# PE Flow Planner 对照实验流程

本实验固定当前单张 `CAM_FRONT` 作为输入，冻结 `PE-Spatial-B16-512` 提取视觉条件，分别训练：

- raw planner：flow matching 直接生成 `[40,3]` 轨迹；
- token planner：flow matching 生成 V4 `[4,192]` action token，再由冻结 decoder 重建轨迹。

两者使用同一 PE cache、planner core、optimizer steps、batch size 和 5-NFE Euler sampler。下面
命令均在项目根目录执行，环境为 `py312torch210cu126`。

## 0. 环境检查

```bash
conda activate py312torch210cu126
pip install -e '.[pe,dev,viz]'

test -f third_party/perception_models/core/vision_encoder/pe.py
sha256sum /home/alan/AlanLiang/Projects/pure_checkpoints/PE-Spatial-B16-512/PE-Spatial-B16-512.pt
sha256sum output/itae_vggt_omega_v4_output_residual/best.pt
```

期望两个权重哈希分别为：

```text
86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5
78f11556263e56830a1604dada548a30643133c03bd3673ab695cb4c63d2f9a0
```

`third_party/perception_models` 固定使用 commit
`3e352cca660658d4b5c90f42a7808b11469e4c66`。训练阶段不会联网下载权重。

## 1. 生成 scene-disjoint planner split

```bash
python tools/data/split_planner_manifests.py \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --final-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output data/manifests/nuscenes_planner_front_4s \
  --validation-fraction 0.1 \
  --seed 42
```

固定结果：

| split | samples | scenes | 用途 |
|---|---:|---:|---|
| planner train | 17,477 | 630 | 拟合 planner 和 normalization |
| planner validation | 1,893 | 70 | checkpoint 选择与收敛曲线 |
| final eval | 4,017 | 150 | 最终 paired evaluation |

## 2. 缓存当前单帧 PE 特征

三个 split 必须使用同一个 config。cache 支持校验后断点续跑：

```bash
for split in train validation final_eval; do
  python tools/features/cache_planner_vision_features.py \
    --config configs/planner/nuscenes_flow_raw_pe_4s.yaml \
    --manifest data/manifests/nuscenes_planner_front_4s/planner_${split}.jsonl \
    --output /home/alan/AlanLiang/Dataset/itae_planner_cache/pe_spatial_b16_512/${split} \
    --batch-size 8 \
    --shard-size 256 \
    --num-workers 4
done
```

每个样本缓存 `[64,768]` FP16 token，对应 PE `32×32` patch grid 池化后的 `8×8` grid。
RTX 4070 Laptop 实测约 9–16 images/s，总磁盘约 2.3 GiB。

## 3. 导出 V4 action-token teacher targets

planner train/validation 都是原 tokenizer-train manifest 的子集，所以需要用 `--source-manifest`
把子集 sample token 映射回原 rich cache：

```bash
mkdir -p /home/alan/AlanLiang/Dataset/itae_planner_cache/v4_action_targets

for split in train validation; do
  python tools/features/cache_tokenizer_action_targets.py \
    --tokenizer-config configs/nuscenes_vggt_omega_front_4s_v4_output_residual.yaml \
    --source-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
    --manifest data/manifests/nuscenes_planner_front_4s/planner_${split}.jsonl \
    --feature-cache /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_train_rich \
    --checkpoint output/itae_vggt_omega_v4_output_residual/best.pt \
    --output /home/alan/AlanLiang/Dataset/itae_planner_cache/v4_action_targets/${split}.safetensors \
    --batch-size 64
done

python tools/features/cache_tokenizer_action_targets.py \
  --tokenizer-config configs/nuscenes_vggt_omega_front_4s_v4_output_residual.yaml \
  --source-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --manifest data/manifests/nuscenes_planner_front_4s/planner_final_eval.jsonl \
  --feature-cache /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_val_rich \
  --checkpoint output/itae_vggt_omega_v4_output_residual/best.pt \
  --output /home/alan/AlanLiang/Dataset/itae_planner_cache/v4_action_targets/final_eval.safetensors \
  --batch-size 64
```

导出会打印 `GT token -> frozen decoder` 的 oracle ADE/FDE，并在相邻 JSON 中保存 manifest、
rich cache、tokenizer config/checkpoint hash 和严格 sample order。

## 4. 正式训练

先跑 seed 42，确认 TensorBoard 正常，再启动另外两个 seed：

```bash
for seed in 42 43 44; do
  torchrun --standalone --nproc_per_node=1 tools/training/train_flow_planner.py \
    --config configs/planner/nuscenes_flow_raw_pe_4s.yaml \
    --train-manifest data/manifests/nuscenes_planner_front_4s/planner_train.jsonl \
    --val-manifest data/manifests/nuscenes_planner_front_4s/planner_validation.jsonl \
    --output output/planner_raw_pe_seed${seed} \
    --seed ${seed}

  torchrun --standalone --nproc_per_node=1 tools/training/train_flow_planner.py \
    --config configs/planner/nuscenes_flow_token_v4_pe_4s.yaml \
    --train-manifest data/manifests/nuscenes_planner_front_4s/planner_train.jsonl \
    --val-manifest data/manifests/nuscenes_planner_front_4s/planner_validation.jsonl \
    --output output/planner_token_v4_pe_seed${seed} \
    --seed ${seed}
done
```

每个实验固定 50,000 optimizer steps。`best.pt` 按 EMA 模型在 validation 上 5-NFE decoded ADE
选择；`last.pt` 自动断点恢复。不要比较 raw/token flow-loss 数值，它们不在同一空间。

```bash
tensorboard --logdir_spec \
raw42:output/planner_raw_pe_seed42/tensorboard,\
token42:output/planner_token_v4_pe_seed42/tensorboard \
--port 6006
```

重点看 decoded `validation/metric/ade_m`、`validation/metric/fde_m`、梯度、LR 和
`training_history.jsonl`。

## 5. 5-NFE final evaluation

```bash
for seed in 42 43 44; do
  python tools/evaluation/evaluate_flow_planner.py \
    --config configs/planner/nuscenes_flow_raw_pe_4s.yaml \
    --manifest data/manifests/nuscenes_planner_front_4s/planner_final_eval.jsonl \
    --checkpoint output/planner_raw_pe_seed${seed}/best.pt \
    --output output/planner_eval/raw_seed${seed}

  python tools/evaluation/evaluate_flow_planner.py \
    --config configs/planner/nuscenes_flow_token_v4_pe_4s.yaml \
    --manifest data/manifests/nuscenes_planner_front_4s/planner_final_eval.jsonl \
    --checkpoint output/planner_token_v4_pe_seed${seed}/best.pt \
    --output output/planner_eval/token_seed${seed}
done
```

evaluator 强制 `steps=5` 且 `NFE=5`，保存：

- `metrics.json`：共同轨迹指标、motion buckets、配置/权重/hash/参数量；
- `predictions.safetensors`：逐样本预测、GT、ADE/FDE；
- `tensorboard/`：固定样本的当前帧 + GT/预测/overlay 2×2 页面；
- token 分支额外包含 oracle reconstruction、normalized token MSE 和 excess ADE/FDE。

## 6. 每个 seed 的 paired comparison

```bash
for seed in 42 43 44; do
  python tools/evaluation/compare_flow_planners.py \
    --raw-eval output/planner_eval/raw_seed${seed} \
    --token-eval output/planner_eval/token_seed${seed} \
    --raw-history output/planner_raw_pe_seed${seed}/training_history.jsonl \
    --token-history output/planner_token_v4_pe_seed${seed}/training_history.jsonl \
    --output output/planner_eval/comparison_seed${seed}.json
done
```

工具会拒绝不同 sample order、PE condition、planner core 或参数量相差超过 5% 的对比，并报告
validation ADE AUC、达到 1.0/0.75/0.5 m 的 seen samples，以及 scene-level paired bootstrap CI。

## 7. 汇总三个 seed

```bash
python tools/evaluation/summarize_flow_planner_seeds.py \
  --comparisons \
    output/planner_eval/comparison_seed42.json \
    output/planner_eval/comparison_seed43.json \
    output/planner_eval/comparison_seed44.json \
  --output output/planner_eval/three_seed_summary.json
```

最终用三 seed 的 decoded ADE/FDE mean ± std、AUC 和 paired CI 判断，不根据单次 smoke 或训练
loss 下结论。token 的 oracle floor 用来解释瓶颈，但不从正式预测指标中人为扣除。
