# VGGT-Omega tokenizer V2 实验说明

## 改动目的

V1 的最佳 checkpoint 位于 epoch 161，val ADE 约 `0.343 m`；继续到 epoch 499 后约为
`0.352 m`。训练集仍能继续拟合，action token 也没有塌缩，所以当前首要问题更像运动建模和
泛化，而不是参数量不足。

V2 做了以下调整：

1. decoder 预测车体系 `vx / vy / yaw_rate`，再乘每个 LiDAR 点的实际 `delta_t` 做 SE(2)
   积分，避免把轻微不规则的 10 Hz 时间间隔当作固定 displacement；
2. 四个 1 秒 action token 通过小型 Transformer 交换上下文；
3. 增加 body velocity、yaw rate、acceleration、jerk 和 token 边界连续性监督；
4. 训练采样对 turn/fast 适度增权、stationary 降权；
5. 根据 val ADE 早停，checkpoint 会保存并恢复 patience 计数。

## 三组对照

| 实验 | 配置 | cache | 参数量 | 要回答的问题 |
| --- | --- | --- | ---: | --- |
| motion | `nuscenes_vggt_omega_front_4s_v2_motion.yaml` | 现有 mean | 2.05 M | 新运动建模是否优于 V1 |
| large | `nuscenes_vggt_omega_front_4s_v2_motion_large.yaml` | 现有 mean | 5.69 M | 单纯增加容量是否有效 |
| rich | `nuscenes_vggt_omega_front_4s_v2_rich_register.yaml` | 新 full-register | 2.25 M | register mean 是否丢信息 |

先训练 motion。large 和 rich 都以 motion 为控制组，只改变一个主要因素；请勿共用 output
目录，也不要从其他结构的 checkpoint resume。

## 训练命令

```bash
torchrun --standalone --nproc_per_node=1 tools/training/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_motion.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v2_motion

torchrun --standalone --nproc_per_node=1 tools/training/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_motion_large.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v2_motion_large
```

## Rich-register cache 与训练

rich 配置会额外保存 `[5,16,2048]` 的 register hidden，预计 train+val 约需 7–8 GB。命令支持
校验后断点续跑：

```bash
python tools/features/cache_vggt_omega_features.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_rich_register.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_train_rich \
  --shard-size 128 --num-workers 2

python tools/features/cache_vggt_omega_features.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_rich_register.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_val_rich \
  --shard-size 128 --num-workers 2

torchrun --standalone --nproc_per_node=1 tools/training/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_rich_register.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v2_rich_register
```

## 训练后需要比较

用各自 config 和 `best.pt` 运行 `tools/evaluation/evaluate_tokenizer.py`，重点保留：

- overall ADE/FDE、keyframe ADE；
- stationary / straight slow / straight fast / turn 的 ADE/FDE；
- speed、yaw-rate、acceleration、jerk MAE；
- best epoch，以及 train/validation loss 的间隙；
- TensorBoard 中相同类型失败样本的 2×2 图。

判断标准：motion 明显提升说明瓶颈主要在运动参数化；large 进一步提升才支持继续加参数；rich
提升而 large 不提升则说明 mean cache 是主要信息瓶颈。

## 已完成结果与 V3

V2 motion 全量 val ADE 为 `0.3275 m`。原 attention-rich 模型为 `0.3497 m`，且四个
register query 的 attention map cosine 接近 `1.0`，说明 summary query 塌缩。另一方面，打乱
rich register 会令 ADE 恶化到 `0.556 m`，并且 `0.7 × motion + 0.3 × rich` 的轨迹融合可达到
约 `0.3203 m`，证明 full-register 仍包含互补信息。

V3 不再用多个可交换 query。它保留 motion 的 mean 路径，用固定 register slot 的低秩展开
预测残差，并用零初始化逐通道 gate 融合：

```text
frame_geometry = frozen-compatible motion_mean_path
               + tanh(zero_initialized_gate) * centered_register_residual
```

推荐先跑严格结构消融：

```bash
torchrun --standalone --nproc_per_node=1 tools/training/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v3_residual_register.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v3_residual_register
```

该配置自动从 `output/itae_vggt_omega_v2_motion/best.pt` 初始化。主路径学习率是 residual
adapter 的 `0.1` 倍；新 output 已有 checkpoint 时优先正常 resume，不会再次 warm-start。

第二组仅用于判断速度趋势重采样的收益：

```bash
torchrun --standalone --nproc_per_node=1 tools/training/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v3_residual_register_dynamics.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v3_residual_register_dynamics
```

它把 stationary 权重恢复到 `1.0`，并将首尾 1 秒平均速度相差超过 `0.5 m/s` 的 accelerate /
decelerate 窗口各加权 `1.25`。必须在第一组之后再跑，避免同时改变结构和采样后无法归因。

## V3 与 dynamics 结论、V3.1

V3 普通版和 dynamics 版都没有替换初始化的 motion `best.pt`。Dynamics last 的全量 ADE 为
`0.3374 m`；它改善 accelerating FDE，但 steady 和 decelerating 明显退化。两组训练的 gate
mean 都只有约 `9e-4`，且跨样本打乱 centered register 几乎不改变输出，说明 full-register
分支仍未学到输入相关修正。

V3.1 用“零输出层”替代“零 gate”：

- residual 最后额外增加一个 zero-initialized linear，因此初始重建仍与 motion 逐位相同；
- gate 初始化为 `3.0`，经 `tanh` 后约为 `0.995`，输出层从第一个 step 就有有效梯度；
- 前 10 epoch 将 motion 主路径设为 eval 且冻结，只训练 residual adapter；
- 之后主路径以 residual 学习率的 `0.05` 倍联合微调；
- `best_trained.pt` 独立跟踪训练态，早停也跟踪它；`best.pt` 仍只在超过安全基线时更新；
- `resume: auto` 明确优先 `last.pt`，不会误选更新时间更晚的 best checkpoint。

```bash
torchrun --standalone --nproc_per_node=1 tools/training/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v3_1_zero_init_residual.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v3_1_zero_init_residual
```

该实验重新使用 V2 motion 的采样分布，不启用 dynamics 权重。只有 trained residual 在
register shuffle 消融中产生明确退化后，才值得重新引入加速/减速采样或模型容量实验。

如果 `best_trained.pt` 与初始化表现接近但误差互补，可生成预先固定的 0.5 权重 soup：

```bash
python tools/analysis/interpolate_tokenizer_checkpoints.py \
  --left output/itae_vggt_omega_v3_1_zero_init_residual/initial.pt \
  --right output/itae_vggt_omega_v3_1_zero_init_residual/best_trained.pt \
  --alpha 0.5 \
  --output output/itae_vggt_omega_v3_1_zero_init_residual/soup_0.5.pt
```

该工具严格检查 state key、shape、dtype 和非浮点 buffer；输出只用于 evaluation/export，不能
作为完整 optimizer resume checkpoint。不要在 val 上搜索大量 alpha 后再把同一 val 当作无偏
结果；`0.5` 应作为事先固定的诊断候选。

## V3.1 结果与 V4 output residual

V3.1 `best_trained.pt` 的全量 ADE/FDE 为 `0.3271/0.6315 m`，motion 初始化为
`0.3275/0.6364 m`，仅有轻微改善。其 adapter 激活的样本相关能量占比只有 `0.19%`；打乱
centered register 后 ADE 从 `0.327121` 变为 `0.327086 m`，说明改善来自通用速度偏置，而非
视觉条件。解冻主干后的 last ADE 又退化至 `0.3306 m`。

V4 针对该失败模式改变表示和监督位置：

```text
motion_token[128] = frozen V2 mean-register encoder
base_increments    = frozen V2 decoder(motion_token)

residual_token[64] = zero-output(
    full centered registers + relative VGGT pose direction/rotation
)
residual_increments = bounded 10 Hz rate decoder(residual_token)

action_token       = concat(motion_token, residual_token)  # 192-D
reconstruction     = integrate(base_increments + residual_increments)
```

- motion encoder/decoder 全程冻结，避免后期破坏已验证的 `0.3275 m` 基线；
- residual token 的最后一层零初始化且无 bias，初始化重建与 motion 逐位相同；
- 显式监督 `GT body rates - frozen motion body rates`；
- batch-mean residual penalty 抑制跨样本共享偏置；
- interval residual pattern 与 GT residual 使用对称 contrastive alignment；
- 同一个 batch 内循环错配 full-register 与 pose，ranking loss 要求错配条件误差更高；
- `pose_enc` 只提供相对 camera center、单位平移方向、log-distance 和 relative rotation 6D，
  不假设单目 pose translation 已经是米制；
- validation 改为 sample-weighted FP32 聚合，使 checkpoint 排名与独立 evaluator 一致；恢复已
  满足 patience 的 checkpoint 不再多训练一轮。

固定 8 个真实 cache 样本的 300-step 可学习性测试中，正常条件 ADE 从 `0.1297` 降至
`0.0136 m`，错配 register+pose 后为 `0.1723 m`，证明新路径能记住样本相关修正。该结果只是
梯度/条件性验收，不代表 val 泛化结果。

```bash
torchrun --standalone --nproc_per_node=1 tools/training/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v4_output_residual.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v4_output_residual
```

验收不只看 ADE/FDE，还必须同时满足：`normal < base_motion < shuffled_condition`，并分别检查
`shuffled_register` 与 `shuffled_pose` 来判断信息来源；同时要求
TensorBoard 的 `condition/shuffle_error_gap` 为正、`condition/prediction_l2_m` 明显非零。若
ADE 改善但 shuffle 不退化，仍视为视觉条件失败，不继续加参数量。
