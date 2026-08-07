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
torchrun --standalone --nproc_per_node=1 tools/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_motion.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v2_motion

torchrun --standalone --nproc_per_node=1 tools/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_motion_large.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v2_motion_large
```

## Rich-register cache 与训练

rich 配置会额外保存 `[5,16,2048]` 的 register hidden，预计 train+val 约需 7–8 GB。命令支持
校验后断点续跑：

```bash
python tools/cache_vggt_omega_features.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_rich_register.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_train_rich \
  --shard-size 128 --num-workers 2

python tools/cache_vggt_omega_features.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_rich_register.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_val_rich \
  --shard-size 128 --num-workers 2

torchrun --standalone --nproc_per_node=1 tools/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s_v2_rich_register.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v2_rich_register
```

## 训练后需要比较

用各自 config 和 `best.pt` 运行 `tools/evaluate_tokenizer.py`，重点保留：

- overall ADE/FDE、keyframe ADE；
- stationary / straight slow / straight fast / turn 的 ADE/FDE；
- speed、yaw-rate、acceleration、jerk MAE；
- best epoch，以及 train/validation loss 的间隙；
- TensorBoard 中相同类型失败样本的 2×2 图。

判断标准：motion 明显提升说明瓶颈主要在运动参数化；large 进一步提升才支持继续加参数；rich
提升而 large 不提升则说明 mean cache 是主要信息瓶颈。
