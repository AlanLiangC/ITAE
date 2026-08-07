# ITAE: VGGT-Omega Vision Action Tokenizer

ITAE 将 nuScenes 的 5 张 `CAM_FRONT` keyframe（0–4 s、1 Hz）编码成 4 个相邻时间段的
action token，再从 token 重建输入视觉窗口内的 40 个 LiDAR ego pose（0.1–4.0 s、10 Hz）。

```text
5 × CAM_FRONT
  -> frozen VGGT-Omega Aggregator + CameraHead trunk
  -> 5 × frame geometry
  -> 4 × 128-D interval action token
  -> context-free SE(2) increment decoder
  -> 40 × [x forward, y left, yaw]
```

轨迹不是“视觉窗口之后的未来”：anchor 是第 1 张图，40 点覆盖 5 张输入图之间的同一段 4 s
时间。图像来自 1 Hz keyframe；轨迹来自 10 Hz LiDAR keyframe/sweep 的标定后 ego pose。

## 环境与权重

当前验证环境：

- conda：`py312torch210cu126`
- Python 3.12.13、PyTorch 2.10.0+cu126
- GPU：RTX 4070 Laptop 8 GB
- VGGT-Omega checkpoint：
  `/home/alan/AlanLiang/Projects/pure_checkpoints/VGGT-Omega/vggt_omega_1b_512.pt`
- checkpoint SHA256：
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`

安装：

```bash
conda activate py312torch210cu126
python -m pip install -e third_party/vggt-omega
python -m pip install -e ".[dev,viz]"
```

`third_party/vggt-omega` 使用 FAIR Noncommercial Research License，并带有 Acceptable Use
Policy。使用、发布 checkpoint 或衍生权重前，请确认当前自动驾驶研究用途满足其许可证要求。

## 数据与配置

正式配置是
[`configs/nuscenes_vggt_omega_front_4s.yaml`](configs/nuscenes_vggt_omega_front_4s.yaml)。
以下 setting 都可在 YAML 中修改：

- `data.frame_offsets_s`：视觉帧时间，默认 `[0,1,2,3,4]`；
- `data.future_horizon_s` / `trajectory_hz`：默认 4 s / 10 Hz；
- 图像与 LiDAR 时间误差、插值上限、采样策略；
- VGGT resize、checkpoint 与 cache 路径；
- action token 数量/维度、每个 token 的轨迹步数、decoder 维度；
- 四项 loss 权重、训练和 TensorBoard 设置。

已有 manifest：

```text
data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl  # 19,370
data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl    #  4,017
```

如需重建 manifest：

```bash
python tools/build_manifest.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --info data/nuscenes/nuscenes_interp_12Hz_infos_train.pkl \
  --output data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --report data/manifests/nuscenes_lidar10hz_front_4s_train.report.json
```

## 1. 先审计 VGGT 相机运动

在训练前必须检查 pose 的方向、frame order 和尺度：

```bash
python tools/inspect_vggt_camera_motion.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --output-dir output/vggt_camera_motion_audit/train

python tools/inspect_vggt_camera_motion.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --calibration output/vggt_camera_motion_audit/train/camera_motion_calibration.json \
  --output-dir output/vggt_camera_motion_audit/val
```

脚本会自动选静止、约 20 m、长直行和大转弯样本，输出 contact sheet、BEV 对照和 JSON。
第一条命令仅在 train 样本上拟合一个全局 scale，第二条将其原样用于 val，禁止逐 val window
对齐。此前 4 个 val 诊断样本的 translation direction cosine 为 `0.978`；该 pose 尺度只用于
表征审计，不会泄漏到 tokenizer。

轨迹点本身可继续用原始视频投影工具检查：

```bash
python tools/visualize_trajectory_video.py --help
```

## 2. 缓存冻结的 VGGT 特征

1B backbone 不参与常规 tokenizer 反向；先分别缓存 train/val：

```bash
python tools/cache_vggt_omega_features.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --output data/vggt_omega_cache/nuscenes_front_4s_train \
  --shard-size 128 --num-workers 2

python tools/cache_vggt_omega_features.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output data/vggt_omega_cache/nuscenes_front_4s_val \
  --shard-size 128 --num-workers 2
```

每个样本缓存 CameraHead trunk 后的 `camera_hidden [5,2048]`、register mean 和 pose。
`index.json` 固定 manifest/checkpoint/preprocess/third-party commit；每个 shard 有 SHA256。中断后用
同一命令可从最后一个已校验 shard 继续；metadata 或 shard checksum 不一致会直接报错。

本机实测约 3.2 windows/s，完整 23,387 个窗口约需 2 小时。

## 3. 训练与自动恢复

```bash
torchrun --standalone --nproc_per_node=1 \
  tools/train_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_vggt_omega_v1
```

默认 `train.resume: auto`：同一 output 下自动选修改时间最新的 `.pt`，恢复 model、optimizer、
scheduler、AMP scaler、epoch、global step 和 RNG。也可在 YAML 写 checkpoint 路径，或用
`--resume PATH`；`--no-resume` 强制新训练。旧 PE V2 checkpoint 与新图严格不兼容。

TensorBoard：

```bash
tensorboard --logdir output/itae_vggt_omega_v1/tensorboard --port 6006
```

训练会记录各 loss、ADE/FDE/yaw、运动导数、LR、gradient norm 和 action token
std/cosine。evaluation 按配置抽样生成 1200×900 的 2×2 页面：五张主视图、GT BEV、重建
BEV，以及逐时刻 position/body-increment error。即便训练读取 cache，val dataset 仍会为这些
页面读取原始图像。

## 4. 评估与 action latent 导出

```bash
python tools/evaluate_tokenizer.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --pose-calibration output/vggt_camera_motion_audit/train/camera_motion_calibration.json \
  --checkpoint output/itae_vggt_omega_v1/best.pt \
  --output output/itae_vggt_omega_v1/val_metrics.json

python tools/export_action_latents.py \
  --config configs/nuscenes_vggt_omega_front_4s.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --checkpoint output/itae_vggt_omega_v1/best.pt \
  --output output/itae_vggt_omega_v1/train_action_tokens.safetensors
```

评估 JSON 同时报告 tokenizer、始终静止、train-mean、train-calibrated VGGT pose 四组结果，
并按 stationary / straight slow / straight fast / turn 分桶。train manifest 与 pose calibration
都只作为固定 baseline，绝不在 val 上重新拟合。

导出的 latent shape 为 `[N,4,128]`，并包含训练 action expert 所需的 mean/std。decoder 只读取
action tokens 与 query 时间，不读取图像、VGGT hidden 或 GT。

## 验证状态

- 官方 VGGT `max_size` 预处理逐像素一致；
- camera branch checkpoint strict load 与 5 帧 BF16 forward 通过；
- 单窗口峰值显存约 6.02 GB；
- trainable adapter + decoder：1,657,476 参数；
- 4 类样本 overfit：500 step ADE `0.221 m`，最终 `0.117 m`；
- 随机 64 样本 overfit：750 step ADE `0.449 m`；
- `torchrun` 保存、自动恢复、TensorBoard 标量和 2×2 图片已做端到端 smoke test；
- 运行 `python -m pytest -q` 和 `python -m ruff check src tests tools` 做回归检查。

VS Code 已提供 camera audit、val cache、train、evaluation 和当前文件五个 debug configuration，
解释器均指向 `py312torch210cu126`。
