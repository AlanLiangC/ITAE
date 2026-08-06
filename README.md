# Vision-Aligned Action Tokenizer

面向端到端自动驾驶 action expert 的视觉对齐连续动作 tokenizer。默认配置把 4 s 输入
视觉窗口内 `(t0, t0+4s]` 的 `[40, 3] = [x, y, yaw]` 自车轨迹编码为 `[10, 64]`
action tokens，并让 latent 同时满足：

- 可从当前到未来的 PE-Spatial 视觉状态转移中提取；
- 可由纯轨迹编码器得到，并与视觉 latent 对齐；
- 不依赖任何视觉旁路，能够由 `Decoder(Z)` 独立还原轨迹；
- 可作为 diffusion/flow action expert 的生成目标。

当前版本包括：

- MMDetection3D 新旧格式及自定义 12 Hz nuScenes info 适配；
- 5 帧官方 CAM_FRONT keyframe `[0, 1, 2, 3, 4] s` 视觉窗口；
- 官方 `LIDAR_TOP sample_data -> ego_pose` 构建的 40 点、10 Hz 实测轨迹；
- `PE-Spatial-B16-512` patch-token 提取与离线缓存；
- 保留二维网格顺序的视觉转移编码器、轨迹编码器、共享无上下文 decoder；
- direct 和可微 unicycle 两种 trajectory head；
- 重建、动力学、物理、KL、跨模态对齐、视觉转移损失；
- latent diffusion action expert；
- L0 log replay 与 L1 kinematic rollout 接口；
- 数据审计、manifest、训练、latent 导出与测试工具。

## 1. 核心数据流

Tokenizer 训练：

```text
5 images -> frozen PE -> same-grid temporal delta -> z_vis ---+
                                                              +-> shared Decoder(z) -> trajectory
GT trajectory -----------> trajectory encoder -> z_traj -----+
                              z_vis -> stop-grad(z_traj) alignment
```

Action expert 推理：

```text
current/history condition tokens -> latent diffusion expert -> z_hat
z_hat -> frozen Decoder -> [40, 3] trajectory
```

未来图像只用于 tokenizer teacher 的训练和离线 latent target 导出。部署时 decoder 不读取
PE context、图像或未来信息。

## 2. 安装

建议使用 Python 3.10+、PyTorch 2.2+ 和 CUDA 环境：

```bash
cd vision_action_tokenizer
pip install -e ".[dev,viz]"
```

安装 Meta PE 官方代码：

```bash
git clone https://github.com/facebookresearch/perception_models.git third_party/perception_models
pip install -e third_party/perception_models
```

首次运行 PE 时会按官方逻辑下载 checkpoint；也可以通过环境已有的 Hugging Face 缓存读取。
本地权重可直接写入配置，不会再次联网下载：

```yaml
pe:
  model_name: PE-Spatial-B16-512
  checkpoint_path: /path/to/PE-Spatial-B16-512.pt
  forward_batch_size: 6
```

`forward_batch_size` 是 PE 内部的图像分块大小，不改变训练 batch；8 GB 显卡建议从 6
开始。本项目的 PE 预处理使用官方 `mean=std=[0.5, 0.5, 0.5]`，同时按项目设计采用
letterbox 保留前视相机完整横向视野。

## 3. 数据准备

假设已有：

```text
nuscenes_interp_12Hz_infos_train.pkl
nuscenes_interp_12Hz_infos_val.pkl
```

### 3.1 先审计真实 schema

```bash
python tools/inspect_infos.py \
  --info /path/to/nuscenes_interp_12Hz_infos_train.pkl \
  --data-root /path/to/nuscenes
```

该命令只读 pkl，输出：外层结构、样本字段、相机字段、scene 数、时间间隔、路径缺失和
pose 解析情况。只加载可信来源的 pickle 文件。

### 3.2 构建版本化 manifest

```bash
python tools/build_manifest.py \
  --config configs/nuscenes_lidar10hz_front_4s.yaml \
  --info /path/to/nuscenes_interp_12Hz_infos_train.pkl \
  --output data/manifests/train.jsonl \
  --report data/manifests/train_report.json
```

默认定义：

- teacher images：5 张官方 CAM_FRONT keyframe，目标时间为 `t0 + [0,1,2,3,4] s`；
- trajectory：从官方 LiDAR keyframe+sweeps 最近邻抽取视觉窗口内
  `t0 + [0.1,...,4.0] s`，共 40 点；`t0` 恒为局部坐标原点，故不重复保存；
- anchor stride：默认使用每个官方 keyframe，可通过 `anchor_stride_s` 降采样；
- 坐标系：anchor ego frame，`x` 向前、`y` 向左、yaw 逆时针为正。

两个时间轴相互独立：图像只从 pkl 中 32 位官方 token 对应的 keyframe 选择；轨迹不再
使用 12 Hz pkl 的插值 pose，而是流式读取官方 nuScenes 的 `sample_data.json`、
`ego_pose.json` 和 `calibrated_sensor.json`。默认 `trajectory_sampling: nearest` 直接使用
LiDAR 实测 ego pose，不做 SE(3) 插值。轨迹中的每个 `future_times_s` 保存实际 sweep
时间；所有导数都使用这个真实时间。实测当前数据的 LIDAR_TOP 原始间隔约 49.8 ms
（约 20 Hz），默认配置按 10 Hz 目标时间抽取 40 点。

注意：`ego_pose` 定义车辆位姿，`calibrated_sensor` 定义 LiDAR 到 ego 的固定外参；两者
都会解析并校验，但自车轨迹应由 `ego_pose` 构造，不能把 `sensor2ego` 误当成车辆运动。
以下数据设置全部可在 YAML 中修改：

```yaml
data:
  image_source: keyframe
  frame_offsets_s: [0, 1, 2, 3, 4]
  max_image_time_error_s: 0.25
  trajectory_pose_source: lidar_sweeps
  trajectory_sampling: nearest
  future_horizon_s: 4.0
  trajectory_hz: 10
  max_trajectory_time_error_s: 0.03
  max_pose_interpolation_gap_s: 0.15
  anchor_stride_s: 0.0
```

### 3.3 抽查样本

```bash
python tools/visualize_samples.py \
  --manifest data/manifests/train.jsonl \
  --output outputs/sample_check.png \
  --count 4
```

正式训练前应人工检查图像顺序、轨迹朝向和 4 s 覆盖。

也可以生成连续 CAM_FRONT 视频，把每一帧的 4 s / 40 点 LiDAR GT 轨迹直接投影到主视图：

```bash
python tools/visualize_trajectory_video.py \
  --config configs/nuscenes_lidar10hz_front_4s.yaml \
  --info data/nuscenes/nuscenes_interp_12Hz_infos_val.pkl \
  --output outputs/trajectory_projection/val_lidar10hz_4s.mp4
```

视频左下角颜色由近到远，右下角同时显示 `x 向前、y 向左` 的 BEV。配套 JSON 会记录
LiDAR pose 最近邻距离、可见投影点数量以及 local/global 两条解析路径的一致性误差。

## 4. 训练 tokenizer

默认配置位于 `configs/nuscenes_lidar10hz_front_4s.yaml`。最小训练命令：

```bash
/home/alan/miniconda3/envs/openmmlab/bin/torchrun \
  --standalone --nproc_per_node=1 tools/train_tokenizer.py \
  --config configs/nuscenes_lidar10hz_front_4s.yaml \
  --train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --val-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output output/itae_v2_tokenizer
```

多卡时修改 `--nproc_per_node`。训练脚本支持 DDP、BF16/FP16、断点恢复、梯度裁剪和
配置/manifest hash 记录。默认同时把每步训练 loss、学习率、梯度范数、每轮验证指标和
BEV 轨迹重建对比写入 `<output>/tensorboard`：

```bash
tensorboard --logdir output/itae_v2_tokenizer/tensorboard --port 6006
```

默认 `train.resume: auto`：启动时扫描 `--output` 目录下的 `*.pt`，按修改时间自动恢复
最新的完整 checkpoint（包含模型、optimizer、scheduler、scaler、epoch、global step 和
随机数状态）。也可以在 config 中指定文件；相对文件名会在 output 目录内查找：

```yaml
train:
  resume: auto                       # 自动选择 output 中最新 checkpoint
  # resume: last.pt                 # output/last.pt
  # resume: /abs/path/checkpoint.pt # 明确路径
  # resume: null                    # 禁止自动恢复
```

命令行 `--resume /path/to/checkpoint.pt` 的优先级最高；临时强制从头训练可加
`--no-resume`。若自动扫描的 output 目录为空，会打印提示并正常从 epoch 0 开始。

当前 V2 的维度、resampler、visual transition 和 decoder 都与早期
`output/itae_debug_tokenizer` 权重不兼容。首次训练必须使用新的 output（如上面的
`output/itae_v2_tokenizer`）；之后同一目录会继续自动恢复。V2 默认先训练 500 step 的轨迹
codec，再用 1000 step 渐进打开视觉重建/对齐；训练时可在 TensorBoard 的
`train/loss/visual_scale` 确认阶段是否符合预期。

验证可视化数量和频率由配置控制。每个 item 是一个 2×2 诊断页：左上显示 5 帧
CAM_FRONT 输入，其余三个面板分别是 GT、visual-latent reconstruction 和
trajectory-latent reconstruction 的共享尺度 BEV；绿→黄→红表示窗口内时间由近到远，
重建面板中的灰色虚线是 GT。默认从不同 scene 各取一个固定 item，避免连续窗口重复，并
确保不同 epoch 可在同一 TensorBoard image tag 下比较：

```yaml
tensorboard:
  enabled: true
  log_dir: null
  flush_secs: 30
  evaluation_visualization_items: 8
  evaluation_visualization_every_epochs: 1
  evaluation_visualization_include_images: true
  evaluation_visualization_distinct_scenes: true
```

即使训练使用离线 PE feature cache，验证集也会在需要可视化的 item 上读取原始主视图；若
希望完全跳过图像 I/O，可将 `evaluation_visualization_include_images: false`。
图像写入 TensorBoard 的 `evaluation/diagnostic_2x2/item_*`，同一个 tag 在不同 epoch
保持不变，便于拖动 step 对照输入场景和重建轨迹的变化。

建议按以下顺序验证：

1. `--overfit-samples 64`，确认数据、坐标和 decoder 能过拟合；
2. 保持 `sample_posterior: false`、`kl_weight: 0`，先验证 deterministic codec；
3. 观察 `train/posterior/visual_offdiag_cosine`，不能长期接近 1；
4. 基线稳定后再单独消融 KL 或 kinematic decoder，不要直接恢复 V1 checkpoint。

独立评估命令：

```bash
python tools/evaluate_tokenizer.py \
  --config configs/nuscenes_lidar10hz_front_4s.yaml \
  --manifest data/manifests/val.jsonl \
  --checkpoint outputs/tokenizer/best.pt \
  --output outputs/tokenizer/val_metrics.json
```

独立 evaluation 同样会把 scalar 和 BEV item 写到输出 JSON 同级的 `tensorboard_eval/`；
可用 `--tensorboard-dir <path>` 与 `--visualize-items <N>` 覆盖目录和抽样数量。

## 5. PE 特征缓存

PE 冻结时可离线缓存 patch tokens：

```bash
python tools/cache_pe_features.py \
  --config configs/nuscenes_lidar10hz_front_4s.yaml \
  --manifest data/manifests/train.jsonl \
  --output data/pe_cache/train \
  --shard-size 128
```

每个 safetensors shard 带 checkpoint、层号、图像变换和 manifest hash。训练时配置
`data.feature_cache` 即可跳过 PE 前向。train/val 使用不同缓存时写成：

```yaml
data:
  feature_cache:
    train: data/pe_cache/train
    val: data/pe_cache/val
```

## 6. 导出 action latent

```bash
python tools/export_action_latents.py \
  --config configs/nuscenes_lidar10hz_front_4s.yaml \
  --manifest data/manifests/train.jsonl \
  --checkpoint outputs/tokenizer/best.pt \
  --output data/action_latents/train.safetensors
```

默认导出稳定的 posterior mean `mu_vis`，同时保存仅由 train split 估计的 latent
normalizer。action expert 训练和 decoder 推理必须使用同一 normalizer/checkpoint hash。

## 7. 闭环等级

- **L0**：每 0.5 s 读取新的 logged observation 并重规划，只称为 pseudo closed-loop；
- **L1**：执行预测轨迹，使用 unicycle/kinematic tracker 推进 ego，其他 agents 日志回放；
- **L2**：预留 CARLA 或 world-model sensor-in-the-loop adapter，本版本不把 L0/L1 结果
  宣称为真实视觉闭环。

## 8. 测试

```bash
pytest
ruff check src tools tests
```

测试覆盖 SE(2) 坐标转换、时间窗口、导数与 mask、视觉/轨迹 encoder、两种 decoder、
完整 tokenizer loss、latent diffusion 以及 L0/L1 状态推进。

## 9. 关键接口

```python
# 视觉 teacher，仅在 tokenizer 训练/target 导出时使用。
out = tokenizer(
    visual_features=pe_tokens,       # [B, 5, N, C_pe]
    trajectory=trajectory,           # [B, 40, 3]
    frame_times=frame_times,         # [B, 5]
    future_times=future_times,       # [B, 40]
)

# 部署：decoder 不需要当前 PE context。
trajectory_hat = tokenizer.decode(action_tokens_hat, future_times)
```

所有公开张量接口的 docstring 都注明 shape、单位、坐标系和 mask 语义。关键几何、时间
匹配、物理积分与防止信息泄漏的位置均有代码注释。
