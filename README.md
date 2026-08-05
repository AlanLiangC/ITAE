# Vision-Aligned Action Tokenizer

面向端到端自动驾驶 action expert 的视觉对齐连续动作 tokenizer。项目把未来 5 s 的
`[60, 3] = [x, y, yaw]` 自车轨迹编码为 `[10, 256]` action tokens，并让 latent 同时满足：

- 可从当前到未来的 PE-Spatial 视觉状态转移中提取；
- 可由纯轨迹编码器得到，并与视觉 latent 对齐；
- 不依赖任何视觉旁路，能够由 `Decoder(Z)` 独立还原轨迹；
- 可作为 diffusion/flow action expert 的生成目标。

当前版本包括：

- MMDetection3D 新旧格式及自定义 12 Hz nuScenes info 适配；
- 6 帧 `[0, 1, 2, 3, 4, 5] s` 视觉窗口与 60 点、12 Hz 轨迹构建；
- `PE-Spatial-B16-512` patch-token 提取与离线缓存；
- 视觉转移 CVAE、轨迹编码器、共享无上下文 decoder；
- direct 和可微 unicycle 两种 trajectory head；
- 重建、动力学、物理、KL、跨模态对齐、视觉转移损失；
- latent diffusion action expert；
- L0 log replay 与 L1 kinematic rollout 接口；
- 数据审计、manifest、训练、latent 导出与测试工具。

## 1. 核心数据流

Tokenizer 训练：

```text
6 images -> frozen PE -> visual transition encoder -> z_vis --+
                                                              +-> shared Decoder(z) -> trajectory
GT trajectory -----------> trajectory encoder -> z_traj -----+
                                      z_vis <-> z_traj alignment
```

Action expert 推理：

```text
current/history condition tokens -> latent diffusion expert -> z_hat
z_hat -> frozen Decoder -> [60, 3] trajectory
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
  --info /path/to/nuscenes_interp_12Hz_infos_train.pkl \
  --data-root /path/to/nuscenes \
  --output data/manifests/train.jsonl \
  --report data/manifests/train_report.json
```

默认定义：

- teacher images：`t0 + [0, 1, 2, 3, 4, 5] s`；
- trajectory：`t0 + [1/12, ..., 60/12] s`；
- anchor stride：0.5 s；
- 坐标系：anchor ego frame，`x` 向前、`y` 向左、yaw 逆时针为正。

时间匹配使用真实时间戳的 nearest lookup，不依赖列表严格等间隔。窗口不会跨 scene。

### 3.3 抽查样本

```bash
python tools/visualize_samples.py \
  --manifest data/manifests/train.jsonl \
  --output outputs/sample_check.png \
  --count 4
```

正式训练前应人工检查图像顺序、轨迹朝向和 5 s 覆盖。

## 4. 训练 tokenizer

配置文件位于 `configs/nuscenes_12hz_front_5s.yaml`。最小训练命令：

```bash
torchrun --standalone --nproc_per_node=1 tools/train_tokenizer.py \
  --config configs/nuscenes_12hz_front_5s.yaml \
  --train-manifest data/manifests/train.jsonl \
  --val-manifest data/manifests/val.jsonl \
  --output outputs/tokenizer
```

多卡时修改 `--nproc_per_node`。训练脚本支持 DDP、BF16/FP16、断点恢复、梯度裁剪和
配置/manifest hash 记录。

建议按以下顺序验证：

1. `--overfit-samples 64`，确认数据、坐标和 decoder 能过拟合；
2. 将 `loss.kl_weight=0` 做 deterministic warm-up；
3. 开启 KL annealing、轨迹—视觉对齐和 PE transition loss；
4. 比较 direct 与 kinematic decoder。

独立评估命令：

```bash
python tools/evaluate_tokenizer.py \
  --config configs/nuscenes_12hz_front_5s.yaml \
  --manifest data/manifests/val.jsonl \
  --checkpoint outputs/tokenizer/best.pt \
  --output outputs/tokenizer/val_metrics.json
```

## 5. PE 特征缓存

PE 冻结时可离线缓存 patch tokens：

```bash
python tools/cache_pe_features.py \
  --config configs/nuscenes_12hz_front_5s.yaml \
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
  --config configs/nuscenes_12hz_front_5s.yaml \
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
    visual_features=pe_tokens,       # [B, 6, N, C_pe]
    trajectory=trajectory,           # [B, 60, 3]
    frame_times=frame_times,         # [B, 6]
    future_times=future_times,       # [B, 60]
)

# 部署：decoder 不需要当前 PE context。
trajectory_hat = tokenizer.decode(action_tokens_hat, future_times)
```

所有公开张量接口的 docstring 都注明 shape、单位、坐标系和 mask 语义。关键几何、时间
匹配、物理积分与防止信息泄漏的位置均有代码注释。
