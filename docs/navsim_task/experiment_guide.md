# NAVSIM + nuScenes V4 Scratch 联合训练指南

## 1. 环境

```bash
conda activate py312torch210cu126
pip install -e .
pip install -e third_party/navsim --no-deps
pip install --no-deps \
  'nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2'
pip install \
  numpy==1.26.4 scipy==1.14.1 pandas==3.0.5 shapely==2.1.2 \
  pyquaternion hydra-core==1.3.5 joblib requests retry aioboto3 aiofiles
pip install --no-deps \
  geopandas==0.14.4 rasterio==1.4.3 fiona pyogrio pyproj
```

不要直接执行 `pip install -r third_party/navsim/requirements.txt`。官方 requirements 会把
PyTorch 降到 2.0.1、NumPy 降到 1.23.4，与当前 Python 3.12 / PyTorch 2.10 / VGGT-Omega
环境冲突。上述组合已经实际通过 NAVSIM dataloader、nuPlan interpolation 和 mini 数据读取。

每个 shell 设置：

```bash
export OPENSCENE_DATA_ROOT=/home/alan/AlanLiang/Dataset/navsim
export NUPLAN_MAPS_ROOT=/home/alan/AlanLiang/Dataset/navsim/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

快速 import 检查：

```bash
python -c "from navsim.common.dataloader import SceneLoader; \
from nuplan.planning.simulation.trajectory.interpolated_trajectory import \
InterpolatedTrajectory; print('NAVSIM core imports OK')"
```

## 2. 生成 NAVSIM mini manifest

配置文件：
`configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml`。

```bash
python -m tools.data.build_navsim_manifest \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --train-output data/manifests/navsim_mini_train_4s.jsonl \
  --val-output data/manifests/navsim_mini_val_4s.jsonl \
  --report-output data/manifests/navsim_mini_4s_report.json
```

当前 mini 实测输出：

- train：51 个 log，31,462 个有效窗口；
- validation：13 个 log，8,631 个有效窗口；
- train/validation 按完整 log 隔离；
- 每个窗口为五张 `CAM_F0`（0/1/2/3/4s）和 40 个 10Hz pose；
- 官方滑窗产生的跨 scene 候选会被拒绝，不会进入 manifest。

检查 2Hz knot、40 点 GT 和可视化：

```bash
python -m tools.analysis.validate_navsim_interpolation \
  --manifest data/manifests/navsim_mini_train_4s.jsonl \
  --output output/navsim_manifest_audit/interpolation.json

python -m tools.visualization.visualize_navsim_trajectory \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --output output/navsim_manifest_audit/sample.png \
  --index 0

# 输出五帧 2×2 MP4；需要系统 ffmpeg。
python -m tools.visualization.visualize_navsim_trajectory \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --output output/navsim_manifest_audit/sample.mp4 \
  --index 0 --fps 2
```

在 `interpolation.json` 的 `passed` 为 true、并人工确认主视图/BEV 方向前，不要开始 cache。

## 3. 生成 NAVSIM VGGT-Omega feature cache

```bash
python -m tools.features.cache_vggt_omega_features \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --manifest data/manifests/navsim_mini_train_4s.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/navsim_mini_front_4s_train_rich \
  --batch-size 1 --shard-size 128 --num-workers 2

python -m tools.features.cache_vggt_omega_features \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/navsim_mini_front_4s_val_rich \
  --batch-size 1 --shard-size 128 --num-workers 2
```

cache 支持断点续建，并严格检查 manifest SHA256、checkpoint SHA256、sample 顺序、五帧
shape、register token mode 和每个 shard checksum。不要修改 manifest 后复用旧 cache。

nuScenes rich cache 已由主配置指向：

```text
/home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_train_rich
/home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_val_rich
```

## 4. 联合训练

正式训练不再传 `--train-manifest/--val-manifest`；所有 source 路径和比例都在 config 中：

```bash
torchrun --standalone --nproc_per_node=1 --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --output output/itae_v4_scratch_nuscenes_navsim
```

这个配置具有以下强约束：

- `resume: never`，不会自动读取 output 中的历史权重；
- `initial_checkpoint: null`，不加载 V2/V3/V4 tokenizer；
- `freeze_base: false`，V4 action tokenizer 的 2,401,181 个参数全部训练；
- VGGT-Omega 是冻结 cache，仅 action tokenizer from scratch；
- nuScenes/NAVSIM 期望采样比例各 50%，motion balancing 在 source 内归一化，不会改变
  50/50 的 source 总权重；
- best checkpoint 依据 `macro/metric/ade_m`，两个数据集等权；
- TensorBoard 分别记录 `validation/source/nuscenes/*`、
  `validation/source/navsim/*`、`validation/macro/*`、`validation/weighted/*` 和两类
  2×2 图像；训练还会对比 sampler 的 expected/actual source、motion、speed-trend 比例。

只训练单个 source 做 ablation：

```bash
torchrun --standalone --nproc_per_node=1 --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --sources navsim \
  --output output/itae_v4_scratch_navsim_only

torchrun --standalone --nproc_per_node=1 --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --sources nuscenes \
  --output output/itae_v4_scratch_nuscenes_only
```

显式恢复同一次实验：

```bash
torchrun --standalone --nproc_per_node=1 --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --output output/itae_v4_scratch_nuscenes_navsim \
  --resume output/itae_v4_scratch_nuscenes_navsim/last.pt
```

查看 TensorBoard：

```bash
tensorboard --logdir output/itae_v4_scratch_nuscenes_navsim/tensorboard --port 6006
```

## 5. 分数据集评估

```bash
python -m tools.evaluation.evaluate_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --feature-cache /home/alan/AlanLiang/Dataset/vggt_omega_cache/navsim_mini_front_4s_val_rich \
  --checkpoint output/itae_v4_scratch_nuscenes_navsim/best.pt \
  --output output/itae_v4_scratch_nuscenes_navsim/eval_navsim.json \
  --batch-size 32 --visualize-items 8

python -m tools.evaluation.evaluate_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --feature-cache /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_val_rich \
  --checkpoint output/itae_v4_scratch_nuscenes_navsim/best.pt \
  --output output/itae_v4_scratch_nuscenes_navsim/eval_nuscenes.json \
  --batch-size 32 --visualize-items 8
```

定性检查 NAVSIM 连续片段：脚本默认从 validation 自动挑选一段转弯明显的 16 帧片段，
使用 `best.pt` encode/decode，在 CAM_F0 上投影浅色虚线 GT 与绿→黄→红 decode，并在右下角
显示共享尺度的 BEV 对照：

```bash
python -m tools.visualization.visualize_navsim_tokenizer_video
```

默认输出：
`output/itae_v4_scratch_nuscenes_navsim/navsim_gt_vs_reconstruction.mp4`。也可以指定起点：

```bash
python -m tools.visualization.visualize_navsim_tokenizer_video \
  --sample-token navsim:1ac4c62d5ad35d16 \
  --num-frames 24 --fps 2 \
  --output output/itae_v4_scratch_nuscenes_navsim/navsim_selected_clip.mp4
```

同名 JSON 会保存逐帧 ADE/FDE、可见投影点数、scene/log/sample token，便于定位误差大的
样本。`--scene-token`、`--group-token` 可以限制自动选段范围，`--selection first` 可关闭
转弯优先选择。

mini 只用于数据/训练链路和 tokenizer reconstruction。没有 navtest metric cache 时不报告
PDMS。未来 planner 输出 40 点后，可调用
`vision_action_tokenizer.data.navsim.make_navsim_trajectory_from_dense` 还原为官方
8×2Hz `Trajectory`，再交给 PDMS 插值/评分。插值审计同时验证 `40→8→40` round-trip。

## 6. 已完成的 smoke closure

可复现 smoke 配置：
`configs/tokenizer/nuscenes_navsim_v4_scratch_4s_smoke.yaml`。

```bash
torchrun --standalone --nproc_per_node=1 --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_smoke.yaml \
  --output output/navsim_joint_v4_scratch_smoke \
  --no-resume
```

已实际验证：两 source cache 读取、1:1 sampler、forward/backward、contrastive alignment、
分 source/macro validation、TensorBoard 图像、best/last checkpoint、显式 resume 和两个
standalone evaluation 均能运行。

三个 scratch overfit 命令（分别追加 `--sources navsim`、`--sources nuscenes`，以及不加
`--sources` 的 joint）也已闭环。运行时使用
`configs/tokenizer/nuscenes_navsim_v4_scratch_4s_overfit.yaml --overfit-samples 4
--overfit-on-train`。30 epoch 实测 best train-set ADE 分别为 NAVSIM `2.260m`、nuScenes
`0.436m`、joint macro `3.705m`，均较随机初始化显著下降；这是诊断缓存/坐标/梯度链路，
不用于报告泛化指标。
