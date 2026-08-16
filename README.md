# ITAE：面向自动驾驶的视觉 Action Tokenizer 与 Diffusion Planner

ITAE（Image/Trajectory Action Encoding）研究如何从驾驶视觉中学习少量 action token 来表达
同一时间窗口内的连续 ego 运动，并验证下游 flow-matching diffusion planner 预测原始轨迹与
预测 action token 时的收敛速度和规划效果。连续轨迹是 tokenizer 的监督目标，不是视觉
encoder 的额外输入。

当前主线包含两部分：

1. **Vision Action Tokenizer**：从冻结的视觉基础模型特征中提取运动信息，将 4 秒连续轨迹编码成
   4 个 action token，再仅依赖 token 和时间查询重建轨迹；
2. **Flow Planner**：以 PE 等视觉特征和 ego-motion 为条件，用 flow matching 预测原始轨迹或
   tokenized action，推理默认使用 5 个 Euler step。

当前推荐 tokenizer baseline 是 **V4 output-residual tokenizer**，使用 nuScenes 与 NAVSIM 联合
训练。本 README 以在 Linux GPU 集群上复现该 tokenizer 为主；planner 实验见
[Flow Planner 实验指南](docs/FLOW_PLANNER_EXPERIMENT_GUIDE.md)。使用完整 NAVSIM trainval 扩大
tokenizer 训练数据时，见
[NAVSIM trainval Action Tokenizer 训练指南](docs/tokenizer/NAVSIM_TRAINVAL_ACTION_TOKENIZER_GUIDE.md)。
复现 SUV 在 NAVSIM v1 上的官方 PDMS 时，见
[SUV NAVSIM v1 测评指南](docs/suv/NAVSIM_V1_EVALUATION_GUIDE.md)。

## 1. 任务定义

### 1.1 时序语义

每个训练样本包含：

- 5 张主视图图像，时刻为 `t = [0, 1, 2, 3, 4]s`；
- 40 个 ego pose，时刻为 `t = [0.1, 0.2, ..., 4.0]s`；
- 每个 pose 是 anchor ego 坐标系下的 `[x, y, yaw]`，单位为米和弧度；
- 坐标约定为 `x` 向前、`y` 向左、yaw 逆时针。

这里的 40 点轨迹覆盖的是**五张输入图之间的同一段 4 秒视觉窗口**，不是最后一张图之后的
未来 4 秒。第一张图是 anchor，轨迹不包含平凡的 `t=0` 原点。

数据集时序统一方式：

| 数据集 | 主视图 | 原生轨迹 | tokenizer 输入 |
| --- | --- | --- | --- |
| nuScenes | 5 张 `CAM_FRONT` keyframe，1Hz | LiDAR keyframe/sweep ego pose | 40×10Hz pose |
| NAVSIM | 从 2Hz `CAM_F0` 取 0/1/2/3/4s | 8×2Hz rear-axle pose | SE(2) 插值为 40×10Hz pose |

NAVSIM 插值对 `x/y` 做线性插值，对 yaw 先 unwrap 再插值。原生 0.5 秒 knot、项目重建以及
`40→8→40` round-trip 均有测试。

### 1.2 V4 tokenizer

```text
5 × front-camera images
  -> frozen pretrained VGGT-Omega
  -> CameraHead hidden + register tokens + pose encoding
  -> 4 × (128-D motion token + 64-D visual residual token)
  -> 4 × 192-D action token
  -> context-free SE(2) velocity/increment decoder
  -> 40 × [x, y, yaw]
```

联合训练中的 “from scratch” 指 **2,401,181 个 V4 action-tokenizer 参数全量随机初始化并训练**；
VGGT-Omega 1B 视觉骨干仍使用预训练权重且保持冻结，训练通常读取预先生成的 feature cache。

## 2. 仓库结构

```text
configs/
  tokenizer/       # nuScenes + NAVSIM V4 联合训练配置
  planner/         # raw trajectory vs action token flow-planner 配置
src/vision_action_tokenizer/
  data/            # manifest、轨迹插值、NAVSIM adapter、多源 sampler
  models/          # tokenizer、decoder、视觉 backbone wrapper
  losses.py        # reconstruction、动力学、contrastive alignment 等 loss
  trainer.py       # DDP、checkpoint、TensorBoard、source-aware validation
tools/
  data/            # manifest 构建与切分
  features/        # VGGT/PE/action-target cache
  training/        # tokenizer、planner、action expert 训练
  evaluation/      # reconstruction/planner evaluation
  analysis/        # 插值、相机运动和 checkpoint 分析
  visualization/   # 图像投影、BEV 和视频
docs/              # 设计计划、历史实验和详细指南
third_party/       # VGGT-Omega、NAVSIM、SUV、Perception Models submodule
```

## 3. 集群迁移清单

以下内容不会随普通 Git clone 自动出现：

- `data/manifests/`：被 `.gitignore` 忽略；
- `data/nuscenes/` 及原始 NAVSIM 数据；
- VGGT-Omega feature cache；
- `output/` 中的 checkpoint、TensorBoard event 和评估结果；
- VGGT-Omega / PE 等外部模型权重。

因此迁移至少需要以下四类资产：

```text
<DATA_ROOT>/nuscenes/                         # nuScenes 原始数据和 info pkl
<DATA_ROOT>/navsim/                           # NAVSIM logs / sensor_blobs / maps
<CACHE_ROOT>/vggt_omega_cache/                # 四个 rich feature cache
<CHECKPOINT_ROOT>/VGGT-Omega/vggt_omega_1b_512.pt
<PROJECT>/data/manifests/*.jsonl              # 与 cache 严格对应的 manifest
<PROJECT>/output/itae_v4_scratch_nuscenes_navsim/  # 若要续训/评估现有模型
```

当前四个联合训练 cache 的参考大小约为：

| Cache | 大小 |
| --- | ---: |
| nuScenes train rich | 6.7GB |
| nuScenes val rich | 1.4GB |
| NAVSIM mini train rich | 11GB |
| NAVSIM mini val rich | 3.0GB |

推荐先在集群 clone 代码，再用 `rsync` 单独同步不可版本化资产：

```bash
# 在本地执行；按实际登录节点和目标目录修改。
rsync -ahP data/manifests/ USER@CLUSTER:/path/to/ITAE/data/manifests/
rsync -ahP /path/to/vggt_omega_cache/ USER@CLUSTER:/cluster/cache/vggt_omega_cache/
rsync -ahP /path/to/pure_checkpoints/ USER@CLUSTER:/cluster/checkpoints/
rsync -ahP output/itae_v4_scratch_nuscenes_navsim/ \
  USER@CLUSTER:/path/to/ITAE/output/itae_v4_scratch_nuscenes_navsim/
```

注意：cache 的 `index.json` 记录了 manifest SHA256、样本顺序、预处理方式、backbone SHA256
和 shard checksum。**不要修改 manifest 内容后继续复用旧 cache。**

nuScenes manifest 当前使用相对路径 `data/nuscenes/...`，集群上可将该目录连接到共享数据盘。
NAVSIM manifest 当前保存绝对图片路径；若集群数据路径不同，推荐在集群重新生成 NAVSIM
manifest 并重建对应 cache。另一种方式是在集群创建与旧绝对路径一致的兼容软链接，但通常不如
重新导出清晰。即使训练读取 cache，TensorBoard 定性可视化仍需要原始图片路径可访问。

## 4. 环境配置与安装

### 4.1 系统要求

- Linux；
- NVIDIA GPU 和可运行 PyTorch CUDA build 的驱动；
- Conda/Mamba；
- `ffmpeg`，仅视频可视化需要；
- Git 及 submodule 访问权限。

本项目已验证的环境为：

```text
Python      3.12.13
PyTorch     2.10.0+cu126
torchvision 0.25.0+cu126
CUDA build  12.6
NumPy       1.26.4
SciPy       1.14.1
```

集群 CUDA module 名称因平台而异；PyTorch wheel 自带 CUDA runtime，但宿主机 NVIDIA driver
必须兼容。先在计算节点运行 `nvidia-smi`，不要只在无 GPU 的登录节点判断环境是否可用。

### 4.2 Clone 与 submodule

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL> ITAE
cd ITAE
git submodule update --init --recursive
git submodule status
```

如果 `git submodule status` 某行以 `-` 开头，说明该 submodule 尚未初始化。

### 4.3 创建 tokenizer 训练环境

```bash
conda create -n itae python=3.12 -y
conda activate itae

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install -e third_party/vggt-omega
python -m pip install -e ".[dev,viz]"
```

如果还要训练 PE planner，再安装：

```bash
python -m pip install -e ".[pe]"
```

验证核心训练环境：

```bash
python - <<'PY'
import torch
import vision_action_tokenizer

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 4.4 安装 NAVSIM 数据导出依赖

只读取已生成的 manifest/cache 训练 tokenizer 时，不必在每个训练节点导入完整 NAVSIM 栈。
如果要重新生成 NAVSIM manifest、运行官方插值 parity 或后续 PDMS，则安装：

```bash
python -m pip install -e third_party/navsim --no-deps
python -m pip install --no-deps \
  "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit.git@nuplan-devkit-v1.2"
python -m pip install \
  numpy==1.26.4 scipy==1.14.1 pandas==3.0.5 shapely==2.1.2 \
  pyquaternion hydra-core==1.3.5 joblib requests retry aioboto3 aiofiles
python -m pip install --no-deps \
  geopandas==0.14.4 rasterio==1.4.3 fiona pyogrio pyproj
```

不要直接执行 `pip install -r third_party/navsim/requirements.txt`：官方旧版 requirements 会把
PyTorch/NumPy 降级到与当前 Python 3.12、PyTorch 2.10 和 VGGT-Omega 不兼容的版本。

在 shell 或调度脚本中设置：

```bash
export OPENSCENE_DATA_ROOT=/cluster/datasets/navsim
export NUPLAN_MAPS_ROOT=/cluster/datasets/navsim/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

验证：

```bash
python -c "from navsim.common.dataloader import SceneLoader; \
from nuplan.planning.simulation.trajectory.interpolated_trajectory import \
InterpolatedTrajectory; print('NAVSIM imports OK')"
```

## 5. 数据准备

### 5.1 集群目录建议

仓库只保存代码和配置；大文件放在共享数据盘/高速 scratch：

```text
/cluster/datasets/
  nuscenes/
  navsim/
/cluster/checkpoints/
  VGGT-Omega/vggt_omega_1b_512.pt
/cluster/cache/vggt_omega_cache/
  nuscenes_front_4s_train_rich/
  nuscenes_front_4s_val_rich/
  navsim_mini_front_4s_train_rich/
  navsim_mini_front_4s_val_rich/
<repo>/data/manifests/
<repo>/output/
```

nuScenes 可以通过软链接保持项目内相对路径：

```bash
mkdir -p data
ln -s /cluster/datasets/nuscenes data/nuscenes
```

如果 `data/nuscenes` 需要同时连接多个目录，可参照原环境分别连接 `samples/`、`sweeps/`、
`v1.0-trainval/`、`maps/` 和两个 `nuscenes_interp_12Hz_infos_{train,val}.pkl`。

### 5.2 修改集群配置

主配置为
[`configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml`](configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml)。
复制一份集群配置，避免把本机绝对路径直接带入新实验：

```bash
cp configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml
```

至少检查以下字段：

```yaml
data:
  num_workers: 4                 # 每个 DDP process 的 worker 数
  sources:
    nuscenes:
      train_manifest: data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl
      val_manifest: data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl
      train_feature_cache: /cluster/cache/vggt_omega_cache/nuscenes_front_4s_train_rich
      val_feature_cache: /cluster/cache/vggt_omega_cache/nuscenes_front_4s_val_rich
    navsim:
      train_manifest: data/manifests/navsim_mini_train_4s.jsonl
      val_manifest: data/manifests/navsim_mini_val_4s.jsonl
      train_feature_cache: /cluster/cache/vggt_omega_cache/navsim_mini_front_4s_train_rich
      val_feature_cache: /cluster/cache/vggt_omega_cache/navsim_mini_front_4s_val_rich

navsim_export:
  data_root: /cluster/datasets/navsim

vision_backbone:
  source_path: third_party/vggt-omega
  checkpoint_path: /cluster/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt
```

VGGT-Omega 权重必须满足：

```text
SHA256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
```

校验命令：

```bash
sha256sum /cluster/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt
```

### 5.3 联合采样策略

```yaml
data:
  sampling:
    strategy: balanced       # proportional | balanced | weighted
    source_weights:
      nuscenes: 0.2
      navsim: 0.8
```

- `proportional`：按各 source 实际样本量；
- `balanced`：每个 source 等概率，`source_weights` 不参与计算；
- `weighted`：才会按 `source_weights` 归一化采样。

因此当前主配置虽然保留 `0.2/0.8`，但因为 `strategy: balanced`，实际仍是 nuScenes/NAVSIM
各 50%。若希望 NAVSIM 占 80%，必须把 strategy 改为 `weighted`。motion balancing 会在每个
source 内归一化，不会破坏目标 source 比例。训练日志和 TensorBoard 会同时报告 expected 与
actual source/motion/speed-trend fraction。

### 5.4 生成 NAVSIM manifest

```bash
python -m tools.data.build_navsim_manifest \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --train-output data/manifests/navsim_mini_train_4s.jsonl \
  --val-output data/manifests/navsim_mini_val_4s.jsonl \
  --report-output data/manifests/navsim_mini_4s_report.json
```

mini 参考结果为 train `31,462` 个窗口/51 logs，validation `8,631` 个窗口/13 logs，两个 split
按完整 log 隔离。生成后必须审计插值与坐标：

```bash
python -m tools.analysis.validate_navsim_interpolation \
  --manifest data/manifests/navsim_mini_train_4s.jsonl \
  --output output/navsim_manifest_audit/interpolation.json

python -m tools.visualization.visualize_navsim_trajectory \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --output output/navsim_manifest_audit/sample.png \
  --index 0
```

只有 `interpolation.json` 的 `passed=true` 且人工确认 CAM_F0/BEV 方向后才生成 cache。

nuScenes manifest 构建入口为：

```bash
python -m tools.data.build_nuscenes_manifest \
  --config configs/nuscenes_lidar10hz_front_4s.yaml \
  --info data/nuscenes/nuscenes_interp_12Hz_infos_train.pkl \
  --data-root data/nuscenes \
  --output data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --report data/manifests/nuscenes_lidar10hz_front_4s_train.report.json
```

validation 使用对应的 `...infos_val.pkl` 和 val 输出文件。

### 5.5 生成冻结 VGGT-Omega feature cache

四个 split 分别运行一次；以下以 NAVSIM train 为例：

```bash
python -m tools.features.cache_vggt_omega_features \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --manifest data/manifests/navsim_mini_train_4s.jsonl \
  --output /cluster/cache/vggt_omega_cache/navsim_mini_front_4s_train_rich \
  --batch-size 1 \
  --shard-size 128 \
  --num-workers 2
```

将 manifest/output 替换为 NAVSIM val、nuScenes train、nuScenes val 即可。cache 工具支持在
同一目录断点续建，会验证每个 shard；不要让两个作业同时写同一个 cache 目录。

## 6. Tokenizer 训练

### 6.1 先跑 smoke test

迁移后不要直接提交数百 epoch。先将 smoke 配置中的 manifest/cache 路径改为集群路径，然后：

```bash
torchrun --standalone --nproc_per_node=1 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_smoke.yaml \
  --output output/navsim_joint_v4_scratch_smoke \
  --no-resume
```

应看到：

- `Total trainable parameters: 2,401,181`；
- 两个 source 都进入 sampler 和 validation；
- loss finite，产生 `best.pt`、`best_trained.pt`、`last.pt`；
- TensorBoard 中出现 NAVSIM/nuScenes 独立指标和 2×2 图像。

### 6.2 从头联合训练

从头训练时在 cluster config 中显式设置：

```yaml
train:
  resume: never
  initial_checkpoint: null
  freeze_base: false
```

然后运行：

```bash
torchrun --standalone --nproc_per_node=1 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --output output/itae_v4_scratch_nuscenes_navsim
```

也可以用同一配置做单源 ablation：

```bash
# NAVSIM only
torchrun --standalone --nproc_per_node=1 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --sources navsim \
  --output output/itae_v4_scratch_navsim_only

# nuScenes only：将 --sources 改为 nuscenes，并使用独立 output。
```

`train.batch_size` 是**每个 DDP process/GPU** 的 batch size。多卡时 global batch 为
`batch_size × GPU 数`；当前 trainer 没有 gradient accumulation，也不会自动按 GPU 数线性缩放
learning rate。

### 6.3 单节点多 GPU

```bash
torchrun --standalone --nproc_per_node=4 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --output output/itae_v4_scratch_nuscenes_navsim_4gpu
```

自定义 weighted sampler 会生成一个确定性全局采样流，再按 rank 分片；validation 指标也会跨
rank reduce。每个进程都会创建 `data.num_workers` 个 DataLoader worker，因此 4 GPU、
`num_workers: 4` 会使用 16 个 worker。

### 6.4 Slurm 单节点示例

集群的 partition、module 和资源字段需要按实际平台调整：

```bash
#!/usr/bin/env bash
#SBATCH --job-name=itae-v4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=itae-v4-%j.out

set -euo pipefail
cd /path/to/ITAE
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate itae

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENSCENE_DATA_ROOT=/cluster/datasets/navsim
export NUPLAN_MAPS_ROOT=/cluster/datasets/navsim/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0

torchrun --standalone --nproc_per_node=4 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --output output/itae_v4_scratch_nuscenes_navsim_4gpu
```

这是单节点模板。多节点需要由调度器提供一致的 `MASTER_ADDR/MASTER_PORT`、`node_rank` 和
`nnodes`；正式扩展前先验证单节点 DDP。共享 cache 建议放在计算节点可高吞吐读取的文件系统。

### 6.5 恢复训练与 checkpoint

同一次实验被抢占后，推荐显式恢复：

```bash
torchrun --standalone --nproc_per_node=4 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --output output/itae_v4_scratch_nuscenes_navsim_4gpu \
  --resume output/itae_v4_scratch_nuscenes_navsim_4gpu/last.pt
```

也可将 `train.resume` 改为 `auto`；它优先读取 `last.pt`。checkpoint 含 model、optimizer、
scheduler、AMP scaler、epoch、global step、RNG 和 manifest/cache provenance。联合训练恢复时会
严格比较 source、manifest hash、cache 路径、架构与 loss 配置，避免串实验。

建议续训时保持 GPU 数、per-GPU batch size 和数据配置不变。三个 checkpoint 的含义：

- `last.pt`：最后完成的 epoch，恢复训练入口；
- `best.pt`：按 `train.checkpoint_metric` 选择的最优模型；
- `best_trained.pt`：排除某些初始 warm-start baseline 后，本次训练自身最优模型。

当前联合训练参考结果位于 `output/itae_v4_scratch_nuscenes_navsim`：`best.pt` 在 epoch 118、
global step 189,091 达到 macro ADE `0.299m`；`last.pt` 完成 epoch 299。该目录被 Git 忽略，
必须单独同步到集群。

### 6.6 TensorBoard

```bash
tensorboard \
  --logdir output/itae_v4_scratch_nuscenes_navsim/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

通过 SSH port forwarding 访问时，按集群登录/计算节点网络规则建立隧道。重点查看：

- `validation/source/nuscenes/*`、`validation/source/navsim/*`；
- `validation/macro/*`、`validation/weighted/*`；
- reconstruction ADE/FDE/yaw 与速度、加速度、jerk；
- action token std/cosine、contrastive alignment top-1；
- shuffled visual-condition error gap；
- expected/actual source、motion、speed-trend fraction；
- 两个数据集各自的五帧输入和 GT/reconstruction 2×2 页面。

## 7. 评估与定性可视化

### 7.1 分数据集 reconstruction evaluation

```bash
python -m tools.evaluation.evaluate_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --feature-cache /cluster/cache/vggt_omega_cache/navsim_mini_front_4s_val_rich \
  --checkpoint output/itae_v4_scratch_nuscenes_navsim/best.pt \
  --output output/itae_v4_scratch_nuscenes_navsim/eval_navsim.json \
  --batch-size 32 \
  --visualize-items 8

python -m tools.evaluation.evaluate_tokenizer \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --feature-cache /cluster/cache/vggt_omega_cache/nuscenes_front_4s_val_rich \
  --checkpoint output/itae_v4_scratch_nuscenes_navsim/best.pt \
  --output output/itae_v4_scratch_nuscenes_navsim/eval_nuscenes.json \
  --batch-size 32 \
  --visualize-items 8
```

### 7.2 NAVSIM GT/decode 投影视频

```bash
python -m tools.visualization.visualize_navsim_tokenizer_video \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s_cluster.yaml \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --feature-cache /cluster/cache/vggt_omega_cache/navsim_mini_front_4s_val_rich \
  --checkpoint output/itae_v4_scratch_nuscenes_navsim/best.pt \
  --output output/itae_v4_scratch_nuscenes_navsim/navsim_gt_vs_reconstruction.mp4
```

视频在 CAM_F0 上绘制浅色虚线 GT 和绿→黄→红的 decode 轨迹，右下角显示共享尺度 BEV；
同名 JSON 保存逐帧 ADE/FDE 和 sample token。可用 `--sample-token`、`--scene-token`、
`--group-token` 指定片段。

NAVSIM mini 只用于数据、训练和 reconstruction 闭环；没有对应 navtest metric cache 时不报告
真实 PDMS。

## 8. 常见问题

### Cache reports `manifest_sha256` mismatch

cache 与 manifest 内容或样本顺序不同。使用 cache 对应的原始 manifest，或删除到一个新的输出
目录重新生成 cache；不要手工改 `index.json` 绕过检查。

### 集群上找不到 NAVSIM 图片

旧 NAVSIM manifest 写入了本机绝对路径。重新生成 manifest/cache，或建立保持旧路径的软链接。
训练读取 cache 时可能暂时不报错，但 TensorBoard/视频读取原图时会失败。

### `source_weights: 0.2/0.8` 没有效果

检查 `data.sampling.strategy`。`balanced` 恒为 source 等权；只有 `weighted` 使用
`source_weights`。

### CUDA OOM

feature-cache 模式下先减小 `train.batch_size`；在线 VGGT-Omega 模式在 8GB GPU 上只建议
batch size 1。多卡 batch size 是 per-GPU 值。必要时降低 TensorBoard 可视化数量，但这不会改变
训练 forward 的主要显存。

### Resume 后配置不匹配

联合 checkpoint 会检查 source、manifest/cache、架构和 loss provenance。确认使用原实验配置与
资产；新的数据或 loss 实验应使用新的 output 目录，从头训练。

### Slurm 作业训练很慢

确认 cache 位于计算节点高吞吐文件系统；`data.num_workers` 是每个 rank 的 worker 数；避免大量
作业同时读取同一机械盘。检查日志中的 actual source ratio、GPU utilization 和 DataLoader I/O。

## 9. 测试与代码质量

迁移环境后运行：

```bash
python -m pytest -q
python -m ruff check src tools tests
python -m tools.training.train_tokenizer --help
```

正式训练前还应完成 smoke train、显式 resume、standalone evaluation 和至少一段投影视频。

## 10. 建议学习顺序

1. 阅读 [最初总体设计](docs/VISION_ACTION_TOKENIZER_CODE_PLAN.md)，理解视觉窗口和轨迹定义；
2. 阅读 `src/vision_action_tokenizer/data/manifest.py`、`trajectory.py`、`navsim.py`，理解两数据集如何
   统一为 40×10Hz SE(2)；
3. 阅读 `models/tokenizer.py` 和 `models/decoder.py`，理解 4-token 表征与 context-free decode；
4. 阅读 `losses.py`、`trainer.py`，理解 reconstruction、动力学、contrastive alignment、DDP 和
   source-aware validation；
5. 阅读 [V1–V4 tokenizer 实验记录](docs/VGGT_OMEGA_V2_EXPERIMENTS.md) 和
   [NAVSIM 联合训练指南](docs/navsim_task/experiment_guide.md)；
6. 最后阅读 [raw trajectory vs token planner 方案](docs/FLOW_PLANNER_RAW_VS_TOKEN_CODE_PLAN.md)
   与 [2Hz 三帧实验](docs/FLOW_PLANNER_2HZ3F_EXPERIMENT_GUIDE.md)。

## 11. License

项目代码使用仓库 [LICENSE](LICENSE)。`third_party/vggt-omega` 使用 FAIR Noncommercial
Research License 并包含 Acceptable Use Policy；NAVSIM、SUV、Perception Models 以及外部权重
分别遵循其原始许可证。迁移、共享数据或发布 checkpoint 前请分别确认授权范围。
