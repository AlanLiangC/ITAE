# NAVSIM trainval Action Tokenizer 训练指南

本文档给出 V4 output-residual action tokenizer 在 NAVSIM `trainval` 上从数据导出、
VGGT-Omega 特征缓存到单机多卡训练、恢复和评估的完整流程。所有命令均从 ITAE 仓库根目录执行，
使用 Conda 环境 `py312torch210cu126`。

对应配置：

- 正式训练：`configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml`；
- 小规模闭环：`configs/tokenizer/navsim_trainval_v4_scratch_4s_smoke.yaml`。

正式配置按完整 log 做确定性 90%/10% 切分。相同 `split_seed` 和 trainval 日志集合会得到相同
train/validation log 列表；一个 log 不会同时出现在两个 split 中。

## 1. 数据与环境检查

```bash
conda activate py312torch210cu126

export OPENSCENE_DATA_ROOT=/inspire/hdd/global_public/public_datas/NAVSIM
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0

python - <<'PY'
from pathlib import Path

from navsim.common.dataloader import SceneLoader
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

root = Path("/inspire/hdd/global_public/public_datas/NAVSIM")
logs = list((root / "navsim_logs/trainval").glob("*.pkl"))
assert logs, "NAVSIM trainval logs are missing or the symlink is broken"
assert (root / "sensor_blobs/trainval").is_dir()
print(f"NAVSIM trainval logs: {len(logs)}")
print("NAVSIM imports OK")
PY
```

当前共享数据根目录的 `navsim_logs/trainval` 与 `sensor_blobs/trainval` 是软链接。检查数量时应使用
Python `Path.glob` 或 `find -L`，普通 `find` 不跟随软链接，可能错误地报告 0 个文件。当前数据应有
1,310 个 trainval log。

确认 GPU 和权重：

```bash
python - <<'PY'
import torch

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

sha256sum \
  /inspire/hdd/project/spatiotemporal-intelligence-research/ky26298/Projects/pure_checkpoints/VGGT-Omega/vggt_omega_1b_512.pt
```

期望权重 SHA256：

```text
c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
```

## 2. 先运行 trainval smoke 闭环

正式导出和特征缓存规模很大，先从 trainval 真实数据各取 4 个候选窗口：

```bash
python -m tools.data.build_navsim_manifest \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s_smoke.yaml \
  --train-output data/manifests/navsim_trainval_smoke_train_4.jsonl \
  --val-output data/manifests/navsim_trainval_smoke_val_4.jsonl \
  --report-output data/manifests/navsim_trainval_smoke_report.json \
  --max-scenes-per-split 4
```

分别生成缓存：

```bash
python -m tools.features.cache_vggt_omega_features \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s_smoke.yaml \
  --manifest data/manifests/navsim_trainval_smoke_train_4.jsonl \
  --output data/vggt_omega_cache/navsim_trainval_smoke_train_4_rich \
  --batch-size 1 --shard-size 4 --num-workers 0

python -m tools.features.cache_vggt_omega_features \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s_smoke.yaml \
  --manifest data/manifests/navsim_trainval_smoke_val_4.jsonl \
  --output data/vggt_omega_cache/navsim_trainval_smoke_val_4_rich \
  --batch-size 1 --shard-size 4 --num-workers 0
```

用两张 GPU 检查 DDP。`train.batch_size=4` 是每张 GPU 的 batch，因此 global batch 为 8：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s_smoke.yaml \
  --sources navsim \
  --output output/navsim_trainval_v4_ddp_smoke \
  --no-resume
```

成功时应看到 `Total trainable parameters: 2,401,181`、finite loss，以及 `best.pt`、
`best_trained.pt`、`last.pt`。每个 rank 打印一次参数数目是正常现象。

## 3. 流式导出完整 trainval manifest

```bash
python -m tools.data.build_navsim_manifest \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --train-output data/manifests/navsim_trainval_train_4s.jsonl \
  --val-output data/manifests/navsim_trainval_val_4s.jsonl \
  --report-output data/manifests/navsim_trainval_4s_report.json
```

trainval 比 mini 大很多。导出器会按 `navsim_export.log_batch_size` 分批调用官方 NAVSIM scene
filter，并把完成的窗口直接写入临时 JSONL；成功后再原子替换正式 manifest。它不会同时保留全部
log frames 或全部窗口。默认每批 8 个 log，可临时覆盖：

```bash
python -m tools.data.build_navsim_manifest \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --train-output data/manifests/navsim_trainval_train_4s.jsonl \
  --val-output data/manifests/navsim_trainval_val_4s.jsonl \
  --report-output data/manifests/navsim_trainval_4s_report.json \
  --log-batch-size 4
```

减小 batch 可以降低导出峰值内存，但会增加反复扫描 log 目录的开销。输出顺序按
`(group_token, sample_token)` 固定，不随 batch 大小改变。不要并发运行两个进程写同一组输出文件。

检查 report 与 log 隔离：

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path("data/manifests/navsim_trainval_4s_report.json").read_text()
)
train_logs = set(report["train_logs"])
val_logs = set(report["val_logs"])
assert train_logs.isdisjoint(val_logs)
assert report["train"]["num_windows"] > 0
assert report["validation"]["num_windows"] > 0
print("train logs/windows:", len(train_logs), report["train"]["num_windows"])
print("val logs/windows:", len(val_logs), report["validation"]["num_windows"])
print("train SHA256:", report["train_manifest_sha256"])
print("val SHA256:", report["val_manifest_sha256"])
PY
```

完整插值审计：

```bash
python -m tools.analysis.validate_navsim_interpolation \
  --manifest data/manifests/navsim_trainval_train_4s.jsonl \
  --output output/navsim_trainval_manifest_audit/interpolation.json
```

只有 `passed=true` 时才继续生成特征缓存。

## 4. 生成完整 VGGT-Omega 特征缓存

缓存工具支持断点续建、worker 内图像 LRU，以及按连续样本区间独立生成后合并。不要让两个进程写
同一个 partition 目录。当前 48GB RTX 4090 上的实测吞吐如下（真实 trainval 图像，2 workers，
LRU=64）：

| batch/GPU | 吞吐（windows/s） | 峰值 CUDA 显存 |
| ---: | ---: | ---: |
| 1（原脚本） | 5.75 | - |
| 2 | 约 10.6 | 4.59 GiB |
| 4 | 约 11.0 | 5.01 GiB |
| 8 | 约 10.1 | 5.87 GiB |

因此默认使用 `batch-size=4`，盲目继续增大 batch 不会更快。manifest 中每张图平均被约 4 个窗口
复用，`image-cache-size=64` 能避免重复 JPEG decode/resize。正式运行仍建议先在目标机器测 256 个
样本，因为共享存储负载会影响最优值。

### 4.1 单机两卡

优化后的脚本会先用两卡并行生成和合并 train，再用两卡并行生成和合并 validation：

```bash
bash scripts/temp.sh
```

可用 `tail -f output/navsim_trainval_cache_train_part_0.log` 查看速度。相同命令可续跑，每个 partition
会先验证已有 shard。

### 4.2 两台机器各一张 GPU

两台机器必须能看到相同的仓库、manifest 和共享 `CACHE_ROOT`。分别运行：

机器 A：

```bash
conda activate py312torch210cu126
NUM_PARTITIONS=2 CUDA_DEVICE=0 bash scripts/temp.sh all-part 0
```

机器 B：

```bash
conda activate py312torch210cu126
NUM_PARTITIONS=2 CUDA_DEVICE=0 bash scripts/temp.sh all-part 1
```

每台机器会依次生成自己负责的 train 和 validation 分片。两边完成后，在任一机器合并两个 set：

```bash
NUM_PARTITIONS=2 bash scripts/temp.sh merge-all
```

这样两个 set 都使用两张卡：每张卡约处理 241,974 个 train 窗口和 23,331 个 validation 窗口。
按实测吞吐估计总用时约 6.7 小时；合并只创建 hardlink 和索引，通常远短于特征提取。

合并默认使用 hardlink，不复制约 166 GiB 的 train 特征，因此所有 partition 和最终 cache 必须位于
同一文件系统。如果两台机器没有共享存储，先把 partition 目录汇总到同一文件系统，再手工使用
`merge_vggt_omega_feature_caches --mode copy`。

脚本默认写入 `/inspire/qb-ilm2/.../itae_nvsim_cache/vggt_omega_cache`，并在
`data/vggt_omega_cache` 创建与配置匹配的软链接。可在所有命令中用同一个 `CACHE_ROOT` 环境变量覆盖。

开始前用 `df -h "$CACHE_ROOT"` 检查空间。按当前 530,609 个窗口和 rich token 格式，train + val
约需 183 GiB，另需预留文件系统和临时写入余量。若目标目录在运行中掉载，写 shard 会报
`No such file or directory`；应先恢复挂载，再以完全相同的 partition 命令续跑。

确认两个 `index.json` 均完整：

```bash
python - <<'PY'
import json
from pathlib import Path

for split in ("train", "val"):
    path = Path(
        f"data/vggt_omega_cache/navsim_trainval_front_4s_{split}_rich/index.json"
    )
    index = json.loads(path.read_text())
    assert index["complete"] is True
    assert index["num_samples"] == index["expected_num_samples"]
    print(split, index["num_samples"], "samples", len(index["shards"]), "shards")
PY
```

## 5. 单机多卡正式训练

4 GPU 示例：

```bash
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --sources navsim \
  --output output/navsim_trainval_v4_scratch_4gpu \
  --no-resume
```

重要语义：

- `--sources navsim` 必须保留；基础配置沿用联合训练的 source schema；
- `train.batch_size=32` 是每个 DDP rank/GPU 的 batch，4 GPU 的 global batch 为 128；
- 当前 trainer 没有 gradient accumulation，也不会按 GPU 数自动缩放学习率；
- `data.num_workers=4` 同样是每个 rank 的数量，4 GPU 共创建 16 个 worker；
- motion-balanced sampler 先产生确定性全局采样流，再按 rank 分片，每个 epoch 会更新 seed；
- validation 指标跨 rank reduce，只有 rank 0 写 checkpoint 和 TensorBoard；
- 第一次实验使用独立且为空的 output 路径，避免误接其他实验的 checkpoint。

若显存不足，先把正式配置中的 per-GPU batch 从 32 降到 16 或 8。若数据读取跟不上 GPU，优先确认
cache 位于高吞吐文件系统，再逐步增加 `data.num_workers`，不要直接把 worker 数乘以 GPU 数写进
配置。

正式配置设置 `data.verify_cache_checksums=false`：cache 生成和断点恢复阶段仍会对每个 shard 做
SHA256 校验，但训练 worker 不会各自重复哈希数百 GB 文件。训练读取使用 safetensors 的单样本
切片和二分 shard 定位，不会为一个随机样本加载整个 shard。若缓存来自不可信复制，可先用相同的
cache 生成命令重新打开完整目录完成一次全量校验，再开始训练。

显式恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --sources navsim \
  --output output/navsim_trainval_v4_scratch_4gpu \
  --resume output/navsim_trainval_v4_scratch_4gpu/last.pt
```

恢复时保持 GPU 数、per-GPU batch、manifest 和 cache 不变。checkpoint 会严格检查数据 SHA256、
cache 路径、采样配置和模型结构。

## 6. Slurm 单节点多卡模板

```bash
#!/usr/bin/env bash
#SBATCH --job-name=itae-navsim-trainval
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=itae-navsim-trainval-%j.log

set -euo pipefail
cd /inspire/hdd/project/spatiotemporal-intelligence-research/ky26298/Projects/ITAE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py312torch210cu126

export OPENSCENE_DATA_ROOT=/inspire/hdd/global_public/public_datas/NAVSIM
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=4 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --sources navsim \
  --output output/navsim_trainval_v4_scratch_4gpu \
  --no-resume
```

`--cpus-per-task` 至少应覆盖 `GPU 数 × data.num_workers`，并为主进程和文件系统 I/O 留出余量。

## 7. 多节点扩展模板

先完成单节点多卡闭环。多节点时，每个节点启动一个 `torchrun`，并为所有节点提供相同的 rendezvous
地址、端口、节点数和不同的 node rank：

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4

srun --ntasks-per-node=1 bash -c '
torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --sources navsim \
  --output output/navsim_trainval_v4_scratch_multinode \
  --no-resume
'
```

所有节点必须看到完全相同的仓库、manifest、cache 和 output 文件系统。正式运行前先用 smoke 配置和
较短 wall time 验证 NCCL 网络与共享存储。

## 8. 评估与 TensorBoard

```bash
python -m tools.evaluation.evaluate_tokenizer \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --manifest data/manifests/navsim_trainval_val_4s.jsonl \
  --feature-cache data/vggt_omega_cache/navsim_trainval_front_4s_val_rich \
  --checkpoint output/navsim_trainval_v4_scratch_4gpu/best.pt \
  --output output/navsim_trainval_v4_scratch_4gpu/eval_navsim_trainval.json \
  --batch-size 32 --visualize-items 8

tensorboard \
  --logdir output/navsim_trainval_v4_scratch_4gpu/tensorboard \
  --host 127.0.0.1 --port 6006
```

重点检查 loss 是否 finite、expected/actual motion fraction、action token batch std、shuffled
condition gap、validation ADE/FDE/yaw，以及可视化中的 CAM_F0 和 BEV 方向。

## 9. 常见问题

### manifest 导出内存过高

把 `navsim_export.log_batch_size` 或 CLI `--log-batch-size` 从 8 降到 4/2。不要修改已经用于 cache
的 manifest；任何内容变化都会触发 cache SHA256 mismatch。

### cache 生成中断

使用完全相同的命令和输出目录重启。不要删除已完成 shard，也不要让多个进程写同一目录。

### DDP 启动后卡住

检查每个 rank 是否能读到相同 cache，`CUDA_VISIBLE_DEVICES` 数量是否等于
`--nproc_per_node`，以及 NCCL/防火墙配置。多节点尤其要确认 `MASTER_ADDR/MASTER_PORT` 可达。

### 恢复时报 provenance mismatch

恢复必须使用原 manifest、cache、source selection 和模型/loss 配置。新数据版本应使用新的 output
目录从头训练。
