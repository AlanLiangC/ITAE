# 用 ITAE action tokenizer 微调 SUV

本文档给出从 teacher target、SUV action expert 微调，到 NAVSIM v1 闭环 PDM
评分的完整流程。默认使用：

- action tokenizer：`output/navsim_trainval_v4_scratch_4gpu/best.pt`；
- 原始 SUV：`/inspire/hdd/project/spatiotemporal-intelligence-research/ky26298/Projects/pure_checkpoints/SUV_ckpt/suv_navsim.pt`；
- NAVSIM：`/inspire/hdd/global_public/public_datas/NAVSIM`；
- 训练配置：`configs/suv_itae/navsim_trainval_action_tokens.yaml`。

## 1. 方案与边界

SUV 上游仓库没有发布训练脚本和训练配置。因此这里实现的是可复现的
`action-only flow matching` 微调，而不是声称复刻原作者未公开的训练过程。

数据流如下：

```text
当前 CAM_F0 ── Wan VAE + frozen video expert ── video KV cache ─┐
                                                               ├─ SUV action expert ── 4 x 192
静态 RGB text context ──────────────────────────────────────────┘
                                                                     │
                                                   frozen ITAE decoder
                                                                     │
                                                          40 x [x,y,yaw] @ 10 Hz
                                                                     │
                                                            NAVSIM v1 PDM score
```

训练时冻结 Wan VAE、SUV video expert 和 ITAE tokenizer，仅训练 SUV action
expert。原始 SUV action 输入/输出是 `8 x 3`，新模型是 `4 x 192`；两侧投影层
重新初始化，30层 action transformer 中形状兼容的权重从原始 SUV 加载。

训练和评测严格使用相同条件：

- 仅当前时刻的前视图，避免未来图像泄漏；
- 静态 RGB prompt；
- 不使用 proprio，因为 tokenizer trainval manifest 没有可靠的当前 11D
  ego-status；
- action target 使用训练集统计量逐槽归一化，评测时反归一化后送入冻结 decoder。

## 2. 环境

先更新 SUV/NAVSIM v1 环境。脚本现在也会把本项目以 editable 方式安装，从而让
评测进程能导入 `vision_action_tokenizer`：

```bash
bash scripts/suv/setup_navsim_v1_env.sh
```

检查环境：

```bash
conda run --no-capture-output -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 doctor --require-metric-cache
```

若 metric cache 尚未生成，先参考
`docs/suv/NAVSIM_V1_EVALUATION_GUIDE.md` 执行 `cache-metrics`。

## 3. 生成冻结的 action-token targets

这一步只执行一次。它使用现有 rich VGGT-Omega cache 和 tokenizer teacher，生成
与两个 manifest 顺序严格对应的 safetensors：

```bash
GPU=0 BATCH_SIZE=256 NUM_WORKERS=8 \
  bash scripts/suv_itae/prepare_targets.sh
```

输出为：

```text
output/suv_itae/action_targets/train.safetensors
output/suv_itae/action_targets/train.json
output/suv_itae/action_targets/validation.safetensors
output/suv_itae/action_targets/validation.json
```

命令会显示 `Caching ITAE action targets` 进度条，并在结束时打印 teacher
重建轨迹的 ADE/FDE。`.json` 保存 manifest hash、teacher hash、样本顺序和 oracle
指标；训练器逐样本校验 token 顺序，不能把别的 manifest cache 混用。

如果某次生成被中断并只留下 `.safetensors` 或 `.json` 之一，请删除该 split 的
不完整文件后重跑。正式 train cache 约含 48万条样本，写出前会在 CPU 内存中
聚合结果，应预留约 3GB 以上可用内存。

## 4. 生成静态 RGB 文本条件

已有动态 prompt cache 不能替代这里的静态 prompt。执行：

```bash
conda run --no-capture-output -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 precompute \
  --gpus 0 \
  --prompt-mode static
```

训练器会按 prompt 的 SHA256 精确查找 RGB context；找不到时会在加载 6B 模型前
直接报错。

## 5. 先做小闭环

建议先用两张卡跑少量样本，检查模型加载、反向传播、DDP、保存和续训：

```bash
GPUS=0,1 OUTPUT=output/suv_itae/smoke \
  bash scripts/suv_itae/train.sh \
  --overfit-samples 4 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-validation-batches 1 \
  --last-n-blocks 1 \
  --no-resume \
  --num-workers 0
```

这条命令用于验证完整 I/O、真实 SUV 反向传播、保存和 decoder 评测接口，不代表
正式训练配置。训练器在 `max_steps` 结束时一定执行验证和保存；正式配置另外每
250步执行一次。

训练启动日志应包含：

```text
SUV-ITAE world=2 samples=4 trainable=...
step=... loss=... sigma=... lr=... grad=... steps/s=...
validation step=... flow_loss=...
```

## 6. 正式多卡训练

单机两卡：

```bash
GPUS=0,1 bash scripts/suv_itae/train.sh
```

单机四卡：

```bash
GPUS=0,1,2,3 bash scripts/suv_itae/train.sh
```

脚本按 `GPUS` 数量设置 `torchrun --nproc_per_node`。针对当前两张48GB GPU，默认
每卡 batch size 为32、不做梯度累积；单机两卡的有效 batch 是
`32 x 2 x 1 = 64`。每个进程完整持有冻结的 SUV/Wan 模型，但 DDP 只同步
`requires_grad=True` 的 action expert 参数。

实际 profiling 结果如下，吞吐是两卡全局吞吐：

| 每卡 batch | 全局 batch | samples/s | 峰值 allocated | 峰值 reserved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 4.92 | 20.29 GiB | 20.86 GiB |
| 8 | 16 | 24.66 | 21.18 GiB | 22.14 GiB |
| 16 | 32 | 30.90 | 22.20 GiB | 23.17 GiB |
| 24 | 48 | 32.87 | 23.61 GiB | 24.83 GiB |
| **32** | **64** | **33.45** | **25.32 GiB** | **26.62 GiB** |
| 48 | 96 | 34.21 | 28.88 GiB | 30.70 GiB |
| 64 | 128 | 34.65 | 32.38 GiB | 34.68 GiB |

需要在另一种 GPU 或模型配置上重测时，可运行：

```bash
GPUS=0,1 OUTPUT=output/suv_itae/batch_profile \
  bash scripts/suv_itae/train.sh \
  --profile-batch-sizes 1,2,4,8,16,24,32,48,64 \
  --num-workers 8 --no-resume
```

profiling 会执行真实的 forward、backward、DDP all-reduce 和 AdamW step，但不会
运行 validation 或写入巨大的模型 checkpoint。

batch 32 已达到 batch 64 吞吐的96.5%，但少占约8GiB reserved memory，因此选它
作为速度、余量和稳定性的折中。相较原配置的全局有效 batch 16，配置同步采用线性
学习率缩放到 `4e-5`，并把 max steps 从30,000缩至7,500、warmup/eval/save间隔
同比缩短，保持总训练样本数和按样本计的调度位置基本不变。

### 两机训练

两台机器需要访问相同代码、checkpoint、manifest、target cache 和输出目录。选择
第一台机器可被第二台访问的地址，例如 `10.0.0.1`。

机器0：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
GPUS=0,1 bash scripts/suv_itae/train.sh
```

机器1：

```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
GPUS=0,1 bash scripts/suv_itae/train.sh
```

默认配置按两卡调优。如果四张卡仍希望维持全局 batch 64，应增加
`--batch-size 16`；否则每卡32会得到全局 batch 128，需要重新调整学习率和训练步
数。多机训练中只有 global rank 0 写 checkpoint，所有 rank 在验证/保存边界同步。

### 48GB 显存建议

默认训练全部30层 action expert，启用 action block gradient checkpointing，视觉
分支使用无梯度 KV cache。两张48GB卡已经实测每卡 batch 32。如果环境中还有其他
显存占用或输入配置改变导致 OOM：

1. 把 `last_n_blocks: null` 改成 `last_n_blocks: 8`，只训练最后8层和输出投影；
   输入、时间、文本投影与前22层一并冻结，使冻结前缀不保留反向图；
2. 先把每卡 batch 从32降到24或16；如需维持全局有效 batch，再增加梯度累积；
3. 不要关闭 BF16 或 action gradient checkpointing；
4. 降低 `num_workers` 只影响主机内存，不会降低模型显存。

`last_n_blocks: 8` 是节省显存和通信量的消融配置，正式结果应在报告中注明，不能
与“全部30层微调”混为同一设置。

### TensorBoard 收敛曲线与定性轨迹视频

默认配置已经启用 TensorBoard。训练启动时会打印实际日志目录：

```text
TensorBoard log_dir=.../output/suv_itae/navsim_trainval/tensorboard
```

在项目根目录另开终端启动服务：

```bash
conda run --no-capture-output -n suv-navsim1 \
  tensorboard \
  --logdir output/suv_itae/navsim_trainval/tensorboard \
  --bind_all \
  --port 6006
```

浏览器打开 `http://<训练机器IP>:6006`。如果只在本机访问，可去掉 `--bind_all`。
Dashboard 中包括：

- `train/flow_loss`、`validation/flow_loss`、学习率、gradient norm、采样速度；
- 当前显存 allocated/reserved；
- `qualitative_clip/*` 下的 clip ADE/FDE/yaw 指标，以及 `Images` 面板中的三段
  连续驾驶视频；
- `run/config`，记录本次训练的 resolved config 和固定 clip 的全部样本索引。

定性验证默认每250个 optimizer step 执行，和正式 validation 对齐。训练器从
validation set 固定挑选 turn、straight、dynamic 三段真实连续 clip；每段包含16个
按时间排序、相邻间隔不超过0.65秒的 NAVSIM window，通常对应2Hz、约7.5秒的连续
驾驶。相同 clip 会贯穿整次训练，因此可直接比较 step 250、500、750 等视频，而
不会把样本变化误认为模型进步。

视频的每一帧都是一个新的真实 `CAM_F0` 时刻，训练中的 SUV 会在该时刻重新执行
一次 action-flow 推理，再由冻结的 ITAE decoder 得到未来4秒轨迹。左侧在当前图像
上投影该时刻的 GT（绿色虚线）和模型预测（橙色实线），右侧显示同一对轨迹的
BEV；随着 clip 前进，相机图像、GT 和预测都会逐帧更新。它与
`tools/visualization/visualize_navsim_tokenizer_video.py` 的时间轴语义一致，区别是
橙色轨迹来自当前训练中的 SUV，而不是 tokenizer 对 GT action token 的重建。

同一批视频也保存为更适合下载查看的 H.264 MP4，并附逐样本指标：

```text
output/suv_itae/navsim_trainval/qualitative/step_00000250/*.mp4
output/suv_itae/navsim_trainval/qualitative/step_00000250/metrics.json
```

相关选项位于配置的 `tensorboard.qualitative`：

```yaml
tensorboard:
  enabled: true
  qualitative:
    enabled: true
    every_steps: 250
    num_clips: 3
    clip_frames: 16
    max_gap_s: 0.65
    inference_steps: 10
    inference_batch_size: 8
    fps: 2
    frame_size: [432, 768]
    save_mp4: true
```

完整采样和视频编码只在 global rank 0 上进行；其他 GPU 会在该 validation 边界
等待，所以不会重复写视频。每个 clip frame 的预测互相独立，`inference_batch_size`
只是把多个时刻并行计算以提高 GPU 利用率，不会给模型额外的未来帧输入。默认一次
定性验证共执行 `3 x 16 = 48` 个预测。若希望缩短验证时间，可先把 `num_clips` 改为1、
缩短 `clip_frames` 或把 `every_steps` 调大；`inference_steps` 越小越快，但也会改变
预测质量。若完全不需要视频，设置 `tensorboard.qualitative.enabled: false`，标量
曲线仍会保留。

要在一步 smoke training 中专门验证整条视频链路，可覆盖定性间隔和视频数：

```bash
GPUS=0 OUTPUT=output/suv_itae/tensorboard_smoke \
  bash scripts/suv_itae/train.sh \
  --overfit-samples 4 --max-steps 1 --batch-size 1 \
  --max-validation-batches 1 --last-n-blocks 1 --num-workers 0 --no-resume \
  --qualitative-every-steps 1 --num-qualitative-clips 1 \
  --qualitative-clip-frames 4
```

## 7. 断点续训与产物

直接重跑会自动读取输出目录中的 `last.pt`：

```bash
GPUS=0,1 bash scripts/suv_itae/train.sh
```

也可显式指定：

```bash
GPUS=0,1 bash scripts/suv_itae/train.sh \
  --resume output/suv_itae/navsim_trainval/last.pt
```

输出目录包含：

- `resolved_config.json`：实际路径、world size、样本数和可训练参数量；
- `manifest_indices/*.offsets.npy`：大 JSONL 的随机访问索引；
- `tensorboard/events.out.tfevents.*`：训练、验证标量与定性视频；
- `qualitative/step_*`：TensorBoard 视频对应的 MP4 和指标 JSON；
- `best.pt`：验证 flow loss 最优的轻量 adapter，不含 optimizer；
- `last.pt`：最近 adapter、optimizer、scheduler 和训练步数，用于续训。

adapter 不复制 12GB 原始 SUV video expert；评测时必须同时保留原始
`suv_navsim.pt`。`best.pt` 的选择依据是归一化 action-token flow loss，最终模型
选择仍应以完整 NAVSIM v1 PDM 结果为准。

## 8. NAVSIM v1 闭环评分

先做2个 scene 的端到端 smoke test：

```bash
GPUS=0,1 ADAPTER=output/suv_itae/navsim_trainval/best.pt \
OUTPUT_DIR=output/suv_itae/navsim_trainval/navsim_v1_smoke \
  bash scripts/suv_itae/evaluate_navsim_v1.sh \
  --max-scenes 2
```

每张卡会显示独立的 `GPU rank N` scene 进度条，并周期性打印完成数、成功数、失败
数、速度和 ETA。确认 smoke test 没有失败样本后跑完整 navtest：

```bash
GPUS=0,1 ADAPTER=output/suv_itae/navsim_trainval/best.pt \
  bash scripts/suv_itae/evaluate_navsim_v1.sh
```

评测加载顺序是：

1. 实例化 action_dim=192 的 SUV；
2. 加载原始 SUV 中所有形状兼容参数；
3. 覆盖训练得到的 action expert；
4. 加载 tokenizer `best.pt` 和 adapter 中的 normalizer；
5. SUV 生成 `4 x 192`，decoder 输出 `40 x 3 @ 10Hz`；
6. NAVSIM v1 PDM simulator/scorer 计算闭环分数。

汇总结果：

```bash
conda run --no-capture-output -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 summarize \
  --results output/suv_itae/navsim_trainval/navsim_v1_evaluation
```

`--max-scenes` 只用于功能检查，不能与完整 navtest 或论文数字直接比较。

## 9. 常见问题

### 找不到静态 text context

重新执行第4节，必须带 `--prompt-mode static`。动态 cache 即使目录非空，也不包含
训练所需的精确 prompt hash。

### target/manifest order mismatch

target cache 与 manifest 不配套。删除对应 split 的 `.safetensors` 和 `.json`，用
当前配置重新执行 `prepare_targets.sh`，不要手动排序或拼接 metadata。

### OOM 发生在模型初始化

确认每个进程只看到一张本地 GPU；应使用 `scripts/suv_itae/train.sh`，不要在已经
设置全局多卡可见的 Python 进程里再次派生模型。再按第6节设置 `last_n_blocks: 8`。

### OOM 发生在 optimizer.step

AdamW 状态在第一次 step 才完整创建。降低可训练 block 数；降低梯度累积不会减少
optimizer 状态显存，降低 `batch_size` 才减少 activation，但默认已经是1。

### 评测轨迹点数为何从8变成40

原始 SUV 直接预测2Hz的8个姿态；ITAE adapter 预测4个连续 token，每个 token
解码10个0.1秒增量，因此输出40点。两者覆盖的物理时长同为4秒，NAVSIM 的
trajectory sampling 会按时间戳交给 simulator。

## 10. 已验证的最小闭环

项目内已实际完成以下验证，不只是命令 dry-run：

- 单卡 last-block：1步训练、1个 validation batch、adapter 保存；
- 从 `last.pt` 恢复 optimizer/scheduler，由 step 1 推进至 step 2；
- 单卡全30层：真实反向更新 1,021,265,088 个参数并保存 checkpoint；
- 双卡 DDP：last-block 和完整30层两种模式均完成真实反向、聚合验证指标，并只由
  rank 0 写 checkpoint；完整模式同步 1,021,265,088 个可训练参数；
- NAVSIM v1：使用双卡完整30层 smoke adapter，1个真实 scene 成功完成
  `4x192 -> 40x3 -> PDM scorer`，有效样本 `1/1`、失败 `0`。

验证产物位于：

```text
output/suv_itae/closed_loop_smoke
output/suv_itae/closed_loop_full_smoke
output/suv_itae/closed_loop_ddp_smoke
output/suv_itae/closed_loop_ddp_full_smoke
```

这些模型只训练了1至2步，只用于接口和数值闭环验证，分数不具备模型质量意义。
