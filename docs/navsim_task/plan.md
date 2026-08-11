# NAVSIM + nuScenes Action Tokenizer 联合训练 Code Plan

## 1. 目标与本阶段边界

本任务在不改变当前 action tokenizer 时序语义的前提下，引入 NAVSIM 数据，并支持
NAVSIM 与 nuScenes 联合训练：

- action tokenizer 的监督目标统一为 anchor 后 `4.0s`、`10Hz`、共 `40` 个
  `[x, y, yaw]` 轨迹点；
- tokenizer 的视觉输入仍表示同一个 4 秒视觉窗口，统一为
  `t=[0, 1, 2, 3, 4]s` 的 5 张主视图，4 个相邻视觉区间对应 4 个 action token；
- NAVSIM 原生 2Hz 的 8 个未来 pose 用官方 PDMS 等价的 SE(2) 插值生成 40 点 GT；
- 使用当前效果最好的 V4 output-residual tokenizer 架构，但 tokenizer 所有参数随机初始化、
  全部参与训练，不加载旧 tokenizer/motion checkpoint；
- VGGT-Omega 仍作为冻结的预训练视觉特征提取器，并复用离线 feature cache。这里的
  “from scratch”指 action tokenizer from scratch，不是把 1B 视觉骨干也随机初始化；
- 整理 `tools/`，同时更新项目内所有命令、文档、测试和 VS Code 配置；
- mini set 用于数据和训练链路闭环。正式的数据量/泛化结论必须使用 NAVSIM
  `navtrain`，PDMS 则还需要对应的 metric cache，不能用 mini smoke 结果代替。

本阶段不改 tokenizer 架构，也不训练新的 planner。tokenizer 自身只能评估轨迹重建；
PDMS 是 planner 输出的闭环指标，当前阶段只保证数据插值和未来 NAVSIM agent/PDMS 接口兼容。

## 2. 已核查的事实与关键设计决定

### 2.1 本地数据和环境

- NAVSIM mini 位于 `/home/alan/AlanLiang/Dataset/navsim`，已检查到：
  - `navsim_logs/mini`：64 个 pkl log；
  - `sensor_blobs/mini`：64 个 log 目录，包含 `CAM_F0`；
  - `maps`：四个区域的 nuPlan map；
  - 随机检查的 log 时间差约 `500ms`，与 NAVSIM 2Hz 定义一致。
- `py312torch210cu126` 当前为 Python 3.12、PyTorch 2.10、NumPy 1.26，尚未安装
  `nuplan`。官方 NAVSIM requirements 固定了 PyTorch 2.0.1、NumPy 1.23.4、
  scikit-learn 1.2.2，不能直接覆盖当前训练环境中的核心包。

实施时优先在 `py312torch210cu126` 中以 editable + `--no-deps` 安装 NAVSIM，再安装
本任务读取/插值所需的最小兼容依赖，并执行 import/test gate。若 nuPlan 的旧依赖在
Python 3.12 上无法兼容，则单独创建 Python 3.10 的 `navsim` 数据导出/PDMS 环境；导出的
JSONL 和 feature cache 不依赖 NAVSIM runtime，训练仍在 `py312torch210cu126` 中完成。
禁止为安装 NAVSIM 降级现有 PyTorch/NumPy。

### 2.2 NAVSIM 数据读取

使用官方库的以下路径，不自行猜测 pkl 字段语义：

1. `SceneFilter(num_history_frames=1, num_future_frames=8, frame_interval=...)`
   建立当前帧加 8 个未来 2Hz 帧的窗口；
2. `SceneLoader` 读取 scene；`SensorConfig` 只加载 `CAM_F0` 的
   frame index `[0, 2, 4, 6, 8]`，即 `0/1/2/3/4s` 五张图；
3. `Scene.get_future_trajectory(num_trajectory_frames=8)` 获取 anchor rear-axle
   局部坐标中的 8 个 `[x,y,yaw]` pose；
4. exporter 额外检查窗口内 `scene_token` 恒定、timestamp 严格递增、图像存在，避免
   高重叠滑窗跨越原始 scene 边界。

窗口 stride 独立配置，mini smoke 默认 `frame_interval=1` 以测试高重叠样本；正式训练
默认也允许 stride 1，但 train/val 必须按完整 log 划分，不能随机拆 frame token，否则
相邻重叠窗口会泄漏。

### 2.3 统一的时序契约

| 项目 | nuScenes | NAVSIM | tokenizer 输入 |
|---|---:|---:|---:|
| 原始轨迹 pose 频率 | LiDAR 10Hz | key frame 2Hz | — |
| 原始未来 pose | 40 | 8 | — |
| 视觉候选频率 | key frame 1Hz | CAM_F0 2Hz | — |
| 选取的视觉时刻 | 0,1,2,3,4s | 0,1,2,3,4s | 5 frames |
| 统一 GT 时刻 | 0.1…4.0s | 0.1…4.0s（插值） | 40 poses |
| action token | 4 | 4 | 每 token 10 poses |

NAVSIM 多出的 `0.5/1.5/.../3.5s` 图像本次不进入 tokenizer，只用于原生轨迹和插值。
后续若要研究 2Hz 视觉窗口，应作为单独的模型时序 ablation，而不是混入本次数据集对比。

### 2.4 轨迹坐标与插值

统一数值约定为米/弧度、`x` 向前、`y` 向左、yaw 逆时针，所有点均相对 anchor ego。
NAVSIM 明确以 rear axle 为原点；nuScenes 当前 manifest 使用其数据提供的 ego reference。
manifest 必须记录 `coordinate_frame` 和 `reference_point`，不能暗中把二者当成同一物理点。
基线使用各数据集的 native ego reference；同时提供可配置的 dataset-specific 刚体
`reference_point_offset_m`，仅在有可验证标定时启用，并报告开启前后的 turn 样本差异。

NAVSIM 2Hz → 10Hz 的实现遵循以下规则：

1. 在 8 个未来点前加入 anchor identity pose `[0,0,0]`，时刻为
   `[0,0.5,...,4.0]s`；
2. `x/y` 按时间线性插值；yaw 先 unwrap，再插值，输出时 wrap 到 `[-pi,pi)`；
3. 在 `0.1,0.2,...,4.0s` 精确采样 40 点，不包含 `t=0`；
4. 新增官方 parity test：用同一条原生 NAVSIM `Trajectory` 经
   `navsim.evaluate.pdm_score.transform_trajectory` / nuPlan
   `InterpolatedTrajectory`，再按官方 `TrajectorySampling(40, 0.1)` 采样；与项目实现比较；
5. 要求所有原生 0.5s knot（40 点中的 index 4/9/.../39）及终点严格对齐，yaw 跨
   `±pi` 的测试轨迹不能发生跳变。

这样既模仿了 PDMS 的真实插值路径，也避免对 wrapped yaw 直接做错误线性插值。

## 3. 配置设计

新增联合训练配置，例如
`configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml`。数据源、采样、坐标和插值全部
通过 config 控制，不把本机路径或比例写死在 Python 中：

```yaml
data:
  frame_offsets_s: [0.0, 1.0, 2.0, 3.0, 4.0]
  future_horizon_s: 4.0
  trajectory_hz: 10
  sources:
    nuscenes:
      train_manifest: data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl
      val_manifest: data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl
      train_feature_cache: /path/to/nuscenes/train/cache
      val_feature_cache: /path/to/nuscenes/val/cache
      reference_point_offset_m: [0.0, 0.0]
    navsim:
      train_manifest: data/manifests/navsim_mini_train_4s.jsonl
      val_manifest: data/manifests/navsim_mini_val_4s.jsonl
      train_feature_cache: /path/to/navsim/train/cache
      val_feature_cache: /path/to/navsim/val/cache
      reference_point_offset_m: [0.0, 0.0]
  sampling:
    strategy: balanced       # proportional | balanced | weighted
    source_weights:
      nuscenes: 0.5
      navsim: 0.5
    samples_per_epoch: null  # null = 所有 source 样本数之和

navsim_export:
  data_root: /home/alan/AlanLiang/Dataset/navsim
  split: mini
  camera: CAM_F0
  native_hz: 2
  num_history_frames: 1
  num_future_frames: 8
  frame_interval: 1
  visual_frame_indices: [0, 2, 4, 6, 8]
  split_unit: log
  train_fraction: 0.8
  split_seed: 42
  interpolation: pdms_se2

train:
  resume: never
  initial_checkpoint: null
  freeze_base: false
  checkpoint_metric: val/macro/loss
```

`resume: never` 表示默认拒绝 output 中的旧权重；只有用户显式传 `--resume` 时才恢复本次
同配置实验。checkpoint 中保存 config hash、所有 manifest/cache hash、source sampler
状态和 source 列表，避免错误续训。

## 4. 数据层改造

### 4.1 通用 manifest

将当前 nuScenes 专用命名抽象为通用 action window schema，保留旧 JSONL 的向后兼容：

- `sample_token` 使用命名空间，例如 `nuscenes:<token>`、`navsim:<token>`；
- 新增 `dataset_name`、`group_token`（nuScenes scene / NAVSIM log）、
  `coordinate_frame`、`reference_point`、`native_trajectory_hz`、`schema_version`；
- 继续保存绝对 image path、真实 image timestamp、`frame_times_s`、40 点 trajectory、
  `future_times_s`；
- NAVSIM 保存原生 8 点及其时间作为可选 provenance/debug 字段，训练只读取统一 40 点；
- manifest 生成报告包含接受/拒绝原因、每 source/场景样本数、timestamp 误差、速度/转弯
  分布、插值 knot 误差和文件 hash。

`NuScenesWindowDataset` 重命名/抽象为 `ActionWindowDataset`；旧类名临时作为兼容 alias。
sample 至少返回 `dataset_name`、`sample_token`、`group_token`，便于 sampler、metric 和
TensorBoard 按 source 聚合。

### 4.2 NAVSIM exporter 与划分

新增 NAVSIM exporter，流程为：

1. 检查日志、CAM_F0、map 根目录和环境变量；
2. 从官方 `SceneLoader` 生成候选窗口；
3. 先按 log name 以固定 seed 做 train/val 划分，再生成窗口；
4. 拒绝跨 scene、缺图、时间不单调、非有限 pose、长度/频率不符的窗口；
5. 调用唯一共享的 `resample_se2_trajectory` 生成 40 点；
6. 输出 train/val JSONL、stats JSON、split log 清单和若干可视化样本。

正式数据使用官方 navtrain trainval 划分，并在其中保留 log-disjoint validation；navtest、
navhard 只做评估，绝不进入训练。mini 的 hash split 只用于 smoke closure。

### 4.3 视觉 feature cache

把 VGGT-Omega cache 工具改为读取通用 manifest，而不是依赖 nuScenes dataset 类。各 source、
各 split 分别建 cache，不物理拼接大文件。每个 cache index 除现有 backbone/checkpoint/
preprocess/hash 外，再记录 `dataset_name`、manifest schema version、sample token 顺序和
frame times。训练时严格检查所有 source 的输出 tensor shape 与 5-frame 契约一致。

## 5. 联合采样与训练

新增 `MultiSourceActionDataset` 和支持 DDP 的确定性 source sampler：

- `proportional`：按实际数据量；
- `balanced`：每个 source 等概率；
- `weighted`：按 config 的归一化权重；
- source 权重与现有 motion bucket 权重相乘，但每个 epoch 的总样本数固定；
- rank 间无重复分片，`set_epoch()` 后确定性重排，resume 可恢复相同采样序列；
- 每 epoch 记录期望/实际的 source、motion bucket、speed-trend 占比。

联合训练不把 `dataset_name` 作为模型输入，避免 tokenizer 学成 dataset classifier；它只用于
采样与统计。V4 tokenizer 整体随机初始化，`freeze_base=false`，不执行 motion bootstrap、
checkpoint interpolation 或 residual-only freeze；VGGT-Omega 本身保持 frozen/cache-only。

验证同时输出：

- `val/nuscenes/*`、`val/navsim/*`：loss、ADE、FDE、yaw、body velocity、yaw rate、
  acceleration/jerk、motion bucket；
- `val/macro/*`：两个 source 等权平均，用于 best checkpoint；
- `val/weighted/*`：按训练采样权重平均，仅作参考；
- TensorBoard 固定抽取每个 source 的 straight/turn/stationary 样本，沿用现有轨迹颜色，
  显示五张 CAM_F0/CAM_FRONT 和 BEV GT/reconstruction。

## 6. `tools/` 整理方案

使用 `git mv` 做一次原子迁移，目录按目的而非数据集划分：

```text
tools/
  data/
    build_nuscenes_manifest.py
    build_navsim_manifest.py
    split_planner_manifests.py
    subset_manifest.py
    inspect_infos.py
  features/
    cache_vggt_omega_features.py
    cache_planner_vision_features.py
    cache_tokenizer_action_targets.py
    export_action_latents.py
  training/
    train_tokenizer.py
    train_flow_planner.py
    train_action_expert.py
  evaluation/
    evaluate_tokenizer.py
    evaluate_flow_planner.py
    evaluate_closed_loop.py
    compare_flow_planners.py
    summarize_flow_planner_seeds.py
  analysis/
    inspect_vggt_camera_motion.py
    interpolate_tokenizer_checkpoints.py
    validate_navsim_interpolation.py
  visualization/
    visualize_samples.py
    visualize_trajectory_video.py
    visualize_navsim_trajectory.py
```

各目录加入 `__init__.py`，文档统一使用稳定的 module CLI，例如：

```bash
python -m tools.data.build_navsim_manifest --config ...
torchrun --standalone --nproc_per_node=1 --module tools.training.train_tokenizer --config ...
```

同步更新 README、全部 docs、`.vscode/launch.json`、`scripts/` 和 tests 中的旧路径；对每个
entrypoint 执行 `--help` smoke，并用 `rg` 确认不存在有效文档/配置引用旧路径。历史 code
plan 中的文件名可保留原文，但在开头增加迁移提示。`third_party/navsim` 和
`third_party/SUV` 内容不改动。

## 7. NAVSIM 可视化与未来 PDMS 接口

新增可视化命令，随机/按 token 展示：

- 原生 8 个 2Hz GT 点和插值后的 40 个 10Hz 点叠加 BEV；
- 五张选中的 CAM_F0 主视图，标题包含真实 timestamp/time error；
- 可选导出 2×2 布局视频，颜色沿用当前 trajectory visualizer；
- 对转弯、yaw 跨 `±pi`、高速和静止样本分别抽检。

为后续 SUV experiments / PDMS 预留项目内 NAVSIM `AbstractAgent` adapter：模型若输出
40 点，可按 index `4,9,...,39` 还原为 8 个 2Hz pose，并构造官方 `Trajectory(8,0.5)`，
由 PDMS 再走官方插值；同时保留直接构造 `Trajectory(40,0.1)` 的调试模式。两条路径做
round-trip 对比并报告插值误差。SUV 的脚本只作为 official evaluation integration 参考，
不把 SUV 模型代码耦合进 tokenizer。

mini 当前没有声明可用的 metric cache，因此本阶段验收不伪造 PDMS 分数。拿到 navtest
metric cache 后，再用同一 adapter 跑 SUV/官方 PDM evaluator。

## 8. 实施顺序

### Phase A：环境与数据 preflight

1. 记录当前 conda 核心包版本；安装 NAVSIM 最小依赖，不降级训练栈；
2. 验证 `navsim.common.dataloader`、`nuplan` trajectory interpolation import；
3. 扫描 mini log/sensor/map 完整性并生成 machine-readable 报告；
4. 若 Python 3.12 不兼容，建立独立 NAVSIM export env，并记录可复现安装命令/lock。

### Phase B：通用 schema、NAVSIM exporter 与插值

1. 扩展 manifest schema 和 backward compatibility；
2. 实现唯一 SE(2) resampler 与官方 parity tests；
3. 实现 log-disjoint split 和 NAVSIM manifest exporter；
4. 生成可视化视频，人工确认坐标、时间顺序和 2Hz/10Hz knot 对齐。

未通过插值 parity 与可视化审计前，不进入 feature cache 和训练。

### Phase C：工具迁移与 feature cache

1. `git mv` 整理 tools 并更新所有引用；
2. 将 VGGT cache 改为通用 manifest；
3. 分别生成 NAVSIM train/val cache，并用 checksum/shape/token-order gate 验证；
4. 确认旧 nuScenes 单数据集命令和 checkpoint evaluation 不回归。

### Phase D：多源训练

1. 实现 MultiSource dataset/sampler、source-aware metrics 与 checkpoint metadata；
2. 新增 scratch joint config，强制不加载历史 tokenizer；
3. 依次跑 NAVSIM-only、nuScenes-only、1:1 joint 小样本 overfit；
4. 运行 mini + nuScenes 的完整 one-epoch smoke，确认 DDP、resume、TensorBoard、best/last
   checkpoint 和 standalone evaluator 闭环；
5. 输出正式 navtrain 联合训练 guide，包含 manifest、cache、train、eval 的完整命令。

### Phase E：评估兼容

1. 分 source 跑 tokenizer reconstruction evaluation；
2. 验证 40→8→官方 40 的 round-trip；
3. 加入 NAVSIM agent/PDMS 配置模板和依赖检查；
4. 等 navtest + metric cache 到位后再执行真实 PDMS，不将其列为 mini 阻塞项。

## 9. 测试与验收标准

### 单元/集成测试

- 2Hz→10Hz：输出严格为 `[40,3]`，时间严格为 `0.1...4.0`；
- constant velocity、constant yaw rate、静止、急转跨 `±pi`、非有限输入均有测试；
- 原生 0.5s knots 对齐误差：XY `<1e-6m`、yaw `<1e-6rad`；
- 与官方 PDMS interpolation parity：XY/yaw 最大误差目标 `<1e-5`，若官方内部状态插值造成
  更大差异则记录实测阈值和原因，不能静默放宽；
- split 无 log/group overlap，窗口不跨 scene，所有图像存在；
- 旧 nuScenes manifest 可读，新 schema hash/cache mismatch 会 fail fast；
- sampler 在单卡/DDP 下 source 比例符合配置且固定 seed 可复现；
- scratch checkpoint 中不存在任何旧 checkpoint provenance，所有 tokenizer 参数有梯度；
- `pytest`、ruff（仅本项目代码）和所有 tools `--help` 通过。

### 训练闭环

- 三个小样本 overfit（NAVSIM、nuScenes、joint）的 reconstruction loss 均明确下降；若任一
  source 无法 overfit，先定位坐标/cache/模型问题，不启动正式训练；
- joint smoke 每个 epoch 都能看到两个 source 的 train/val 样本和独立指标；
- TensorBoard 至少各可视化 4 个 NAVSIM 与 nuScenes item，图像顺序、BEV GT 和重建正确；
- `last.pt`、`best.pt` 可被 standalone evaluator 加载，resume 后 global step、sampler 和
  learning-rate schedule 连续；
- 正式 guide 能从空 manifest/cache 开始复现到一个可评估 checkpoint。

## 10. 预期主要文件变更

- `src/vision_action_tokenizer/data/manifest.py`：通用 schema、provenance 与兼容读取；
- `src/vision_action_tokenizer/data/dataset.py`：通用 dataset 和 multi-source dataset；
- `src/vision_action_tokenizer/data/trajectory.py`（新增）：共享 SE(2) resampler；
- `src/vision_action_tokenizer/data/navsim.py`（新增）：官方 loader adapter/export helpers；
- `src/vision_action_tokenizer/data/sampler.py`（新增）：DDP multi-source sampler；
- `src/vision_action_tokenizer/trainer.py`、loss/evaluator：source-aware 聚合与 scratch gate；
- `configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml`（新增）；
- `tools/`：按第 6 节整体迁移，并新增 NAVSIM build/validation/visualization entrypoints；
- `tests/`：插值 parity、NAVSIM exporter、schema、sampler、joint train smoke；
- README、实验 guide、`.vscode/launch.json`：更新新 module CLI 和完整实验流程。

## 11. 需要在实施前确认的方案点

默认按以下三点实施：

1. “from scratch”仅指 V4 action tokenizer 全量随机初始化，VGGT-Omega 保持预训练冻结；
2. NAVSIM 视觉帧降采样到与 nuScenes 相同的 `0/1/2/3/4s`，不改变 4-token 架构；
3. mini 只闭环数据/训练，真实 PDMS 等 navtest metric cache 到位后执行。

若这三点符合预期，后续实现将严格按 Phase A→E 推进。
