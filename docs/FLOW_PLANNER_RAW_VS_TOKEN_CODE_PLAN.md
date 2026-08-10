# Flow Planner：原始轨迹点与 Action Token 对比实验 Code Plan

## 1. 目标与边界

本阶段不再修改 trajectory tokenizer 的结构，只实现一个受当前主视图图像条件约束的 flow
matching planner，对比两种预测空间：

1. **raw trajectory**：直接预测当前车体系下未来 4 秒、10 Hz 的 40 个原始轨迹点
   `[x, y, yaw]`，张量形状为 `[40, 3]`；
2. **action token**：预测冻结 tokenizer 的 4 个 action token，形状为 `[4, 192]`，再由冻结
   decoder 重建同样的 40 个 `[x, y, yaw]` 轨迹点。

planner 的视觉特征提取器必须是可配置组件，不能与 VGGT-Omega 写死。本次正式对照统一使用
冻结的 **PE-Spatial-B16-512** 提取当前帧特征；以后可以在不修改 planner 和 dataset 主逻辑的
前提下切换 VGGT-Omega 或其他 image encoder。

实验要回答两个问题：

- 哪种预测空间在相同训练样本数和 optimizer step 下更快收敛；
- 在严格相同的单帧输入、5-step 推理和评测样本上，哪种预测空间的最终轨迹质量更好。

本阶段不包括闭环驾驶、地图/导航指令、多相机、历史帧、LiDAR 点云和 tokenizer 架构搜索。
由于只输入当前主视图，路口意图等场景天然存在多解；主结果使用单样本预测，另外报告多样本
`minADE@6/minFDE@6` 作为多模态诊断，但后者不能作为部署指标。

## 2. Baseline tokenizer 选择

选择 V4 output-residual tokenizer：

- config：`configs/nuscenes_vggt_omega_front_4s_v4_output_residual.yaml`；
- checkpoint：`output/itae_vggt_omega_v4_output_residual/best.pt`；
- checkpoint SHA256：`78f11556263e56830a1604dada548a30643133c03bd3673ab695cb4c63d2f9a0`；
- checkpoint：epoch 25，global step 15756，记录的 val ADE 为 `0.324596 m`；
- action token：4 个 1 秒 token，每个 token 为 128-D motion token 与 64-D visual
  residual token 的拼接，共 192-D。

选择原因：这是目前单模型中 ADE/FDE 最好的 checkpoint，同时 register/pose 条件错配会使结果
退化，说明它确实使用了视觉条件，而不只是学到跨样本共享偏置。实现时必须校验 tokenizer
config 和 checkpoint 哈希；不允许在 planner 实验期间更新 tokenizer 的 encoder 或 decoder。

这里需要区分两条互不混用的视觉路径：

```text
planner condition: 当前一张 CAM_FRONT -> frozen PE-Spatial -> flow planner
token teacher:     完整五帧 rich VGGT cache -> frozen V4 encoder -> action-token label
```

VGGT-Omega 在本实验中只负责离线生成 V4 action-token 监督以及提供 frozen decoder，不是 planner
的图像条件。raw/token planner 都读取同一份 PE current-frame cache，避免 backbone 差异污染结论。

必须单独评测并记录 `GT action token -> frozen decoder` 的 oracle reconstruction。当前 V4 约有
`0.325 m ADE / 0.629 m FDE` 的重建下界，因此 token planner 的轨迹误差包含：

```text
tokenizer representation/reconstruction error + planner token prediction error
```

该 oracle floor 是解释结果的必要对照，不把它从每个样本的预测中人为扣除。

## 3. 可配置的当前帧视觉特征提取器

### 3.1 统一接口

新增 `PlannerVisionBackbone` 接口和 registry/factory。每种 backbone 只负责把当前 RGB 转成统一
结构，不允许把模型专属字段渗透进 flow planner：

```text
VisionCondition:
    tokens       [B, N, C]
    token_mask   [B, N]       # True 表示有效 token
    grid_size    [H, W] | null
    metadata     model/preprocess/checkpoint identifiers
```

planner 内部的 `ConditionProjector` 再将任意 `C` 投影到共同 `d_model`。支持两种运行方式：

- `mode: cached`：冻结 backbone，训练时读取离线 feature cache；
- `mode: online`：从当前 RGB 实时提取，供端到端调试或未来微调 backbone 使用。

首轮 PE 固定冻结并使用 cached mode；raw/token 两条分支必须引用同一份 cache。新增 backbone 时
只实现 adapter、preprocess 和 metadata 校验，不修改 planner core。

### 3.2 本次实验默认：PE-Spatial-B16-512

- model：`PE-Spatial-B16-512`；
- checkpoint：`/home/alan/AlanLiang/Projects/pure_checkpoints/PE-Spatial-B16-512/PE-Spatial-B16-512.pt`；
- checkpoint SHA256：`86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5`；
- 官方实现：`facebookresearch/perception_models`，作为固定版本的 third-party dependency；
- 输入：只读取 manifest 的 `image_paths[0]`，默认采用 PE 的 512 方形 resize 和 `[0.5]*3`
  mean/std；`resize_mode` 和 transform hash 写入 config/cache；
- 特征：`forward_features(image, strip_cls_token=True, layer_idx=...)`；默认最后一层；
- 原始输出：`[B,1024,768]`，对应 `32×32` patch grid；
- cache 前做固定二维 adaptive average pooling，默认 `32×32 -> 8×8`，保存
  `[B,64,768]`，同时保留二维 grid 信息；
- planner 投影后叠加可学习的 2D position embedding，再作为 cross-attention condition。

默认不把 PE-Spatial 压成单个 global token，因为道路边缘、车道和交通参与者的空间关系正是
选择 PE-Spatial 的原因。`pool_grid: [8,8]`、`layer_idx`、resize mode、cache dtype 都由 config
控制，未来可以做消融，但本次 raw/token 对照必须固定为同一设置。

当前 `py312torch210cu126` 尚未安装 `perception_models`；落实代码时在项目
`third_party/perception_models` 固定官方版本并以 editable/项目路径方式使用，禁止运行时从网络
隐式下载权重。加载本地 checkpoint 后检查 missing/unexpected keys，非空即 fail fast。

### 3.3 其他 backbone

首轮同时把接口写完整但不做额外 backbone 对照：

- `type: pe_spatial`：输出保序 patch-grid tokens；
- `type: vggt_omega`：必须以 `F=1` 独立运行，可将 camera/register special tokens 映射为统一
  `VisionCondition`；
- `type: custom`：通过 registry 添加 DINO、CLIP、其他 PE 型号等 adapter。

不能使用现有五帧 VGGT rich cache 的第 0 项作为 planner condition，因为该 hidden 已通过
attention 看过后四帧。无论选择何种 backbone，都必须对 `image_paths[0]` 独立前向。

cache metadata 保存 sample token、当前图像路径/时间戳、原始/池化 grid、preprocess、backbone
type/model、代码版本、checkpoint SHA256、manifest hash 和 dtype。训练启动时逐项校验，不允许
静默混用其他 backbone 的 cache。

线上语义为 `current RGB -> configured frozen vision backbone -> flow planner`；离线 cache 仅用于
节省训练时间。两条分支不输入未来图像、ego pose/speed、轨迹、pose token 或 route command。

新增 future-leakage 测试：修改同一 record 的 `image_paths[1:]` 后重新取 planner condition，要求
condition bitwise 相同；只修改 `image_paths[0]` 时 condition 应发生变化。

## 4. 两种监督目标

### 4.1 Raw trajectory 分支

- 直接读取 manifest 的 `trajectory [40,3]`；
- `[x,y]` 是以 anchor LiDAR ego pose 为原点的累计绝对坐标，不改成 increment、velocity 或
  spline control point，确保实验名称中的“原始轨迹点”含义不变；
- `yaw` 在每个样本内先 unwrap，再做标准化；生成后反标准化并 wrap 到 `[-pi, pi)`；
- 使用每个样本实际的 `future_times_s`，而不是假设时间戳严格等于 0.1 秒整数倍。

### 4.2 Action token 分支

- 使用现有五帧 rich cache 和冻结 V4 encoder 离线导出 teacher target `[4,192]`；
- teacher 制作 label 时可以看完整 5 帧窗口，这是 action token 的定义；planner 的 condition
  仍只允许看当前帧；
- 导出时同时用 frozen decoder 计算 oracle reconstruction 并保存/汇总 ADE、FDE；
- token cache 保存 sample token、tokenizer checkpoint/config hash、源 rich-cache hash、
  `future_times_s` 和 FP32 action token；
- planner 推理只生成 action token，decoder 的 `decode(action_tokens, future_times)` 不接收任何
  视觉信息。

### 4.3 归一化

只用 planner train split 拟合并固化统计量：

- raw：保存 `[40,3]` 的逐时间点/逐分量 mean、std；
- token：保存 `[4,192]` 的逐 slot/逐 channel mean、std；
- std 设可配置下限，避免近常量维度放大数值误差；
- metadata 保存 split hash、样本数、dtype 和统计量 hash；
- 提供 normalize/denormalize round-trip 测试。

两个空间的 MSE 量纲和维度不同，**不得用 raw flow loss 与 token flow loss 的绝对值判断谁更好**。

## 5. 数据划分与泄漏控制

沿用当前 nuScenes 4 秒 manifest：

- tokenizer train：19,370 个窗口；
- tokenizer val：4,017 个窗口。

planner 的固定划分方案：

1. 按 `scene_token` 对 tokenizer train manifest 做确定性 90%/10% 划分；
2. 90% 为 planner train，10% 为 planner validation；
3. 现有 tokenizer val manifest 只用于最终 paired evaluation；
4. 保存三个实际 JSONL 文件和 split summary，禁止按窗口随机划分，避免相邻重叠窗口跨 split；
5. raw/token 两条分支使用完全相同的 sample token 集合与顺序。

限制说明：V4 tokenizer 曾使用当前 val aggregate 指标选择 best checkpoint，因此最终 val 对
tokenizer 本身不是全新的无偏 test；但 planner 没有在该 split 上训练，两条 planner 仍可做成对
比较。未来加入其他数据集时，应预留一个 tokenizer 和 planner 都从未用于训练/选型的独立 test。

## 6. Flow matching 定义

令归一化后的数据目标为 `x1`，同形状高斯噪声为 `x0 ~ N(0,I)`，采样
`t ~ Uniform(0,1)`：

```text
x_t = (1 - t) * x0 + t * x1
v_target = d(x_t)/dt = x1 - x0
loss = mean((v_theta(x_t, t, current_frame_condition) - v_target)^2)
```

网络预测 velocity `v`，不预测 epsilon、score 或 clean sample。loss 先对每个样本的所有有效
元素取 mean，再对 batch 取 mean，避免 token 分支仅因元素更多而改变样本权重。当前数据均为
40 个有效点，但实现仍保留 mask 检查并对非法/非单调 `future_times_s` fail fast。

## 7. Planner 模型与公平性

实现一个供两条分支共用的 conditional Transformer/DiT core：

- raw 输入序列：40 个 3-D trajectory slots；
- token 输入序列：4 个 192-D action-token slots；
- 分支专属部分仅为 input/output adapter 和 slot/time embedding；
- 共享设计：`d_model=256`、8 blocks、8 heads、MLP ratio 4、dropout 0.1；
- diffusion time 使用 Fourier/sinusoidal embedding，经 MLP 注入每个 block；
- 配置的视觉 backbone tokens 先通过 `ConditionProjector` 投影到 `d_model`；PE 默认使用 64 个
  保留 `8×8` 空间位置的 condition tokens；
- condition tokens 通过相同的 cross-attention 注入 raw/token planner，diffusion time 可另外通过
  AdaLN 注入；
- raw slot embedding 使用实际 `future_times_s`；token slot embedding 使用四个 1 秒区间中心；
- 所有尺寸均写入 config，以上只作为首轮默认值。

两条分支的 core、初始化规则、optimizer、scheduler、EMA、precision 和训练预算完全相同。实现后
输出可训练参数明细；总参数差要求不超过 5%。正常情况下差异只来自很小的 adapter。如果超过
5%，先缩小/匹配 adapter，不通过给弱分支随意增加专属网络来追平。

默认训练设置：AdamW、bf16、gradient clipping、相同 batch size、相同 warmup/learning rate、
相同固定 max optimizer steps、EMA。best checkpoint 对两条分支都按 planner-validation 上
**反标准化并解码后的 trajectory ADE** 选择，而不是按各自 flow loss 选择。

## 8. 严格的 5-step 推理

主实验从同形状固定高斯噪声 `x(0)` 出发，使用等间隔 Euler ODE：

```text
for i in 0..4:
    t_i = i / 5
    x = x + (1 / 5) * v_theta(x, t_i, condition)
return x  # approximation of x(1)
```

这里 5 step 必须等于 **5 次网络 velocity evaluation（NFE=5）**。主实验不启用 CFG，也不用
每步调用两次网络的 Heun；否则表面上 5 个 solver step 实际不再是 5 NFE。evaluator 中增加
NFE counter 和 assertion。raw/token 对相同 sample 和 seed 使用对应的固定 noise bank；由于
两者 shape 不同，不能要求噪声数值逐元素相同，但 RNG seed 和采样协议必须相同。

主结果每个输入生成一次（`K=1`）。`K=6` 只用于额外的多模态 upper-bound 诊断，必须标为
oracle min metric，不参与“5-step 单次推理谁更好”的结论。

## 9. 训练协议与收敛定义

为了让“收敛更快”可复现，预先固定：

- seeds：42、43、44；
- 每个 seed 相同的数据顺序、batch size、optimizer steps 和 evaluation interval；
- 主比较按 optimizer step 和累计 seen samples 对齐；
- 固定预算训练，不允许某个分支因早停而少看数据；best 与 last 均保存；
- 默认每 500 optimizer steps 在固定 planner-validation 子集/噪声上评测，完整 validation 的
  频率可更低并通过 config 设置；
- wall-clock、samples/s、显存和推理 latency 作为效率指标单列，不与 sample-efficiency 混合。

“更快”的主指标：

1. validation ADE-vs-seen-samples 曲线的 AUC（越小越好）；
2. 首次达到预注册 ADE 阈值 `1.0/0.75/0.5 m` 所需 seen samples，未达到记为 N/A；
3. 前 10%、25%、50%、100% 训练预算处的 ADE/FDE。

“更好”的主指标：三个 seed 的 final-test ADE/FDE mean ± std，以及 paired scene-level bootstrap
95% CI。不要根据一次 seed 的最优 step 下结论。

建议先做 seed 42 的小预算 smoke/趋势实验，所有验收通过后再跑三 seed 正式实验；smoke 结果
不得替代正式结论。

## 10. 评测、TensorBoard 与报告

两条分支最终都转换为同一 `[40,3]` trajectory 后复用现有 evaluator，记录：

- ADE、FDE、1/2/3/4 秒 keyframe ADE；
- yaw、speed、yaw-rate、acceleration、jerk error；
- stationary / straight-slow / straight-fast / turn 及速度趋势 buckets；
- collision/route 等当前没有 GT 或地图支持的指标不伪造。

token 分支额外记录：

- teacher token 的 oracle decoded ADE/FDE；
- planner predicted-token decoded ADE/FDE；
- predicted token 到 teacher token 的标准化误差；
- predicted trajectory 相对 oracle reconstruction 的 excess ADE/FDE，用于判断瓶颈主要来自
  tokenizer floor 还是 planner。

TensorBoard 使用相同固定 sample token、相同评测 seed：

- train/val flow loss（raw 与 token 分开看）；
- decoded ADE/FDE、收敛 AUC、LR、gradient norm、EMA/non-EMA 差异；
- 当前主视图以及 BEV 中 GT/raw/token/oracle-token 四条轨迹；
- 复用 `tools/visualize_trajectory_video.py` 的轨迹配色，输出 2×2 图或短视频；
- 失败样本按 FDE、turn、accelerating 等分类，避免只展示手选成功案例。

比较工具输出机器可读 `comparison.json`、逐样本预测文件和 Markdown summary，报告 config/hash、
checkpoint、seed、NFE、参数量、数据 split 以及上述统计置信区间。

## 11. 计划新增的代码与配置

```text
src/vision_action_tokenizer/data/planner_dataset.py
src/vision_action_tokenizer/models/vision_backbones/base.py
src/vision_action_tokenizer/models/vision_backbones/factory.py
src/vision_action_tokenizer/models/vision_backbones/pe_spatial.py
src/vision_action_tokenizer/models/vision_backbones/vggt_omega.py
src/vision_action_tokenizer/models/flow_planner.py
src/vision_action_tokenizer/flow_matching.py
src/vision_action_tokenizer/planner_evaluator.py

tools/split_planner_manifests.py
tools/cache_planner_vision_features.py
tools/cache_tokenizer_action_targets.py
tools/train_flow_planner.py
tools/evaluate_flow_planner.py
tools/compare_flow_planners.py

configs/planner/nuscenes_flow_raw_4s.yaml
configs/planner/nuscenes_flow_token_v4_4s.yaml

tests/test_planner_dataset.py
tests/test_planner_vision_backbones.py
tests/test_flow_matching.py
tests/test_flow_planner.py
tests/test_flow_planner_evaluator.py
```

两个 config 采用相同 schema，只允许下列 target-specific 字段不同：

```yaml
vision_condition:
  type: pe_spatial            # pe_spatial | vggt_omega | custom registry name
  mode: cached                # cached | online
  model_name: PE-Spatial-B16-512
  source_path: third_party/perception_models
  checkpoint_path: /home/alan/AlanLiang/Projects/pure_checkpoints/PE-Spatial-B16-512/PE-Spatial-B16-512.pt
  checkpoint_sha256: 86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5
  freeze: true
  image_size: 512
  resize_mode: squash
  layer_idx: -1
  strip_cls_token: true
  pool_grid: [8, 8]
  cache_dtype: float16
  cache:
    train: null
    val: null
    final_eval: null

planner:
  target: raw_trajectory       # 或 v4_action_token
  target_shape: [40, 3]        # 或 [4, 192]
  solver: euler
  inference_steps: 5
  expected_nfe: 5
```

其余公共字段在启动时生成 stable hash 并做差异检查。每个 output 保存完整 resolved config、git
commit/dirty 状态、manifest/cache/checkpoint hash 和 normalization 文件，支持 `resume: auto`。

## 12. 实施顺序

### Phase A：数据和 target 闭环

1. 按 scene 生成 planner train/validation/final-eval manifests；
2. 实现通用 vision-backbone adapter/cache schema，并接通本地 PE-Spatial checkpoint；
3. 对每个 split 生成真正单帧的 PE current-frame cache；
4. 用五帧 VGGT rich cache 和冻结 V4 生成 action target cache 与 oracle reconstruction；
5. 拟合并保存两类 train-only normalization；
6. 做 sample-token 对齐、hash、shape、finite、时间戳与 future-leakage 检查。

### Phase B：flow planner 与单元测试

1. 实现 rectified/conditional flow matching objective；
2. 实现共享 Transformer core 和两类 adapter；
3. 实现 5-NFE Euler sampler、EMA 与 checkpoint；
4. 用 toy distribution 验证 velocity target 和 ODE 方向；
5. 检查参数量差异与 raw/token normalize/decode round trip。

### Phase C：小样本可学习性验收

1. 分别 overfit 相同的 8 个样本，再 overfit 64 个样本；
2. 要求 train decoded ADE 明显下降，打乱 current-frame condition 后不能改善；
3. 可视化固定样本，排除坐标系、yaw wrap、token 顺序和时间戳错误；
4. assertion 确认推理恰好调用网络 5 次。

### Phase D：正式对照

1. 先运行 seed 42 pilot，确认两条曲线数值稳定；
2. 固定配置后运行 42/43/44 三个 seed；
3. 在 final manifest 上一次性导出 paired predictions；
4. 生成 AUC、阈值步数、最终指标、bootstrap CI、效率和可视化报告。

## 13. 开始正式训练前的验收门槛

- current-frame condition 不随未来四张图变化；
- 本次 resolved config 的 `vision_condition.type` 必须为 `pe_spatial`，raw/token cache metadata 和
  checkpoint/transform hash 完全一致；
- PE 输出保持二维空间顺序，原始 `[1024,768]` 和默认池化后 `[64,768]` 的 shape 检查通过；
- PE 本地权重严格加载且全程冻结，训练过程无隐式下载；
- raw 分支读取的确实是 LiDAR ego 轨迹 `[40,3]`，时间严格递增且落在约 `(0,4] s`；
- V4 checkpoint/config hash 匹配，decoder 全程冻结；
- action target decode 与独立 tokenizer evaluator 数值一致；
- 两条分支 sample tokens、condition tensors 和 evaluation noise protocol 一致；
- normalization 只来自 planner train split，往返误差在数值容差内；
- 8/64 样本 overfit 通过，且固定可视化坐标/颜色正确；
- Euler sampler 的 `steps=5` 与 `NFE=5` assertion 通过；
- 两条模型总参数差不超过 5%；
- checkpoint 选择统一使用 decoded trajectory ADE；
- 单元测试、最小训练和单卡 resume 测试全部通过后，才启动正式三 seed 实验。

## 14. 结论判定规则

- **token 收敛更快**：三 seed 的 validation ADE AUC 更低，且达到预注册阈值所需样本更少；
- **token 最终更好**：5-NFE、K=1 下 final decoded ADE/FDE 和关键动态指标显著优于 raw，paired
  bootstrap CI 支持该差异；
- **token 受 tokenizer 限制**：predicted-token 已接近 teacher token，但 decoded trajectory 接近
  oracle floor 且仍弱于 raw；
- **raw 更适合当前数据量**：raw 在相同训练预算下收敛更快或 final 指标更好；
- 如果不同 seed 或不同 motion bucket 结论冲突，报告条件性结论，不用单一总 ADE 掩盖差异。

最终判断只基于共同轨迹空间的指标与收敛曲线，不基于两个不可直接比较的训练 loss。
