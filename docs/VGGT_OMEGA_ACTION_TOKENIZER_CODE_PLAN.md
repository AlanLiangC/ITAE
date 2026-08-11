# VGGT-Omega Action Tokenizer 重构计划

> 状态：**Implemented / 2026-08-07**  
> 主训练路径已完成重构；本文保留最初的设计理由和验收门槛，并在末尾记录实测结果。

## 1. 目标与结论

用 `VGGT-Omega-1B-512` 的多帧几何表征完全替换现有 PE + 双分支 tokenizer。新模型只保留
一条清晰的数据路径：

```text
5 × CAM_FRONT frames
    -> frozen VGGT-Omega Aggregator
    -> pretrained CameraHead trunk
    -> 4 interval action tokens
    -> context-free SE(2) increment decoder
    -> 40 × [x, y, yaw]
```

推荐的 action token 不是任意 learned query，而是与四个 1 s 运动区间一一对应：

```text
z0: [t0, t1] -> trajectory points 00..09
z1: [t1, t2] -> trajectory points 10..19
z2: [t2, t3] -> trajectory points 20..29
z3: [t3, t4] -> trajectory points 30..39
```

默认 latent shape 从 `[B,10,64]` 改为 `[B,4,128]`。每个 token 解码 10 个 10 Hz
SE(2) 增量，再顺序积分成 40 点轨迹。这样 token、视觉帧间运动和轨迹时间段具有明确对应，
更适合当前 1 Hz 图像、10 Hz LiDAR、小数据量 setting。

新版本不再保留以下核心设计：

- PE backbone 和 spatial resampler；
- visual/trajectory 两个 encoder 互相对齐；
- stochastic posterior、KL 和跨模态 contrastive alignment；
- learned action queries；
- 带零速 clamp 的 kinematic decoder。

原因是新模型只有一个预训练几何 encoder，action token 由它直接产生，不再需要另一个可移动的
trajectory teacher。监督直接落到轨迹、每步 SE(2) 增量和 1 s keyframe endpoint 上。

## 2. Gate 0：实施前必须解决的问题

以下任一项未解决，都不开始修改主训练代码。

### 2.1 许可证确认

`third_party/vggt-omega/LICENSE` 是 **FAIR Noncommercial Research License**，不是常规
Apache/MIT 许可证。其 Acceptable Use Policy 在“可能造成人身风险”的类别下明确列出
`transportation technologies`。本项目是自动驾驶 action/trajectory 研究，可能落入该限制。

因此需要项目负责人确认：

1. 当前用途是否获得该许可证和 AUP 的授权；
2. 是否仅限允许的非商业研究；
3. 后续论文、模型发布和衍生权重是否满足归属及再分发要求。

这不是法律意见；在得到明确许可结论前，不将 VGGT-Omega 接入训练或产出衍生 checkpoint。

### 2.2 checkpoint 路径

当前仓库和 `/home/alan/AlanLiang/Projects` 下没有找到
`vggt_omega_1b_512.pt`，`docs/task.md` 中“模型权重在”之后也没有路径。配置需要一个可读的
绝对路径，并在任何特征提取前完成：

- 文件存在及大小检查；
- SHA256 记录；
- 只加载 `aggregator.*` 和 `camera_head.*`；
- 对这两个模块做严格 key/shape 校验；
- checkpoint 中允许存在未使用的 dense/text head，但不允许 camera 分支缺 key。

### 2.3 Python / PyTorch 环境

现有 `openmmlab` 环境为 Python 3.9.20、PyTorch 2.1.0；VGGT-Omega 声明需要
Python 3.10+、PyTorch 2.3+，源码也使用了 Python 3.10 union type 语法。

不建议直接升级现有 `openmmlab` 环境。新建独立环境，例如 `itae-vggt`：

```text
Python 3.10 或 3.11
PyTorch >= 2.3（与本机 CUDA/driver 匹配）
torchvision 与 torch 严格配套
pip install -e third_party/vggt-omega
pip install -e ".[dev,viz]"
```

环境验收需要包含：import、checkpoint strict load、单窗口 forward 和 CUDA BF16 测试。

### 2.4 显存约束

本机 RTX 4070 Laptop 只有约 7.6 GiB。第三方 README 给出的 1 帧和 10 帧完整推理峰值分别
约 6.02 GB 和 6.67 GB（A100 测试）；5 帧在线 forward 已接近本机上限。因此：

- VGGT-Omega 只用于离线、batch-size=1 的窗口特征提取；
- 一个窗口内的 5 帧不能拆开 forward，因为 Aggregator 有跨帧 attention；
- 构造模型时关闭 depth/text head，只加载 aggregator + camera head；
- 正式 tokenizer 训练默认只读取缓存，不在线运行 1B backbone；
- 若单窗口仍 OOM，先用 `max_size` 的 `512×288` 输入，不能通过拆帧破坏时序几何。

## 3. 对 CameraHead 的准确理解

本地 VGGT-Omega 默认参数为：

```text
embed_dim              = 1024
aggregated feature dim = 2 × embed_dim = 2048
camera tokens/frame    = 1
registers/frame        = 16
patch_token_start      = 17
camera trunk           = 4 × SelfAttentionBlock(dim=2048, heads=16)
camera output          = 9D = translation(3) + quaternion(4) + FoV(2)
```

Aggregator 最后一层输出为 `[B,F,N,2048]`。CameraHead 实际执行：

```text
special = final_tokens[:, :, :17]              # [B,F,17,2048]
special = token_norm(special)
special = reshape(special, [B,F*17,2048])
special = 4 × pretrained self-attention block
hidden  = reshape(special, [B,F,17,2048])
camera  = trunk_norm(hidden[:, :, 0])           # [B,F,2048]
pose    = MLP(camera)                            # [B,F,9]
```

特别注意：demo 返回的 `predictions["camera_and_register_tokens"]` 是 Aggregator 输出、
即 CameraHead **输入**，并没有经过 CameraHead 自己的四层 trunk。不能直接把 demo 字段当作
CameraHead 编码结果。

实现时新增项目侧 wrapper，不直接魔改 third-party 文件：

```python
OmegaCameraFeatureExtractor.forward(images) -> {
    "camera_hidden": [B, F, 2048],
    "register_hidden_mean": [B, F, 2048],
    "pose_enc": [B, F, 9],
}
```

wrapper 复用预训练的 `token_norm`、四层 `trunk` 和 `trunk_norm`；`trunk_norm` 同时用于
camera token 和 mean-register token，所有输出均在缓存前转成 FP16。

## 4. 输入预处理

现有 PE 数据管线把图像 letterbox 成 `512×512` 并归一化到 `[-1,1]`，不能复用到
VGGT-Omega。新预处理必须和官方 `load_and_preprocess_images` 一致：

- 输入 RGB `[0,1]`，不要在 dataset 中做 ImageNet normalize；
- Aggregator 内部自行使用 `_RESNET_MEAN/_RESNET_STD`；
- 默认 `mode=max_size`、`image_resolution=512`、`patch_size=16`；
- nuScenes CAM_FRONT 的 16:9 图像得到约 `512×288`，保留完整横向 FOV；
- 5 帧必须执行完全相同的 resize/crop，并在 metadata 中记录最终 H/W；
- 禁止将已有的 `LetterboxNormalize` tensor 再输入 VGGT-Omega。

Dataset 继续输出原有标签契约：

```text
images          [B,5,3,288,512]
frame_times     [B,5]       # 实际 keyframe 时间
trajectory      [B,40,3]    # anchor ego frame, x forward, y left
future_times    [B,40]      # 实际 LiDAR sweep 时间
trajectory_mask [B,40]
```

## 5. 先做几何可用性审计

在训练 tokenizer 前新增 `tools/analysis/inspect_vggt_camera_motion.py`，对 train/val 各抽取直行、静止、
转弯和长距离样本，执行：

1. 从 `pose_enc` 解码 `camera-from-world` extrinsics；
2. 明确矩阵方向并计算相对第一帧的 camera motion；
3. 用 nuScenes 固定 CAM_FRONT 外参将 LiDAR ego GT 转到相同相机约定；
4. 分别评估 rotation、translation direction 和 scale；
5. 只在 train split 拟合一个全局 axis/scale calibration，并原样应用到 val；
6. 输出输入 contact sheet、VGGT camera path 和 LiDAR GT 的 BEV/3D 可视化。

VGGT-Omega 的单目多视图重建可能存在 gauge/scale ambiguity，所以不能直接把 9D pose translation
当作米制 ego trajectory，也不能在每个 val window 上单独做 Sim(3) 对齐后再报告指标。逐窗口
对齐只用于诊断；可用于模型输入的 calibration 必须只由 train split 估计。

几何审计的 go/no-go 条件：

- 静止、直行、转弯的相对旋转和位移方向能够被区分；
- val 上的全局 train-fitted calibration 明显优于“始终静止”和 train-mean trajectory；
- pose 不出现 NaN、四元数退化或 frame order 反转；
- 如果 pose head 不合格但 hidden token 可线性 probe 出运动，仍可继续；两者都不合格则停止重构。

## 6. 新 action encoder

### 6.1 Frame geometry token

CameraHead trunk 已经在 85 个 camera/register tokens 上完成跨帧 attention。每帧构造一个小型
geometry token：

```text
c_f = camera_hidden[f]                         # [2048]
r_f = register_hidden_mean[f]                  # [2048]
g_f = LN(Linear([c_f, r_f]))                   # [256]
```

这里不读取 patch/depth tokens，避免大模型表征重新混入大量外观细节，也将每个样本缓存压缩到
可接受大小。

### 6.2 Interval action token

五个 frame geometry tokens 变为四个相邻区间 action tokens：

```text
z_i = LN(MLP([g_i, g_{i+1}, g_{i+1}-g_i, time_embedding(dt_i)]))
z_i shape = [128], i = 0..3
```

输出为 `[B,4,128]`。固定的相邻差分路径必须保留，不使用 learned query pooling。`dt_i` 使用
manifest 的真实 frame time，不假设严格等于 1.0 s。

为限制小数据过拟合：

- projection/MLP 默认 dropout=0；
- adapter 参数量控制在 2M 以内；
- CameraHead trunk 第一阶段完全冻结；
- action token 跨 batch 的 std/cosine 持续写入 TensorBoard；
- 不添加 KL 或随机 posterior。

## 7. 新 trajectory decoder

使用 `SE2IncrementDecoder` 替换 Transformer direct head 和 unicycle head。

每个 interval token 结合该区间内 10 个真实相对时间 query，经共享 MLP 输出：

```text
[dx_body, dy_body, sin(dyaw), cos(dyaw)] × 10
```

归一化 sin/cos 后得到 `dyaw`，再按 SE(2) 递推：

```text
p_t   = p_{t-1} + R(yaw_{t-1}) @ [dx_body, dy_body]
yaw_t = wrap(yaw_{t-1} + dyaw)
```

该 decoder：

- 不读取图像、VGGT hidden、当前场景或 GT；
- 只依赖 `[B,4,128]` action tokens 和时间戳；
- 不预测 speed/acceleration，因此没有 zero-speed clamp 的死区；
- 原生输出恰好 4×10=40 点，并保留轨迹连续性；
- 后续 action expert 只需生成相同的四个 token 即可复用 decoder。

GT 的 body-frame SE(2) increments 从现有 40 点局部轨迹严格反算，作为额外监督。

## 8. Loss 设计

MVP 总损失：

```text
L = 1.0 * trajectory_reconstruction
  + 0.5 * body_increment_smooth_l1
  + 0.5 * one_second_keyframe_loss
  + 0.1 * yaw_periodic_loss
```

其中：

- trajectory reconstruction：40 点 XY robust distance；
- body increment：逐 0.1 s 的 `[dx_body,dy_body,dyaw]`；
- keyframe loss：第 10/20/30/40 点，对应四个 action token 的区间终点；
- yaw 使用 `1-cos(delta_yaw)`；
- 所有项使用真实 `future_times` 和 mask。

第一版不启用 physical/dynamics/KL/contrastive。理由是 trajectory loss 和 interval increment loss 已经
直接监督唯一的视觉 action encoder；不存在需要 contrastive 对齐的第二模态分支。待基础模型超过
mean baseline 后，再逐项加入速度/jerk 正则，不能一次叠加全部 loss。

## 9. 特征缓存

新增 `tools/features/cache_vggt_omega_features.py`，默认缓存 CameraHead trunk 后的：

```text
camera_hidden          [5,2048] fp16
register_hidden_mean   [5,2048] fp16
pose_enc               [5,9]    fp32
```

当前 train/val 共 23,387 个窗口。仅 camera + register mean 的理论 payload 约 0.96 GB；若缓存
CameraHead 输入的全部 17 special tokens，则约 8.1 GB。因此默认只缓存 trunk 后的压缩结果。

cache index 必须保存：

- manifest SHA256 和样本顺序；
- VGGT checkpoint SHA256；
- third-party git commit；
- preprocessing mode/resolution/final shape；
- token source、dtype 和 tensor shape；
- 分 shard checksum、start/end index；
- 支持原子写入和断点续跑。

训练发现任一 metadata 不一致时直接拒绝加载，不允许静默混用 PE cache、不同 VGGT 权重或不同
resize mode。

## 10. 训练阶段

### Phase A：backbone probe

- 严格加载 checkpoint；
- 跑几何审计；
- 记录单窗口峰值显存和耗时；
- 建立 train/val VGGT feature cache。

### Phase B：4 样本强制过拟合

选择静止、20 m 直行、最长直行和大转弯四类样本，只训练 action adapter + decoder。

验收：

- 500 step 内 visual ADE 明显下降；
- 最终 ADE ≤ 0.25 m；
- 四个 endpoint 分开且运动类型正确；
- action token off-diagonal cosine 不贴近 1；
- decoder gradient 有限，不持续触发 clip。

### Phase C：64 样本过拟合

验收：

- train ADE ≤ 0.75 m；
- 训练曲线持续下降且 resume 后连续；
- 静止/直行/转弯都不能退化为均值轨迹；
- TensorBoard 2×2 页面能同时显示五帧输入和三种轨迹诊断。

新架构没有 trajectory encoder，因此 2×2 页第四格改为：

```text
CAM_FRONT inputs | GT | VGGT-action reconstruction | error/increment diagnostics
```

### Phase D：完整训练

首先只训练缓存特征上的 adapter + decoder。模型选择使用 val visual ADE，并同时报告 FDE、yaw、
keyframe ADE 和 increment error。必须和以下 baseline 比较：

1. 始终静止；
2. train mean trajectory；
3. VGGTOmega pose + train-fitted global calibration；
4. 当前 PE V2 最佳 checkpoint（仅作为历史基线）。

go 条件：val ADE 至少比 train-mean baseline 降低 20%，且各速度/转弯 bucket 都有提升。

### Phase E：可选微调

只有 Phase D 已收敛但仍受表示上限约束时，才考虑解冻 CameraHead trunk 最后一层：

- LR 为 adapter LR 的 1/50 或更小；
- gradient checkpointing；
- online batch=1 + gradient accumulation；
- Aggregator 仍冻结；
- 单独新 run，不覆盖 frozen-backbone baseline。

由于 post-trunk cache 无法支持 trunk fine-tune，Phase E 必须切回 CameraHead 输入 cache 或 online
forward；不能假装在 post-trunk cache 上微调 backbone。

## 11. 配置草案

```yaml
vision_backbone:
  type: vggt_omega
  source_path: third_party/vggt-omega
  checkpoint_path: null  # REQUIRED: /abs/path/vggt_omega_1b_512.pt
  checkpoint_sha256: null
  image_resolution: 512
  resize_mode: max_size
  patch_size: 16
  freeze_aggregator: true
  freeze_camera_trunk: true
  cache_token_mode: camera_register_mean
  cache_dtype: float16

action_tokenizer:
  num_frames: 5
  num_action_tokens: 4
  action_token_dim: 128
  frame_geometry_dim: 256
  steps_per_token: 10
  decoder_type: se2_increment
  dropout: 0.0

loss:
  trajectory_weight: 1.0
  increment_weight: 0.5
  keyframe_weight: 0.5
  yaw_weight: 0.1
  dynamics_weight: 0.0
  physical_weight: 0.0

train:
  resume: auto
  batch_size: 32       # cached features; 根据显存实测调整
  learning_rate: 0.0003
  weight_decay: 0.01
  precision: bf16
  grad_clip_norm: 10.0
```

数据部分继续沿用当前可配置的 5 帧、4 s、40 点、LiDAR 10 Hz setting。

## 12. 文件级实施清单

| 文件 | 操作 |
|---|---|
| `src/.../data/dataset.py` | 增加 VGGT `[0,1]`、保 FOV 的 resize transform；与 PE transform 分离 |
| `src/.../models/vggt_omega.py` | 新增 checkpoint loader、CameraHead hidden wrapper、freeze contract |
| `src/.../models/tokenizer.py` | 用 4-token interval action encoder 替换现有双 encoder |
| `src/.../models/decoder.py` | 实现 body-frame SE(2) increment decoder/integrator |
| `src/.../models/factory.py` | 只构建 VGGT 新图，不再在主配置兼容 PE 架构 |
| `src/.../losses.py` | 改为 trajectory/increment/keyframe/yaw 四项主损失 |
| `src/.../data/dataset.py` | 增加严格 VGGT cache dataset 和 metadata 校验 |
| `tools/analysis/inspect_vggt_camera_motion.py` | 新增 camera motion 几何审计和可视化 |
| `tools/features/cache_vggt_omega_features.py` | 新增离线 feature cache |
| `tools/training/train_tokenizer.py` | 默认要求 VGGT cache；online 仅用于 smoke/fine-tune |
| `tools/evaluation/evaluate_tokenizer.py` | 增加 bucket、keyframe、increment baseline 指标 |
| `src/.../visualization.py` | 调整 2×2 第四格为 increment/error diagnostics |
| `configs/nuscenes_vggt_omega_front_4s.yaml` | 新建唯一正式配置 |
| `.vscode/launch.json` | 增加 probe/cache/train 三个 debug entry |
| `tests/` | 增加 shape、SE(2)、cache hash、freeze、checkpoint、overfit 测试 |

`src/...` 在实施时展开为 `src/vision_action_tokenizer/...`。

当新架构通过 Phase C 后，删除主路径中不再使用的 PE resampler、trajectory encoder、KL 和
contrastive 兼容分支；不为无法收敛的旧 checkpoint 继续增加兼容逻辑。旧 output 不删除，但新
run 必须使用 `output/itae_vggt_omega_v1`，严格禁止自动恢复 PE V2 权重。

## 13. 单元与集成测试

至少覆盖：

1. 官方预处理和项目 transform 输出逐像素/shape 一致；
2. CameraHead input `[B,5,17,2048]`、hidden 和 pose shape；
3. checkpoint 仅允许 dense/text head 差异，camera key 缺失必须失败；
4. 冻结 backbone 后无 gradient，adapter/decoder 有 gradient；
5. 4 interval tokens 与真实 frame `dt` 对齐；
6. GT pose -> body increment -> SE(2) integrate 能在数值容差内无损还原 40 点；
7. yaw 跨 `±pi` 不跳变；
8. mask 后的 loss/metric 不改变有效点结果；
9. cache manifest/checkpoint/preprocess hash 任一不一致时失败；
10. 训练 checkpoint 保存、自动恢复和 TensorBoard 2×2 写入；
11. 4 样本和 64 样本过拟合门禁。

## 14. 完成定义

只有同时满足以下条件，才认为 VGGT-Omega 重构完成：

- Gate 0 的许可证、checkpoint 和环境问题已关闭；
- camera motion 审计表明预训练几何表征对 nuScenes 自车运动有效；
- cache 可复现且 metadata 严格；
- 4/64 样本过拟合通过；
- 完整 val 明显超过 train-mean baseline；
- action tokens 不坍缩；
- decoder 完全 context-free；
- TensorBoard 能定位输入图像、区间 token、increment 和轨迹误差；
- 新 checkpoint 自动恢复通过，且旧 PE checkpoint 被明确拒绝。

## 15. 确认前需要用户补充/决定

1. 提供 `vggt_omega_1b_512.pt` 的绝对路径；
2. 确认 VGGT-Omega 许可证/AUP 对本项目用途可用；
3. 确认采用推荐的 `[4,128]` interval action tokens，而不是继续使用 `[10,D]` learned tokens；
4. 确认第一阶段强制使用离线 cache + frozen CameraHead trunk；
5. 确认新模型通过 64 样本门禁后，删除旧 PE 主架构而不是长期维护两套训练图。

## 16. 实施结果

用户提供 checkpoint 路径并要求按本计划实施后，已完成以下闭环：

- 在 `py312torch210cu126` 安装 VGGT-Omega 和本项目 dev/viz 依赖；
- checkpoint SHA256 为
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`，
  Aggregator + CameraHead strict load 通过；
- 官方 `max_size` 预处理逐像素一致；5 帧 BF16 forward 峰值约 6.02 GB；
- camera motion 四类 val 样本审计 direction cosine 为 0.978，使用 train 拟合的统一尺度
  35.40 后 keyframe ADE 为 3.31 m；
- 正式 tokenizer 输出 `[B,4,128]`，decoder 输出 `[B,40,3]`，trainable 参数
  1,657,476；
- 4 样本 500 step ADE 0.221 m、最终 0.117 m；随机 64 样本 750 step ADE
  0.449 m；
- cache 支持严格 metadata、分 shard SHA256、原子 index 和中断恢复；
- `torchrun` 的保存、自动恢复、TensorBoard scalars 与含五张输入图的 2×2 页面均已验证；
- 已删除旧 PE extractor/resampler、trajectory encoder、KL/contrastive loss 主实现；
- 回归检查为 Ruff clean、24 tests passed。

完整 train/val cache 与正式训练属于数据产物和长任务；命令、默认路径和自动恢复策略见
`README.md`。VGGT-Omega 的许可证/AUP 是否覆盖具体研究及发布方式，仍由项目负责人负责确认。
