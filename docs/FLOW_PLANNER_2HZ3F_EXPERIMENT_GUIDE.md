# PE Flow Planner：2Hz 三帧 raw vs token 实验

本实验只改变 planner 的视觉条件窗口，公平比较 raw trajectory 与 V4 trajectory token：

- 条件帧：官方 `CAM_FRONT` keyframe，时刻 `[-1.0,-0.5,0.0]s`；
- 预测目标：锚点之后 `(0,4.0]s`，从 LiDAR sweep ego pose 最近邻采样 40 点；
- 视觉编码：冻结 PE-Spatial-B16-512，每帧 64 token，拼接为 `[192,768]`；
- ego motion：三帧各一个 `[x,y,yaw,vx,vy,yaw_rate]` LiDAR state token；
- 生成器：相同 8-layer flow planner，flow matching 预测 velocity，推理固定 5-step/NFE=5；
- 输出 A：raw `[40,3]`；输出 B：V4 token `[4,192]`，经冻结 decoder 解码。

“2Hz 3 帧”的名义窗口长度是 `3/2=1.5s`，但第一帧到当前帧的 timestamp span 是 `1.0s`。
代码与报告一律以显式 offsets 为定义。每个 PE patch token 还带所属图像的实际相对时间。

所有命令从项目根目录运行：

```bash
conda activate py312torch210cu126
pip install -e '.[pe,dev,viz]'
```

## 1. 生成 manifest 与 scene-disjoint split

```bash
mkdir -p data/manifests/nuscenes_2hz3f_front_4s

python tools/data/build_nuscenes_manifest.py \
  --config configs/planner/nuscenes_flow_raw_pe_2hz3f_4s.yaml \
  --info data/nuscenes/nuscenes_interp_12Hz_infos_train.pkl \
  --output data/manifests/nuscenes_2hz3f_front_4s/train.jsonl \
  --report data/manifests/nuscenes_2hz3f_front_4s/train_report.json

python tools/data/build_nuscenes_manifest.py \
  --config configs/planner/nuscenes_flow_raw_pe_2hz3f_4s.yaml \
  --info data/nuscenes/nuscenes_interp_12Hz_infos_val.pkl \
  --output data/manifests/nuscenes_2hz3f_front_4s/val.jsonl \
  --report data/manifests/nuscenes_2hz3f_front_4s/val_report.json
```

本机开启 ego-motion 后验证结果为 train `17,548/700 scenes`、official val `3,652/150 scenes`。
每个 scene 最早的一部分 anchor 没有覆盖第一张 camera timestamp 的历史 LiDAR pose，会被严格
丢弃而不是外推。info 虽包含
12Hz 插值 camera records，`image_source: keyframe` 会明确排除它们。实际 LiDAR sweep 间隔中位数
约 49.8ms；目标仍按 10Hz 请求，并保留最近 sweep 的真实时间。

为复用固定 V4 teacher 的原五帧 rich cache，raw/token 使用 teacher sample universe 的交集：

```bash
python tools/data/split_planner_manifests.py \
  --train-manifest data/manifests/nuscenes_2hz3f_front_4s/train.jsonl \
  --final-manifest data/manifests/nuscenes_2hz3f_front_4s/val.jsonl \
  --sample-universe-train-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
  --sample-universe-final-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --output data/manifests/nuscenes_planner_2hz3f_front_4s \
  --validation-fraction 0.1 \
  --seed 42
```

固定 split 为 train `15,815/630 scenes`、validation `1,713/70 scenes`、final eval
`3,646/150 scenes`。过滤的 20 个 train 和 6 个 final 样本在原 teacher 中没有完整窗口。

## 2. 缓存三帧 PE 条件

```bash
for split in train validation final_eval; do
  python tools/features/cache_planner_vision_features.py \
    --config configs/planner/nuscenes_flow_raw_pe_2hz3f_4s.yaml \
    --manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_${split}.jsonl \
    --output /home/alan/AlanLiang/Dataset/itae_planner_cache/pe_spatial_b16_512_2hz3f_ego_motion/${split} \
    --batch-size 8 --shard-size 256 --num-workers 4
done
```

cache v3 保存 PE token/mask/time，以及与三帧对齐的 LiDAR ego-motion state/time。状态为
`[x_anchor,y_anchor,yaw_anchor,vx_body,vy_body,yaw_rate]`，全部来自 `t<=0` 的 pose；不要复用
旧单帧或无 ego-motion 的 cache。工具支持校验后断点续跑。

## 3. 导出 V4 teacher target

```bash
mkdir -p /home/alan/AlanLiang/Dataset/itae_planner_cache/v4_action_targets_2hz3f_ego_motion

for split in train validation; do
  python tools/features/cache_tokenizer_action_targets.py \
    --tokenizer-config configs/nuscenes_vggt_omega_front_4s_v4_output_residual.yaml \
    --source-manifest data/manifests/nuscenes_lidar10hz_front_4s_train.jsonl \
    --manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_${split}.jsonl \
    --feature-cache /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_train_rich \
    --checkpoint output/itae_vggt_omega_v4_output_residual/best.pt \
    --output /home/alan/AlanLiang/Dataset/itae_planner_cache/v4_action_targets_2hz3f_ego_motion/${split}.safetensors \
    --batch-size 64
done

python tools/features/cache_tokenizer_action_targets.py \
  --tokenizer-config configs/nuscenes_vggt_omega_front_4s_v4_output_residual.yaml \
  --source-manifest data/manifests/nuscenes_lidar10hz_front_4s_val.jsonl \
  --manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_final_eval.jsonl \
  --feature-cache /home/alan/AlanLiang/Dataset/vggt_omega_cache/nuscenes_front_4s_val_rich \
  --checkpoint output/itae_vggt_omega_v4_output_residual/best.pt \
  --output /home/alan/AlanLiang/Dataset/itae_planner_cache/v4_action_targets_2hz3f_ego_motion/final_eval.safetensors \
  --batch-size 64
```

teacher target 仅作为固定标签；两种 planner 的输入都只有三张历史/当前图像，不读取未来图像。

## 3.1 Flow target 的 normalization

当前实现不是 min-max 到 `[-1,1]`。raw `[40,3]` 和 token `[4,192]` 分别只用 planner train
split 拟合逐 slot、逐维 mean/std：

```text
normalized_target = (target - train_mean) / max(train_std, 1e-4)
```

flow 在标准化空间中使用 Gaussian source，预测直线路径 velocity；采样完成后严格反标准化，
再计算米制 ADE/FDE。标准化值没有硬边界，超过 `[-1,1]` 是正常行为。

## 4. 正式训练（三个 seed）

```bash
for seed in 42 43 44; do
  torchrun --standalone --nproc_per_node=1 tools/training/train_flow_planner.py \
    --config configs/planner/nuscenes_flow_raw_pe_2hz3f_4s.yaml \
    --train-manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_train.jsonl \
    --val-manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_validation.jsonl \
    --output output/planner_raw_pe_2hz3f_ego_seed${seed} --seed ${seed}

  torchrun --standalone --nproc_per_node=1 tools/training/train_flow_planner.py \
    --config configs/planner/nuscenes_flow_token_v4_pe_2hz3f_4s.yaml \
    --train-manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_train.jsonl \
    --val-manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_validation.jsonl \
    --output output/planner_token_v4_pe_2hz3f_ego_seed${seed} --seed ${seed}
done
```

训练会自动恢复各自目录的 `last.pt`。TensorBoard 2×2 diagnostic 左上同时显示三张条件帧及实际
相对时间。不要比较 raw/token flow loss；比较 decoded validation ADE/FDE、相同 seen samples 下的
ADE AUC 和 final paired metrics。

```bash
tensorboard --logdir_spec \
raw:output/planner_raw_pe_2hz3f_ego_seed42/tensorboard,\
token:output/planner_token_v4_pe_2hz3f_ego_seed42/tensorboard --port 6006
```

## 5. 5-NFE final eval、paired comparison 与汇总

```bash
mkdir -p output/planner_eval_2hz3f_ego
for seed in 42 43 44; do
  python tools/evaluation/evaluate_flow_planner.py \
    --config configs/planner/nuscenes_flow_raw_pe_2hz3f_4s.yaml \
    --manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_final_eval.jsonl \
    --checkpoint output/planner_raw_pe_2hz3f_ego_seed${seed}/best.pt \
    --output output/planner_eval_2hz3f_ego/raw_seed${seed}

  python tools/evaluation/evaluate_flow_planner.py \
    --config configs/planner/nuscenes_flow_token_v4_pe_2hz3f_4s.yaml \
    --manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_final_eval.jsonl \
    --checkpoint output/planner_token_v4_pe_2hz3f_ego_seed${seed}/best.pt \
    --output output/planner_eval_2hz3f_ego/token_seed${seed}

  python tools/evaluation/compare_flow_planners.py \
    --raw-eval output/planner_eval_2hz3f_ego/raw_seed${seed} \
    --token-eval output/planner_eval_2hz3f_ego/token_seed${seed} \
    --raw-history output/planner_raw_pe_2hz3f_ego_seed${seed}/training_history.jsonl \
    --token-history output/planner_token_v4_pe_2hz3f_ego_seed${seed}/training_history.jsonl \
    --output output/planner_eval_2hz3f_ego/comparison_seed${seed}.json
done

python tools/evaluation/summarize_flow_planner_seeds.py \
  --comparisons output/planner_eval_2hz3f_ego/comparison_seed{42,43,44}.json \
  --output output/planner_eval_2hz3f_ego/three_seed_summary.json
```

先回答三帧 setting 内 token 是否比 raw 收敛更快/最终更好；再与旧单帧
`output/planner_eval/three_seed_summary.json` 比较 history condition 是否有效。单帧与三帧的
condition-position 参数量不同，因此这是 setting ablation；三帧 raw vs token 才是严格表示对照。
