# SUV NAVSIM v1 测评指南

本文说明如何从 ITAE 项目根目录调用 `experiments/navsimv1`，使用发布的
`suv_navsim.pt` 在 NAVSIM v1 `navtest` 上复现官方 PDM Score（PDMS），并自动生成与论文
结果的差值表。

## 1. 复现口径

SUV 仓库给出的 NAVSIM v1 结果如下，表内均为百分数：

| Metric | NC | DAC | TTC | Comfort | EP | PDMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SUV published | 99.1 | 97.8 | 96.7 | 100.0 | 84.6 | 90.8 |

对应口径为：单个前视相机、4 帧历史、8 帧未来、单条候选轨迹、10 个 diffusion solver
step。项目入口默认固定这些设置，并使用 NAVSIM v1 官方的 `PDMResults` 与 `pdm_score`。

当前资产已经确认：

- SUV commit：`486a97a81e277f9a51c24d2f5825602052360d2b`；
- NAVSIM devkit：tag `v1.1`，commit
  `0811876c274e8b058ab2be9b3dcd4d37bd23f177`；
- `suv_navsim.pt` SHA256：
  `e9f036eff6f2afd4cc4c154cbbe38f50998e700d4d0a5adc43fe28155d55102a`；
- 数据中的地图元数据为 `nuplan-maps-v1.0`；
- 当前 `navtest` 共 136 个有效日志文件、12,146 个 token。

仓库的 `third_party/navsim` 是 NAVSIM v2，评分函数已经增加
`traffic_agents_policy`，不能直接用于 NAVSIM v1 论文复现。下面使用独立环境及 v1.1
worktree，不会改动现有 `py312torch210cu126` 环境或 NAVSIM v2 子模块。

## 2. 一次性安装

在 ITAE 根目录执行：

```bash
bash scripts/suv/setup_navsim_v1_env.sh
```

脚本会：

1. 从 `py312torch210cu126` 克隆 `suv-navsim1`；
2. 在 `output/suv/navsim_v1_devkit` 创建 NAVSIM v1.1 的 detached worktree；
3. editable 安装 NAVSIM v1.1 和 `third_party/SUV`；
4. 安装 SUV 推理及 NAVSIM 评分所需依赖；
5. 运行入口自检。

NAVSIM v1.1 必须 editable 安装。它发布的 wheel 不包含 Hydra YAML，虽然 Python import
可以成功，运行 metric caching 时仍会报 `Cannot find primary config`。安装脚本已处理该问题。

若要使用不同的源/目标环境，可覆盖变量：

```bash
SOURCE_ENV=py312torch210cu126 TARGET_ENV=suv-navsim1 \
  bash scripts/suv/setup_navsim_v1_env.sh
```

本配置遵循本项目指定的 Python 3.12、PyTorch 2.10.0+cu126 环境。SUV 上游 README 使用
Python 3.10、PyTorch 2.7.1+cu128；若需要逐版本复刻上游软件栈，应另建其原始环境。比较论文
数值时，应把这项软件版本差异记录在实验元数据中。

## 3. 路径和自检

入口已经内置当前机器的默认路径：

```text
checkpoint: /inspire/hdd/project/spatiotemporal-intelligence-research/ky26298/Projects/pure_checkpoints/SUV_ckpt
NAVSIM:     /inspire/hdd/global_public/public_datas/NAVSIM
run root:   <ITAE>/output/suv/navsim_v1
```

运行严格自检：

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 doctor
```

当准备开始评分时，再要求 metric cache 非空：

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 doctor --require-metric-cache
```

`ok` 必须为 `true`。报告还会显示实际导入的 NAVSIM 路径、API 版本特征、地图版本、权重
文件、日志数量及 metric cache 条目数。若换机器，可给所有子命令传
`--checkpoint-root`、`--data-root`、`--map-root` 或 `--run-root`。

## 4. 先做单场景 smoke test

smoke test 使用独立目录，不会污染全量结果：

```bash
SMOKE_ROOT=output/suv/navsim_v1_smoke

# 1) CPU 生成一个场景的官方 metric cache
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 cache-metrics \
  --run-root "${SMOKE_ROOT}" --max-scenes 1 --workers 2

# 2) GPU 生成该场景所需的动态 prompt embedding
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 precompute \
  --run-root "${SMOKE_ROOT}" --gpus 0 --max-scenes 1 --batch-size 1

# 3) 加载 SUV 权重并完成一次官方 PDM scoring
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 evaluate \
  --run-root "${SMOKE_ROOT}" --gpus 0 --max-scenes 1

# 4) 检查结果文件和汇总器
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 summarize \
  --run-root "${SMOKE_ROOT}"
```

`--max-scenes` 只用于联调；一个或少量场景的结果不能与论文表格比较。当前代码已经实际通过
单场景 metric cache 生成和真实数据 prompt 扫描。GPU smoke 前应通过 `nvidia-smi` 确认目标卡
没有其他大模型任务。

## 5. 双卡全量评估

### 5.1 生成 NAVSIM v1 metric cache

当前公共数据目录没有可直接使用的 v1 metric cache，因此需要先生成一次：

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 cache-metrics \
  --workers 32
```

这是 CPU/NuPlan 地图预处理任务，`--workers` 按机器 CPU 和内存调整。缓存写到
`output/suv/navsim_v1/metric_cache`，重复运行会复用已有场景。全量结束后执行：

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 doctor --require-metric-cache
```

在当前数据上，`metric_cache_entries` 应为 12,146；少于该值时，评分入口只会评估缓存与
navtest 的交集，并打印缺失 token 警告。

### 5.2 双卡生成 text embedding cache

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 precompute \
  --gpus 0,1 --batch-size 16
```

每个 rank 各自加载一份 Wan UMT5 encoder，唯一 prompt 按 rank 分片；输出采用原子写入。
已有文件默认复用，需要重算时添加 `--overwrite`。若显存不足，只降低 `--batch-size`，例如
`--batch-size 4`。

### 5.3 双卡 PDM scoring

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 evaluate \
  --gpus 0,1
```

这里使用的是项目根目录的
`experiments/navsimv1/run_pdm_score_multigpu.py` 和
`experiments/navsimv1/pdm_agent.py`，不是从 `third_party/SUV` 启动 shell 脚本。父进程把
token 均匀分给两张卡，每张卡各加载一份 SUV 模型并独立推理，最后合并 rank CSV；没有
DDP 梯度同步。48GB 显卡适合这一运行方式，但评分期间目标卡应保持空闲。

可通过以下命令先检查最终启动命令而不执行：

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 evaluate --gpus 0,1 --dry-run
```

默认 `--inference-steps 10` 与发布口径一致。改变该值后的结果不应直接写入论文复现表。

## 6. 生成论文差值表

评分成功后执行：

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 summarize
```

入口自动选取 evaluation 目录中最新的 CSV，并生成：

```text
output/suv/navsim_v1/
├── metric_cache/
├── text_embeddings/
└── evaluation/
    ├── YYYY.MM.DD.HH.MM.SS.csv
    ├── summary.json
    └── comparison.md
```

`comparison.md` 同时显示 reproduced、published 和差值 `delta (pp)`。差值单位是百分点，
例如复现 89.8、发布 90.8 时，PDMS delta 为 `-1.0 pp`。汇总器也会报告 total、valid 和
failed 样本数；正式结果要求 `failed=0` 且样本数为 12,146。

也可以汇总指定 CSV：

```bash
conda run -n suv-navsim1 \
  python -m tools.suv.evaluate_navsim_v1 summarize \
  --results /path/to/result.csv
```

## 7. 常见问题

- `PDMResults is missing` 或出现 `traffic_agents_policy`：导入了 NAVSIM v2。重新运行安装脚本，
  再看 `doctor.module_paths.navsim` 是否指向 `output/suv/navsim_v1_devkit`。
- `Cannot find primary config`：NAVSIM v1.1 被安装成了 wheel，而非 editable worktree。
- `No module named pytorch_lightning` 或 `pkg_resources`：环境来自旧版安装脚本。重新运行
  `bash scripts/suv/setup_navsim_v1_env.sh`，脚本会安装 Lightning 并固定兼容的 setuptools。
- `metric cache missing or empty`：先执行 `cache-metrics`；NAVSIM v2 cache 不能混用。
- `Missing text embedding cache`：metric cache 和 text cache 是两类不同缓存，需执行
  `precompute`。
- CUDA OOM：先确认 GPU 没有其他进程；text encoder 阶段可降低 batch size。评分阶段每张卡
  都会加载完整 SUV，增加 GPU 数量只做数据分片，不降低单卡模型显存。
- 结果样本数不足：查看 runner 的 missing metric cache 警告，并核对 doctor 的
  `metric_cache_entries`。

查看完整 CLI：

```bash
conda run -n suv-navsim1 python -m tools.suv.evaluate_navsim_v1 --help
```
