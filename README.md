# MLPHC-GPU

本文仓库提供论文 **Tensorized GPU-Accelerated Hyper-Construction for Large-Scale Sensor--Interceptor--Target Assignment With Time Windows** 中 MLPHC-GPU 的可复现实现。代码以 PyTorch 张量化执行候选四元组评分、约束状态更新、可行性掩码、HCLPSO 粒子更新和全局最优归约。

## 论文默认配置

`scripts/run_paper_configuration.py` 默认采用论文田口实验确定的配置：

| 项目 | 默认值 |
|---|---:|
| MLP 架构 | 3–8–1 |
| 隐藏单元数 | 8 |
| MLP 参数维数 | 40 |
| 参数取值范围 | `[-2, 2]^40` |
| HCLPSO 种群规模 | 20 |
| 探索/开发子种群 | 8 / 12 |
| 最大函数评估次数 | `50 × TargetNum` |
| 学习样本刷新 | 连续 6 代未改进后刷新 |
| 数值精度 | `float64` |

这些默认值集中定义在 `mlphc_gpu/paper_config.py`，命令行参数可用于消融实验，但复现论文结果时无需修改。

## 环境

论文实验环境为 Windows 11、NVIDIA RTX 4090、CUDA 11.8、Python 3.9、PyTorch 2.0 和 NumPy 1.23。其他支持 CUDA 的 PyTorch 2.x 环境也可运行。

建议先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与本机 CUDA 匹配的 PyTorch，再安装本项目：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 快速验证

先生成一个小型演示算例，再使用论文默认配置运行：

```bash
python scripts/generate_demo_instance.py
python scripts/run_paper_configuration.py examples/demo_instance.mat
```

结果默认写入 `results/demo_instance_paper_configuration.json`，包括最优目标值、40 维 MLP 参数、构造方案、收敛记录、评估次数和运行时间。

## 运行论文的 36 个算例

```bash
python scripts/run_paper_configuration.py benchmarks/paper_36/case_01.mat
```

仓库中的 `benchmarks/paper_36/case_01.mat`--`case_36.mat` 与论文 Case 1--Case 36 严格对应。算例分组、规模、候选四元组数、参考解类型和 SHA-256 见 [`benchmarks/paper_36/manifest.csv`](benchmarks/paper_36/manifest.csv)。

一次运行全部 36 例：

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 1
```

按照论文协议，每例独立运行 25 次：

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 25
```

只运行部分算例时，可使用单个编号、逗号列表或区间：

```bash
python scripts/run_paper_suite.py --cases 20,22,24 --runs 25
```

默认使用 `cuda`，每次运行采用独立随机种子，结果写入 `results/paper_36/case_XX/`。每个 JSON 文件包含最优目标值、40 维 MLP 参数、所选四元组、收敛记录、实际评估次数和优化时间。仅用于功能检查时，可运行：

```bash
python scripts/run_paper_configuration.py examples/demo_instance.mat --device cpu --disable-cuda-graph
```

输入支持论文实验所用的 MATLAB `.mat` 算例格式。主要字段包括 `radar_num`、`weapon_pr_matrix`、`weapon_vp_matrix`、`danyao_constraint`、`strike_constraint`、`radar_constraint`、`V_matrix`、`radar_tbegin_cell`、`radar_tend_cell` 和 `radar_pr_cell`。详细说明见 [`benchmarks/paper_36/README.md`](benchmarks/paper_36/README.md)。

## 代码结构

- `mlphc_gpu/hclpso_gpu.py`：张量化 HCLPSO。
- `mlphc_gpu/optimized_dynamic_mlp_gpu.py`：3–8–1 MLP 评分器。
- `mlphc_gpu/optimized_dynamic_mmrra_rbf_gpu.py`：静态形状的 GPU 构造与 CUDA Graph 执行。
- `mlphc_gpu/dynamic_mmrra_rbf_gpu.py`：候选四元组、时间窗和约束张量。
- `mlphc_gpu/instance_io.py`：MATLAB 算例读取。
- `benchmarks/paper_36/`：论文最终实验使用的 36 个算例及清单。
- `scripts/run_paper_suite.py`：按论文编号批量运行一个或多个算例。
- `taguchi/`：田口 `L9(3^3)` 参数设置与汇总结果。
- `tests/`：论文默认配置检查。

## 测试

```bash
python -m unittest discover -s tests
```

## 方法说明

每个 HCLPSO 粒子编码一个 3--8--1 MLP 构造规则的 40 个参数。对同一代中的粒子，程序在 GPU 上批量计算候选四元组的三个特征及其 MLP 分数，并为每个粒子选择当前得分最高的可行四元组。每次选择后，索引张量同步更新目标剩余拦截次数、拦截器弹药和传感器各时刻的剩余通道容量；布尔可行性掩码随即停用违反更新后约束的四元组。上述“评分--选择--状态更新--掩码更新”过程持续到没有可行四元组为止，得到一个完整可行解及其目标值。

HCLPSO 以该目标值作为粒子适应度。探索与开发子种群通过行索引划分并在设备端并行更新，边界检查、个体最优更新、停滞检测和全局最优归约也均采用张量运算。不同粒子的构造过程并行执行，但单个构造过程中连续选择四元组的步骤保持顺序依赖。

## 引用

如在研究中使用本代码，请引用对应论文。论文正式出版后，可在本节补充卷期、页码和 DOI。
