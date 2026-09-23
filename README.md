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

## 运行基准算例

```bash
python scripts/run_paper_configuration.py D:\path\to\instance.mat --output results\case01.json
```

默认使用 `cuda`。仅用于功能检查时，可运行：

```bash
python scripts/run_paper_configuration.py examples/demo_instance.mat --device cpu --disable-cuda-graph
```

输入支持论文实验所用的 MATLAB v7 与 v7.3 算例格式。主要字段包括 `radar_num`、`weapon_pr_matrix`、`weapon_vp_matrix`、`danyao_constraint`、`strike_constraint`、`radar_constraint`、`V_matrix`、`radar_tbegin_cell`、`radar_tend_cell` 和 `radar_pr_cell`。基准算例文件体积较大，且其再分发权限与代码不同，因此未直接放入仓库。

## 代码结构

- `mlphc_gpu/hclpso_gpu.py`：张量化 HCLPSO。
- `mlphc_gpu/optimized_dynamic_mlp_gpu.py`：3–8–1 MLP 评分器。
- `mlphc_gpu/optimized_dynamic_mmrra_rbf_gpu.py`：静态形状的 GPU 构造与 CUDA Graph 执行。
- `mlphc_gpu/dynamic_mmrra_rbf_gpu.py`：候选四元组、时间窗和约束张量。
- `mlphc_gpu/instance_io.py`：MATLAB 算例读取。
- `taguchi/`：田口 `L9(3^3)` 参数设置与汇总结果。
- `tests/`：论文默认配置检查。

## 测试

```bash
python -m unittest discover -s tests
```

## 方法说明

每个粒子编码一个 MLP 构造规则。对同一代粒子，代码批量计算所有可行四元组的 MLP 分数；选中四元组后，以索引张量更新目标、拦截器和传感器通道状态，并用布尔掩码停用违反约束的四元组。HCLPSO 的探索与开发子种群同样通过行索引并行更新，边界检查、个体最优更新、停滞检测和全局最优选择均在设备端完成。

## 引用

如在研究中使用本代码，请引用对应论文。论文正式出版后，可在本节补充卷期、页码和 DOI。
