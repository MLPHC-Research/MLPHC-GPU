# 论文中的 36 个 SITA-TW 算例

本目录包含论文 **Tensorized GPU-Accelerated Hyper-Construction for Large-Scale Sensor--Interceptor--Target Assignment With Time Windows** 最终实验使用的 36 个算例。文件名 `case_01.mat`--`case_36.mat` 与论文中的 Case 1--Case 36 严格对应。

## 算例分组

| 论文编号 | 来源与设置 |
|---|---|
| Case 1--12 | 从 MOP5 benchmark 的 12 个算例导出 SITA-TW 时间窗，`m_j=1` |
| Case 13--24 | 与 Case 1--12 配对，`m_j=2` |
| Case 25--30 | 使用 MOP5 instance generator 生成的 6 个高冲突算例，`m_j=1` |
| Case 31--36 | 与 Case 25--30 配对，`m_j=2` |

其中，`m_j` 表示目标 `j` 最多可分配的拦截次数；`CS` 和 `DS` 分别表示 centralized 与 decentralized layouts。高冲突算例通过收紧传感器通道容量并延长制导区间，提高资源竞争与时间冲突强度。

`manifest.csv` 给出每个文件的论文编号、原始全局编号、来源、布局、规模、候选四元组数、参考解类型和 SHA-256。`Reference=gurobi_optimal` 表示 Gurobi 已证明最优；`Reference=best_known` 表示论文采用已知最好值作为参考。

## 数据格式

每个文件均为 MATLAB `.mat` 格式，主要字段包括：

- `weapon_pr_matrix`：拦截成功概率；
- `weapon_vp_matrix`：拦截器可用时间/窗口信息；
- `radar_pr_cell`：传感器探测概率；
- `radar_tbegin_cell`、`radar_tend_cell`：制导开始与结束时刻；
- `danyao_constraint`：拦截器弹药容量；
- `strike_constraint`：每个目标最多可分配的拦截次数 `m_j`；
- `radar_constraint`：传感器并发制导通道容量；
- `V_matrix`：目标价值。

## 运行方法

在仓库根目录运行单个算例：

```bash
python scripts/run_paper_configuration.py benchmarks/paper_36/case_01.mat
```

运行全部 36 个算例，每个算例独立运行一次：

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 1
```

按论文协议执行每个算例 25 次独立运行：

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 25
```

也可以只运行指定子集，例如 Case 20、22 和 24：

```bash
python scripts/run_paper_suite.py --cases 20,22,24 --runs 25
```

默认使用 GPU，并采用论文田口实验确定的 3--8--1 MLP、20 个粒子和 `50 × TargetNum` 函数评估预算。结果按算例和独立运行编号写入 `results/paper_36/`。仅做功能检查时，可添加 `--device cpu --disable-cuda-graph`。
