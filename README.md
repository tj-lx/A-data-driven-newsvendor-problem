# From Data to Decision: Newsvendor Replication（课程复现项目）

本项目复现 Huber 等人在 EJOR (2019) 的数据驱动报童问题方法，围绕三层框架展开：Level 1 需求预测、Level 2 库存优化（参数法 vs 非参数 SAA）、Level 3 集成优化（分位数回归）。数据来源为 Kaggle 的 French Bakery 销售数据(https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales/data)。

## 项目概述
- 目标：在未知需求分布下，利用历史数据与特征进行订货量决策，最小化报童成本。
- 方法：对比 `Normal` 参数法与 `SAA` 非参数法，并实现 `Integrated`（分位损失直接拟合订货量）。
- 结果素材与报告均可在本仓库内生成与查看。

## 目录结构
- `data/` 原始与清洗后的数据
  - `Bakery sales.csv` 原始交易数据（放置于此）
  - `prepared/bakery_daily.csv` 交易聚合后的日粒度数据（脚本生成）
  - `bakery_daily.csv` 清洗后的日粒度数据（脚本生成，主脚本读取）
- `scripts/` 数据处理脚本
  - `prepare_bakery.py` 原始交易到日粒度聚合
  - `clean_bakery_daily.py` 清洗与过滤（零销量比、非零天数、Top N）
- `model_and_optimization.py` 复现主脚本（训练、优化、可视化、摘要）
- `output/` 运行生成的图表与指标
  - `inventory_result.png` Level 2 决策可视化
  - `level23_cost_curve.png` Normal/SAA/Integrated 成本对比曲线
  - `level3_cost_curve.png` Level 3（调参）成本曲线
  - `metrics_summary.csv` 关键指标汇总
  - `summary.txt` 文本摘要（含数据来源与方法说明）
- `latex/` LaTeX 报告
  - `present.tex` 报告源码（已用 `[H]` 固定图片位置）
  - `present.pdf` 编译生成的报告

## 环境准备
- Python ≥ 3.9
- 依赖安装：
  ```bash
  pip install pandas numpy scikit-learn matplotlib holidays scipy
  ```
- LaTeX（编译报告）：建议 TeX Live + XeLaTeX，已使用 `ctex`。

## 快速开始
1. 放置原始数据到 `data/Bakery sales.csv`
2. 生成日粒度数据（可选但推荐）：
   ```bash
   # 从原始交易聚合到日粒度
   python scripts/prepare_bakery.py "data/Bakery sales.csv" --out data/prepared/bakery_daily.csv
   # 清洗与过滤，得到主脚本读取的数据
   python scripts/clean_bakery_daily.py data/prepared/bakery_daily.csv --out data/bakery_daily.csv --top_n_articles 10
   ```
3. 运行复现主脚本并生成图表与摘要：
   ```bash
   python model_and_optimization.py --top_n_products 10 --service_levels 0.5,0.7,0.8,0.9,0.95 --tune_level3
   ```
   生成文件位于 `output/`：
   - `inventory_result.png`, `level23_cost_curve.png`, `level3_cost_curve.png`
   - `metrics_summary.csv`, `summary.txt`

## 报告编译
```bash
cd latex
xelatex -interaction=nonstopmode -halt-on-error present.tex
```
编译后得到 `latex/present.pdf`，图片位置已使用 `[H]` 固定（参见 `latex/present.tex:137` 与 `latex/present.tex:145`）。

## 关键代码位置
- 数据读取：`model_and_optimization.py:16` 读取 `data/bakery_daily.csv`
- 成本计算与比较：`model_and_optimization.py:134` 定义报童成本并在 140–151 处计算/输出
- Level 3 分位损失模型与说明：`model_and_optimization.py:311` 起，摘要中记录方法与调参
- 报告图片固定浮动：`latex/present.tex:137`、`latex/present.tex:145` 使用 `\begin{figure}[H]`

## 致谢
数据：Kaggle「French Bakery Daily Sales」  
方法：Huber et al., EJOR (2019) 数据驱动报童问题

