import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm, shapiro
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 1. 准备工作：划分训练集和测试集
# ==========================================
print("正在读取数据...")
df = pd.read_csv('data/bakery_daily.csv')
df['date'] = pd.to_datetime(df['date'])
parser = argparse.ArgumentParser()
parser.add_argument('--top_n_products', type=int, default=10)
parser.add_argument('--service_levels', type=str, default='0.5,0.7,0.8,0.9,0.95')
parser.add_argument('--tune_level3', action='store_true')
parser.add_argument('--gb_grid_n_estimators', type=str, default='100,200,400')
parser.add_argument('--gb_grid_max_depth', type=str, default='2,3,4')
parser.add_argument('--gb_grid_learning_rate', type=str, default='0.03,0.05,0.1')
parser.add_argument('--outdir', type=str, default='output')
args = parser.parse_args()
TOP_N = args.top_n_products
service_levels = [float(x) for x in args.service_levels.split(',') if x.strip()]
grid_n_estimators = [int(x) for x in args.gb_grid_n_estimators.split(',') if x.strip()]
grid_max_depth = [int(x) for x in args.gb_grid_max_depth.split(',') if x.strip()]
grid_learning_rate = [float(x) for x in args.gb_grid_learning_rate.split(',') if x.strip()]
gb_grid = {'n_estimators': grid_n_estimators, 'max_depth': grid_max_depth, 'learning_rate': grid_learning_rate}
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
top_articles = (
    df.groupby('article')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(TOP_N)
    .index
)
df = df[df['article'].isin(top_articles)].copy()
print(f"Top {TOP_N} 产品: {', '.join(list(top_articles))}")
try:
    df['is_public_holiday'] = df['is_public_holiday'].astype(int)
except Exception:
    df['is_public_holiday'] = (
        df['is_public_holiday']
        .astype(str)
        .str.lower()
        .map({'true': 1, 'false': 0, '1': 1, '0': 0})
        .fillna(0)
        .astype(int)
    )

# 模拟论文：按时间切分。最后 20% 的时间作为测试集 (Test Set)，用来评估模型
split_date = df['date'].quantile(0.8)
train = df[df['date'] < split_date].copy()
test = df[df['date'] >= split_date].copy()

# 定义特征 (X) 和 目标 (y)
features = ['weekday', 'month', 'is_public_holiday', 'lag_1', 'lag_7']
target = 'sales'

print(f"训练集大小: {len(train)}, 测试集大小: {len(test)}")
print(f"使用的特征: {features}")

# ==========================================
# 2. Level 1: 需求预测 (Demand Estimation)
#    复现论文：对比 传统统计模型(用线性回归模拟) vs 机器学习(随机森林)
# ==========================================

# A. 线性回归 (模拟论文中的 Linear / Traditional benchmarks)
lr_model = LinearRegression()
lr_model.fit(train[features], train[target])
test['pred_lr'] = lr_model.predict(test[features])

# B. 随机森林 (模拟论文中的 ANN/Trees - ML方法)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train[features], train[target])
test['pred_rf'] = rf_model.predict(test[features])

# --- 评估预测精度 (Table 3 复现) ---
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{model_name}] RMSE: {rmse:.2f}, MAE: {mae:.2f}")

print("\n--- Level 1: 预测精度评估 ---")
evaluate(test[target], test['pred_lr'], "Linear Regression")
evaluate(test[target], test['pred_rf'], "Random Forest (ML)")
r2_lr = r2_score(test[target], test['pred_lr'])
r2_rf = r2_score(test[target], test['pred_rf'])
print(f"R^2 (测试集) - Linear Regression: {r2_lr*100:.2f}%")
print(f"R^2 (测试集) - Random Forest: {r2_rf*100:.2f}%")
rmse_lr = np.sqrt(mean_squared_error(test[target], test['pred_lr']))
mae_lr = mean_absolute_error(test[target], test['pred_lr'])
rmse_rf = np.sqrt(mean_squared_error(test[target], test['pred_rf']))
mae_rf = mean_absolute_error(test[target], test['pred_rf'])


# ==========================================
# 3. Level 2: 库存优化 (Inventory Optimization)
#    复现论文：对比 Parametric (正态分布) vs Non-parametric (SAA)
# ==========================================

# 设定商业参数：假设这是一个高利润产品
# 比如：成本 0.5欧，售价 1.5欧 -> 缺货损失(Cu)=1.0, 超储损失(Co)=0.5
Cu = 1.0 
Co = 0.5
target_service_level = Cu / (Cu + Co)  # 临界分位数 (Critical Ratio) ≈ 0.67
print(f"\n--- Level 2: 库存决策优化 (Target SL={target_service_level:.2f}) ---")

# 计算预测误差 (Residuals) - 使用训练集误差来估计分布
train['resid_rf'] = train[target] - rf_model.predict(train[features])
sigma_hat = train['resid_rf'].std()  # 参数法需要的标准差
residuals_sample = train['resid_rf'].values  # SAA需要的误差样本
resid_for_shapiro = train['resid_rf'].sample(n=min(5000, len(train)), random_state=42).values
sw_stat, sw_p = shapiro(resid_for_shapiro)
print(f"Shapiro-Wilk 正态性检验 p-value: {sw_p:.3e}")

# 方法 A: Parametric (Normal Distribution Assumption)
# q = y_hat + Z * sigma
z_score = norm.ppf(target_service_level)
test['q_normal'] = test['pred_rf'] + z_score * sigma_hat

# 方法 B: Non-parametric (SAA - Sample Average Approximation)
# q = y_hat + Empirical Quantile of Residuals
# 直接找历史误差的分位数
quantile_saa = np.quantile(residuals_sample, target_service_level)
test['q_saa'] = test['pred_rf'] + quantile_saa

# --- 计算真实的报童成本 (Ex-post Cost) ---
def calculate_cost(q, d, cu, co):
    # q: 订货量, d: 真实需求
    overage = np.maximum(q - d, 0)
    underage = np.maximum(d - q, 0)
    return cu * underage + co * overage

test['cost_normal'] = calculate_cost(test['q_normal'], test[target], Cu, Co)
test['cost_saa'] = calculate_cost(test['q_saa'], test[target], Cu, Co)

avg_cost_normal = test['cost_normal'].mean()
avg_cost_saa = test['cost_saa'].mean()

print(f"平均日成本 (Normal假设): {avg_cost_normal:.4f}")
print(f"平均日成本 (SAA数据驱动): {avg_cost_saa:.4f}")
if avg_cost_saa < avg_cost_normal:
    print(">> 结论: SAA (非参数方法) 优于 正态分布假设！(符合论文结论)")
else:
    print(">> 结论: 两种方法差异不显著或正态假设表现更好。")
pd.DataFrame([{
    'train_size': len(train),
    'test_size': len(test),
    'r2_lr': r2_lr,
    'r2_rf': r2_rf,
    'shapiro_pvalue': sw_p,
    'avg_cost_normal': avg_cost_normal,
    'avg_cost_saa': avg_cost_saa
}]).to_csv(os.path.join(outdir, 'metrics_summary.csv'), index=False)
print(f"已写入 '{os.path.join(outdir, 'metrics_summary.csv')}'")

print("\n--- 服务水平敏感性分析 (Cost vs Service Level) ---")
avg_costs_normal = []
avg_costs_saa = []
avg_costs_integrated = []
for sl in service_levels:
    z = norm.ppf(sl)
    qn = test['pred_rf'] + z * sigma_hat
    qs = test['pred_rf'] + np.quantile(residuals_sample, sl)
    int_model = GradientBoostingRegressor(loss='quantile', alpha=sl, n_estimators=200, max_depth=3, random_state=42)
    int_model.fit(train[features], train[target])
    qi = np.clip(int_model.predict(test[features]), 0, None)
    cn = calculate_cost(qn, test[target], Cu, Co).mean()
    cs = calculate_cost(qs, test[target], Cu, Co).mean()
    ci = calculate_cost(qi, test[target], Cu, Co).mean()
    avg_costs_normal.append(cn)
    avg_costs_saa.append(cs)
    avg_costs_integrated.append(ci)
    better = "Integrated更优" if (ci < cn and ci < cs) else ("SAA更优" if cs < cn else "Normal更优或相当")
    print(f"SL={sl:.2f} | Normal={cn:.4f} | SAA={cs:.4f} | Integrated={ci:.4f} | {better}")

plt.figure(figsize=(8, 5))
plt.plot(service_levels, avg_costs_normal, 'r-o', label='Normal')
plt.plot(service_levels, avg_costs_saa, 'b-o', label='SAA')
plt.plot(service_levels, avg_costs_integrated, 'g-o', label='Integrated (Quantile)')
plt.xlabel("Service Level")
plt.ylabel("Average Daily Cost")
plt.title("Cost vs Service Level (Normal/SAA/Integrated)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
print(f"正在保存图片 '{os.path.join(outdir, 'level23_cost_curve.png')}' ...")
plt.savefig(os.path.join(outdir, 'level23_cost_curve.png'))
pd.DataFrame({
    'service_level': service_levels,
    'normal_cost': avg_costs_normal,
    'saa_cost': avg_costs_saa,
    'integrated_cost': avg_costs_integrated
}).to_csv(os.path.join(outdir, 'costs_service_levels.csv'), index=False)
print(f"已写入 '{os.path.join(outdir, 'costs_service_levels.csv')}'")

print("\n--- Level 3: Integrated Optimization ---")
# ==========================================
# Level 3: Integrated Optimization (Quantile Regression)
# ==========================================
def level3_train_and_cost(train_df, test_df, feats, tgt, sl, grid, cu, co):
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = None
    best_cost = np.inf
    for n_estimators in grid['n_estimators']:
        for max_depth in grid['max_depth']:
            for learning_rate in grid['learning_rate']:
                cv_costs = []
                for tr_idx, val_idx in tscv.split(train_df[feats]):
                    X_tr = train_df[feats].iloc[tr_idx]
                    y_tr = train_df[tgt].iloc[tr_idx]
                    X_val = train_df[feats].iloc[val_idx]
                    y_val = train_df[tgt].iloc[val_idx]
                    mdl = GradientBoostingRegressor(loss='quantile', alpha=sl, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
                    mdl.fit(X_tr, y_tr)
                    q_val = np.clip(mdl.predict(X_val), 0, None)
                    cv_costs.append(calculate_cost(q_val, y_val.values, cu, co).mean())
                mean_cost = float(np.mean(cv_costs))
                if mean_cost < best_cost:
                    best_cost = mean_cost
                    best_params = (n_estimators, max_depth, learning_rate)
    final_model = GradientBoostingRegressor(loss='quantile', alpha=sl, n_estimators=best_params[0], max_depth=best_params[1], learning_rate=best_params[2], random_state=42)
    final_model.fit(train_df[feats], train_df[tgt])
    q_test = np.clip(final_model.predict(test_df[feats]), 0, None)
    test_cost = calculate_cost(q_test, test_df[tgt], cu, co).mean()
    return test_cost, best_params

avg_costs_integrated_tuned = []
best_params_list = []
if args.tune_level3:
    for sl in service_levels:
        cost_i, best_params = level3_train_and_cost(train, test, features, target, sl, gb_grid, Cu, Co)
        avg_costs_integrated_tuned.append(cost_i)
        best_params_list.append((sl, best_params))
        print(f"SL={sl:.2f} | Integrated (tuned)={cost_i:.4f} | best={best_params}")
    plt.figure(figsize=(8, 5))
    plt.plot(service_levels, avg_costs_integrated_tuned, 'g-o', label='Integrated (tuned)')
    plt.xlabel("Service Level")
    plt.ylabel("Average Daily Cost")
    plt.title("Level 3: Integrated Cost vs Service Level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    print(f"正在保存图片 '{os.path.join(outdir, 'level3_cost_curve.png')}' ...")
    plt.savefig(os.path.join(outdir, 'level3_cost_curve.png'))
    pd.DataFrame({
        'service_level': service_levels,
        'integrated_tuned_cost': avg_costs_integrated_tuned
    }).to_csv(os.path.join(outdir, 'costs_service_levels_integrated_tuned.csv'), index=False)
    print(f"已写入 '{os.path.join(outdir, 'costs_service_levels_integrated_tuned.csv')}'")

# ==========================================
# 4. 可视化 (为 PPT 准备素材)
# ==========================================
# 选取某一个产品的一段时期进行绘图
sample_product = test['article'].unique()[0]
plot_data = test[test['article'] == sample_product].head(30)

plt.figure(figsize=(12, 6))
plt.plot(plot_data['date'], plot_data[target], 'k-o', label='Actual Demand', linewidth=2)
plt.plot(plot_data['date'], plot_data['pred_rf'], 'b--', label='Forecast (ML)', alpha=0.7)
# 画出 SAA 的订货量决策
plt.plot(plot_data['date'], plot_data['q_saa'], 'g-', label='Order Quantity (SAA)', linewidth=2)
# 填充库存覆盖区域
plt.fill_between(plot_data['date'], plot_data[target], plot_data['q_saa'], 
                 where=(plot_data['q_saa'] >= plot_data[target]), 
                 color='green', alpha=0.1, label='Overstock (Waste)')
plt.fill_between(plot_data['date'], plot_data[target], plot_data['q_saa'], 
                 where=(plot_data['q_saa'] < plot_data[target]), 
                 color='red', alpha=0.1, label='Stockout (Lost Sales)')

plt.title(f"Inventory Decision Visualization: {sample_product} (Level 2 SAA)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
print(f"\n正在保存图片 '{os.path.join(outdir, 'inventory_result.png')}' ...")
plt.savefig(os.path.join(outdir, 'inventory_result.png'))
summary_lines = []
summary_lines.append(f"Top {TOP_N} 产品: {', '.join(list(top_articles))}")
summary_lines.append(f"训练集大小: {len(train)}, 测试集大小: {len(test)}")
summary_lines.append(f"使用的特征: {', '.join(features)}")
summary_lines.append("")
summary_lines.append("--- Level 1: 预测精度评估 ---")
summary_lines.append(f"Linear Regression RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}, R^2: {r2_lr*100:.2f}%")
summary_lines.append(f"Random Forest RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R^2: {r2_rf*100:.2f}%")
summary_lines.append("")
summary_lines.append(f"--- Level 2: 库存决策优化 (Target SL={target_service_level:.2f}) ---")
summary_lines.append(f"Shapiro-Wilk 正态性检验 p-value: {sw_p:.3e}")
summary_lines.append(f"平均日成本 (Normal假设): {avg_cost_normal:.4f}")
summary_lines.append(f"平均日成本 (SAA数据驱动): {avg_cost_saa:.4f}")
summary_lines.append("")
summary_lines.append("--- 服务水平敏感性分析 (Cost vs Service Level) ---")
for sl, cn, cs, ci in zip(service_levels, avg_costs_normal, avg_costs_saa, avg_costs_integrated):
    better = "Integrated更优" if (ci < cn and ci < cs) else ("SAA更优" if cs < cn else "Normal更优或相当")
    summary_lines.append(f"SL={sl:.2f} | Normal={cn:.4f} | SAA={cs:.4f} | Integrated={ci:.4f} | {better}")
if args.tune_level3 and len(avg_costs_integrated_tuned) > 0:
    summary_lines.append("")
    summary_lines.append("--- Level 3: Integrated Optimization (tuned) ---")
    for (sl, params), cost in zip(best_params_list, avg_costs_integrated_tuned):
        summary_lines.append(f"SL={sl:.2f} | Integrated (tuned)={cost:.4f} | best={params}")
summary_lines.append("")
summary_lines.append("--- Level 3 方法说明 ---")
summary_lines.append("模型：GradientBoostingRegressor（loss=quantile, alpha=服务水平），预测结果进行非负裁剪")
summary_lines.append("调参：TimeSeriesSplit(3) + 网格 n_estimators∈{100,200,400}, max_depth∈{2,3,4}, learning_rate∈{0.03,0.05,0.1}")
summary_lines.append("结论：当前特征与数据规模下，Integrated 在高服务水平未稳定优于 SAA；需扩展外生变量与更细网格")
summary_lines.append("")
summary_lines.append("数据来源：")
summary_lines.append("原始交易数据：data/Bakery sales.csv")
summary_lines.append("清洗后日粒度数据：data/bakery_daily.csv")
report_path = os.path.join(outdir, "summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"已写入 '{report_path}'")
plt.show()
