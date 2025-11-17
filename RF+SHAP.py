import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV

# === 全局字体设置 ===
plt.rcParams['font.family'] = 'Times New Roman'   # 新罗马字体
plt.rcParams['font.weight'] = 'bold'              # 全局加粗
plt.rcParams['font.size'] = 20                    # 全局字号
plt.rcParams['axes.labelweight'] = 'bold'         # 坐标轴标题加粗
plt.rcParams['axes.titlesize'] = 22               # 图表标题字号
plt.rcParams['axes.titleweight'] = 'bold'         # 图表标题加粗
plt.rcParams['xtick.labelsize'] = 18              # X轴刻度字号
plt.rcParams['ytick.labelsize'] = 18              # Y轴刻度字号
plt.rcParams['legend.fontsize'] = 18              # 图例字体
plt.rcParams['legend.title_fontsize'] = 20        # 图例标题

# ========== Step 1: 读取并清洗数据 ==========
df = pd.read_csv("总表.csv", na_values=["#VALUE!"])
df = df.dropna()

X = df.drop(columns=["GI"])
y = df["GI"]

# ========== Step 2: 随机森林训练 ==========
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# ========== Step 3: SHAP 分析 ==========
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# ========== Step 4: 保存 shap value 表格 ==========
shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
os.makedirs("output", exist_ok=True)
shap_df.to_csv("output/shap_values.csv", index=True)
print("✅ 已保存 shap_values.csv 到 output 文件夹。")

# ========== Step 5: 保存特征重要性表 ==========
importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsShap": importance
}).sort_values(by="MeanAbsShap", ascending=False)
importance_df.to_csv("output/shap_importance.csv", index=False)
print("✅ 已保存 shap_importance.csv 到 output 文件夹。")

# ========== Step 6: 绘制并保存 SHAP 特征重要性图 ==========
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("SHAP Feature Importance", fontsize=22, weight="bold")
plt.xlabel("Mean |SHAP Value|", fontsize=20, weight="bold")
plt.ylabel("Features", fontsize=20, weight="bold")
plt.savefig("output/shap_summary_bar.png", bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot", fontsize=22, weight="bold")
plt.xlabel("SHAP Value", fontsize=20, weight="bold")
plt.ylabel("Features", fontsize=20, weight="bold")
plt.savefig("output/shap_summary_dot.png", bbox_inches='tight', dpi=300)
plt.close()
print("✅ 已保存 SHAP summary 图到 output 文件夹。")

# ========== Step 7: 绘制并保存 SHAP 依赖图 ==========
top_indices = importance.argsort()[-10:][::-1]
top_features = X.columns[top_indices]
print("Top features based on SHAP:", top_features.tolist())

for feat in top_features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X, show=False)
    plt.title(f"SHAP Dependence: {feat}", fontsize=22, weight="bold")
    plt.xlabel(feat, fontsize=20, weight="bold")
    plt.ylabel("SHAP Value", fontsize=20, weight="bold")
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.savefig(f"output/dependence_{feat}.png", bbox_inches='tight', dpi=300)
    plt.close()
print("✅ 已保存 Top 特征的 SHAP 依赖图到 output 文件夹。")
