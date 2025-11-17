import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 尝试导入 XGBoost
try:
    import xgboost as xgb
    use_xgb = True
except:
    print("XGBoost not available.")
    use_xgb = False

# ======================= 公共函数 =======================
def build_dnn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# ======================= 数据读取 =======================
df = pd.read_csv("总表.csv")
df = df[df["GI"] > 0].copy()
X = df.drop(columns=["GI"])
y = df["GI"]

X = X.select_dtypes(include=np.number)
X = X.fillna(X.mean())
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.mean())

# ======================= 模型定义 =======================
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVM": SVR()
}
if use_xgb:
    models["XGBoost"] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

models["DNN"] = "DNN_PLACEHOLDER"  # 占位符

# ======================= 交叉验证 =======================
summary_results = []
detailed_results = []
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
best_fold_per_model = {}

for name, model in models.items():
    r2_scores = []
    rmse_scores = []
    best_rmse = float('inf')

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_cv)
        X_test_scaled = scaler.transform(X_test_cv)

        if name == "DNN":
            model_instance = build_dnn(X_train_scaled.shape[1])
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model_instance.fit(X_train_scaled, y_train_cv,
                               validation_split=0.2,
                               epochs=200,
                               batch_size=32,
                               shuffle=True,
                               callbacks=[early_stop],
                               verbose=0)
            y_pred = model_instance.predict(X_test_scaled).flatten()
        else:
            model_instance = model
            model_instance.fit(X_train_scaled, y_train_cv)
            y_pred = model_instance.predict(X_test_scaled)

        r2 = r2_score(y_test_cv, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

        detailed_results.append({
            "Model": name,
            "Fold": fold + 1,
            "R2": r2,
            "RMSE": rmse
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_fold_per_model[name] = {
                "train_idx": train_idx,
                "test_idx": test_idx
            }

    summary_results.append({
        "Model": name,
        "R2_Mean": np.mean(r2_scores),
        "RMSE_Mean": np.mean(rmse_scores)
    })

# ======================= 图像风格设置 =======================
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 12

# ======================= 图1: SCI风格预测拟合图（自定义顺序 & 美化） =======================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2行4列
axes = axes.flatten()

# ✅ 自定义模型显示顺序
desired_order = ["RandomForest", "XGBoost", "KNN", "SVM", "DecisionTree", "LinearRegression", "Ridge", "DNN"]
labels = [f"({chr(97+i)})" for i in range(len(desired_order))]

for i, name in enumerate(desired_order):
    if name not in best_fold_per_model:
        continue

    info = best_fold_per_model[name]
    ax = axes[i]
    train_idx = info["train_idx"]
    test_idx = info["test_idx"]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if name == "DNN":
        model_instance = build_dnn(X_train_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model_instance.fit(X_train_scaled, y_train,
                           validation_split=0.2,
                           epochs=200,
                           batch_size=32,
                           shuffle=True,
                           callbacks=[early_stop],
                           verbose=0)
    else:
        model_instance = models[name]
        model_instance.fit(X_train_scaled, y_train)

    y_train_pred = model_instance.predict(X_train_scaled).flatten()
    y_test_pred = model_instance.predict(X_test_scaled).flatten()

    # 指标计算
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # 拟合参考线（不加入图例）
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.8, alpha=0.6, label='_nolegend_')

    # 拟合点（含边缘线更易分辨）
    ax.scatter(y_train, y_train_pred, color='royalblue',
           label=f'Train (R²={r2_train:.2f})',
           alpha=0.6, s=30, edgecolor='k', linewidth=0.4)

    ax.scatter(y_test, y_test_pred, color='darkorange',
           label=f'Val (R²={r2_test:.2f})',
           alpha=0.6, s=30, edgecolor='k', linewidth=0.4)
    # 标签设置（加粗）
    ax.set_xlabel("True Pb", fontweight='bold')
    ax.set_ylabel("Predicted Pb", fontweight='bold')

    # 坐标刻度数字加粗
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    # 图例放在右上角不遮挡
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')

    # 子图编号（顶层 + 加粗 + 白底）
    ax.text(0.02, 0.95, labels[i],
        transform=ax.transAxes,
        fontsize=16,                     
        fontweight='bold',               
        va='top',
        zorder=10,
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.7))


# 删除空子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# 紧凑排列
plt.tight_layout(pad=1.0)
plt.savefig("all_models_best_fold_fits_SCI.png", dpi=600)
plt.close()





# ======================= 图2: 性能柱状图（RMSE & R²） =======================
summary_df = pd.DataFrame(summary_results).sort_values(by="RMSE_Mean")

fig, ax1 = plt.subplots(figsize=(10, 6))
bar_width = 0.4
x = np.arange(len(summary_df))

ax1.bar(x - bar_width/2, summary_df["RMSE_Mean"], width=bar_width, label='RMSE', color='gray')
ax1.set_ylabel("RMSE")
ax1.set_xlabel("Model")
ax1.set_xticks(x)
ax1.set_xticklabels(summary_df["Model"], rotation=45, ha="right")

ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, summary_df["R2_Mean"], width=bar_width, label='R²', color='lightblue')
ax2.set_ylabel("R²")

fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
fig.tight_layout()
plt.savefig("model_performance_barplot.png", dpi=300)
plt.close()

# ======================= 图3: RMSE箱线图 =======================
detailed_df = pd.DataFrame(detailed_results)
fig, ax = plt.subplots(figsize=(10, 6))
detailed_df.boxplot(column="RMSE", by="Model", ax=ax)
ax.set_title("RMSE Distribution (10-fold CV)")
ax.set_xlabel("Model")
ax.set_ylabel("RMSE")
plt.suptitle("")
plt.tight_layout()
plt.savefig("model_rmse_boxplot.png", dpi=300)
plt.close()

# ======================= 保存 CSV 结果 =======================
summary_df.to_csv("model_evaluation_summary.csv", index=False)
detailed_df.to_csv("model_evaluation_detailed.csv", index=False)

print("✅ 所有图像和结果已生成完毕。")
