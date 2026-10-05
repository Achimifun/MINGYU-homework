# -*- coding: utf-8 -*-
"""题目二：Kaggle 泰坦尼克号生存预测
EDA -> 缺失值处理 -> 特征工程 -> 模型比较（分层5折CV）-> 最佳模型评估 -> 生成提交文件

运行方式：在本目录下执行  python titanic_analysis.py
所有预处理（填充/标准化/编码）都放在 Pipeline/ColumnTransformer 内部，
交叉验证每一折只用该折训练部分拟合，无信息泄漏；随机种子固定 42。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.unicode_minus": False,
})

# ---------------------------------------------------------------- 数据录入
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """确定性特征派生（不拟合任何统计量，故可在 Pipeline 外做）"""
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = ["Mr", "Mrs", "Miss", "Master"]
    title = title.where(title.isin(common),
                        np.where(df["Sex"] == "female", "Mrs", "Rare"))
    df["Title"] = title
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


train = add_features(train)
test = add_features(test)
y = train["Survived"].to_numpy()
FEATS = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title",
         "FamilySize", "IsAlone", "HasCabin"]
X = train[FEATS]

# ---------------------------------------------------------------- 预处理（全部在 Pipeline 内，逐折拟合）
def make_preprocess() -> ColumnTransformer:
    return ColumnTransformer([
        # Age：中位数填充（约20%缺失，不可删行）+ 标准化
        ("age", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), ["Age"]),
        # Fare：中位数填充（测试集缺1个）+ log1p 压右偏 + 标准化
        ("fare", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("log", FunctionTransformer(np.log1p,
                                                       feature_names_out="one-to-one")),
                           ("scale", StandardScaler())]), ["Fare"]),
        # 名义变量：众数填充 + one-hot（Embarked 缺2个；Sex/Title 无缺失）
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]),
         ["Sex", "Embarked", "Title"]),
        # 有序/0-1 特征直接通过：Pclass 是有序变量，FamilySize/IsAlone/HasCabin 本身就是数字
        ("ord", "passthrough", ["Pclass", "FamilySize", "IsAlone", "HasCabin"]),
    ])


def make_models() -> dict:
    return {
        # ① 性别基线：只用 Sex 一个特征的逻辑回归（参照系）
        "Sex baseline (LR)": Pipeline([
            ("prep", ColumnTransformer(
                [("sex", OneHotEncoder(drop="if_binary"), ["Sex"])],
                remainder="drop")),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        # ② 逻辑回归：全部构造特征 + 标准化
        "Logistic Regression": Pipeline([
            ("prep", make_preprocess()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        # ③ 随机森林
        "Random Forest": Pipeline([
            ("prep", make_preprocess()),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                           random_state=SEED, n_jobs=-1)),
        ]),
        # ④ 梯度提升
        "Gradient Boosting": Pipeline([
            ("prep", make_preprocess()),
            ("clf", GradientBoostingClassifier(random_state=SEED)),
        ]),
    }


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------------------------------------------------------------- 图2：模型比较（分层5折 CV）
models = make_models()
scores = {}
for name, pipe in models.items():
    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    scores[name] = {"acc_mean": acc.mean(), "acc_std": acc.std(),
                    "auc_mean": auc.mean(), "auc_std": auc.std(), "acc": acc, "auc": auc}
    print(f"{name:<22} ACC {acc.mean():.4f} ± {acc.std():.4f}   "
          f"AUC {auc.mean():.4f} ± {auc.std():.4f}")

order = sorted(scores, key=lambda k: scores[k]["acc_mean"], reverse=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (metric, lab) in zip(axes, [("acc_mean", "Accuracy"),
                                    ("auc_mean", "ROC-AUC")]):
    std_key = metric.replace("_mean", "_std")
    means = [scores[k][metric] for k in order]
    stds = [scores[k][std_key] for k in order]          # 误差棒 = 折间标准差（不除 sqrt(5)）
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, means, xerr=stds, height=0.6,
            color="#4C72B0", error_kw={"capsize": 4})
    ax.set_yticks(ypos, order)
    ax.set_xlabel(lab)
    ax.set_xlim(min(means + stds) - 0.05, 1.0)
    for yp, m, s in zip(ypos, means, stds):              # 柱末标注均值（4位小数），位置避开误差棒
        ax.text(m + s + 0.02, yp, f"{m:.4f}", va="center", fontsize=10)
    ax.set_title(f"5-fold CV {lab} (error bar = fold std)")
fig.suptitle("Figure 2  Model Comparison (Stratified 5-fold, seed=42)", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 最佳模型
best_name = max(order, key=lambda k: (scores[k]["acc_mean"], scores[k]["auc_mean"]))
best = models[best_name]
print(f"\nbest model = {best_name}")

# ---------------------------------------------------------------- 图3：最佳模型评估
oof_label = cross_val_predict(best, X, y, cv=skf)                 # 全 891 人 OOF 预测标签
oof_proba = cross_val_predict(best, X, y, cv=skf,
                              method="predict_proba")[:, 1]       # OOF 预测概率

cm = confusion_matrix(y, oof_label)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        share = cm[i, j] / cm[i].sum()                            # 占该真实类行的比例
        ax.text(j, i, f"{cm[i, j]}\n({share:.1%})",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Not Survived", "Survived"])
ax.set_yticks([0, 1], ["Not Survived", "Survived"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Out-of-fold Confusion Matrix ({best_name})")

ax = axes[1]
for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):            # 每折各自的 ROC
    m = clone(best).fit(X.iloc[tr_idx], y[tr_idx])
    p = m.predict_proba(X.iloc[va_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y[va_idx], p)
    ax.plot(fpr, tpr, lw=1, alpha=0.5,
            label=f"fold {k + 1} (AUC={roc_auc_score(y[va_idx], p):.3f})")
fpr, tpr, _ = roc_curve(y, oof_proba)                             # 全部 OOF 概率的汇总 ROC
ax.plot(fpr, tpr, lw=2.5, color="black",
        label=f"pooled OOF (AUC={roc_auc_score(y, oof_proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="random (AUC=0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (out-of-fold)")
ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Figure 3  Best Model Evaluation", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_model_evaluation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("confusion matrix (rows=actual [dead, survived]):")
print(cm)

# ---------------------------------------------------------------- 图4：特征重要性（前15）
best.fit(X, y)                                                    # 全训练集拟合最佳模型
names = best.named_steps["prep"].get_feature_names_out()
clf = best.named_steps["clf"]
if hasattr(clf, "feature_importances_"):                          # 树模型
    imp = clf.feature_importances_
    imp_note = "feature_importances_"
else:                                                             # 逻辑回归：特征已标准化，用|系数|
    imp = np.abs(clf.coef_[0])
    imp_note = "|standardized coefficient|"
idx = np.argsort(imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(9, 6))
ypos = np.arange(len(idx))[::-1]
ax.barh(ypos, imp[idx], height=0.6, color="#55A868")
ax.set_yticks(ypos, [names[i].split("__", 1)[-1] for i in idx])   # 去掉管道前缀，哑变量如实单列
ax.set_xlabel(f"Importance ({imp_note})")
ax.set_title(f"Figure 4  Top-{len(idx)} Feature Importance ({best_name})")
for yp, v in zip(ypos, imp[idx]):
    ax.text(v, yp, f" {v:.3f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 图1：EDA（2×2）
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]                                                   # (a) 生存率 × 性别 × 舱位
g = train.groupby(["Pclass", "Sex"])["Survived"].agg(["mean", "count"]).reset_index()
w = 0.35
for i, (sex, color) in enumerate([("female", "#4C72B0"), ("male", "#DD8452")]):
    sub = g[g["Sex"] == sex]
    xpos = sub["Pclass"] + (i - 0.5) * w
    ax.bar(xpos, sub["mean"], width=w, color=color, label=sex)
    for xp, m, n in zip(xpos, sub["mean"], sub["count"]):
        ax.text(xp, m + 0.02, f"n={n}", ha="center", fontsize=8)
ax.set_xticks([1, 2, 3], ["1st", "2nd", "3rd"])
ax.set_ylim(0, 1.05)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
ax.set_title("(a) Survival Rate by Sex and Class")
ax.legend()

ax = axes[0, 1]                                                   # (b) 年龄分布
for s, color in [(0, "#DD8452"), (1, "#4C72B0")]:
    ax.hist(train.loc[train["Survived"] == s, "Age"].dropna(),
            bins=np.arange(0, 85, 5), density=True, alpha=0.55,
            color=color, label=f"Survived = {s}")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
ax.set_title("(b) Age Distribution by Survival")
ax.legend()

ax = axes[1, 0]                                                   # (c) 票价分布（对数横轴）
for s, color in [(0, "#DD8452"), (1, "#4C72B0")]:
    fare = train.loc[(train["Survived"] == s) & (train["Fare"] > 0), "Fare"]
    ax.hist(fare, bins=np.logspace(np.log10(4), np.log10(600), 40),
            density=True, alpha=0.55, color=color, label=f"Survived = {s}")
ax.set_xscale("log")
ax.set_xlabel("Fare (log scale, zero fares excluded)")
ax.set_ylabel("Density")
ax.set_title("(c) Fare Distribution by Survival")
ax.legend()

ax = axes[1, 1]                                                   # (d) 家庭规模与生存率
g = train.groupby("FamilySize")["Survived"].agg(["mean", "count"])
ax.bar(g.index, g["mean"], color="#4C72B0")
for xp, m, n in zip(g.index, g["mean"], g["count"]):
    ax.text(xp, m + 0.02, f"n={n}", ha="center", fontsize=8)
ax.set_xlabel("Family Size (SibSp + Parch + 1)")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1.05)
ax.set_title("(d) Survival Rate by Family Size")

fig.suptitle("Figure 1  Exploratory Data Analysis", y=1.0)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_eda.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 生成提交文件
pred = best.predict(test[FEATS])
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})
submission.to_csv("submission.csv", index=False)

# ---------------------------------------------------------------- 自查（题目原文给的代码）
sub = pd.read_csv("submission.csv")
test_raw = pd.read_csv("data/test.csv")
assert list(sub.columns) == ["PassengerId", "Survived"], sub.columns
assert len(sub) == 418, len(sub)
assert sub["Survived"].isin([0, 1]).all(), "Survived 只能取 0 或 1"
assert set(sub["PassengerId"]) == set(test_raw["PassengerId"]), "编号与 test.csv 不匹配"
assert sub["PassengerId"].is_unique, "有重复编号"

print("\n格式检查通过；预测生还人数 =", int(sub["Survived"].sum()))
print(f"预测生还率 = {sub['Survived'].mean():.4f}（训练集生还率 0.3838，gender_submission 为 0.3636）")
for name in order:
    s = scores[name]
    print(f"[report] {name:<22} ACC {s['acc_mean']:.4f}±{s['acc_std']:.4f}  "
          f"AUC {s['auc_mean']:.4f}±{s['auc_std']:.4f}")
