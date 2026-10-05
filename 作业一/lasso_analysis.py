# -*- coding: utf-8 -*-
"""题目一：酱油风味物质 Lasso 降维

Lasso 回归 + 10 折交叉验证,从 40 个风味指标中筛选与咸味评分相关的变量。
一键运行:python lasso_analysis.py(相对路径,输出三张 300 dpi 图到 figures/)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.model_selection import KFold

RNG = 42

# 学术插图风格:Times New Roman,英文标注
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 中文列名 -> 英文标签(任务书附录)
NAME_MAP = {
    '丙酮酸': 'Pyruvic acid', '酒石酸': 'Tartaric acid', 'L-苹果酸': 'Malic acid',
    '焦谷氨酸': 'Pyroglutamic acid', '乳酸': 'Lactic acid', '富马酸': 'Fumaric acid',
    '琥珀酸': 'Succinic acid', '草酸': 'Oxalic acid', '柠檬酸': 'Citric acid',
    '维生素C': 'Ascorbic acid',
    "5'-AMP": "5'-AMP", "5'-GMP": "5'-GMP", "5'-CMP": "5'-CMP",
    "5'-UMP": "5'-UMP", "5'-IMP": "5'-IMP",
    'Ala丙氨酸': 'Ala', 'Arg精氨酸': 'Arg', 'Asp天冬氨酸': 'Asp', 'Glu谷氨酸': 'Glu',
    'Gly甘氨酸': 'Gly', 'His组氨酸': 'His', 'Ile异亮氨酸': 'Ile', 'Leu亮氨酸': 'Leu',
    'Lys赖氨酸': 'Lys', 'Met蛋氨酸': 'Met', 'Phe苯丙氨酸': 'Phe', 'Pro脯氨酸': 'Pro',
    'Ser丝氨酸': 'Ser', 'Thr苏氨酸': 'Thr', 'Trp色氨酸': 'Trp', 'Tyr酪氨酸': 'Tyr',
    'Val缬氨酸': 'Val', 'cys胱氨酸': 'Cys',
    'Na+': 'Na+', 'K+': 'K+', 'Ca2+': 'Ca2+', 'Mg2+': 'Mg2+', 'Cl-': 'Cl-',
    'NSSS': 'NSSS', 'NaCl': 'NaCl',
}

# ---------- 1. 读数据 ----------
df_std = pd.read_csv('data/soy_sauce_flavor_standardized.csv', encoding='utf-8')
df_raw = pd.read_csv('data/soy_sauce_flavor_raw.csv', encoding='utf-8')

X_raw = df_raw.drop(columns=['ID', '咸味评分'])
y = df_raw['咸味评分'].to_numpy(dtype=float)

# 验证标准化文件确为 raw 的 Z-score(不进 Pipeline,仅核对)
z_check = (X_raw - X_raw.mean()) / X_raw.std(ddof=0)
X_std_file = df_std.drop(columns=['ID', '咸味评分'])
max_dev = np.abs(z_check.to_numpy() - X_std_file.to_numpy()).max()
print(f'Z-score 校验:standardized 文件与 raw 计算的最大偏差 = {max_dev:.2e}')

X = df_std.drop(columns=['ID', '咸味评分']).to_numpy(dtype=float)
features_en = [NAME_MAP[c] for c in df_std.columns[2:]]
print(f'数据规模: X={X.shape}, y={y.shape}')

# ---------- 2. LassoCV:10 折交叉验证 ----------
# 新版 sklearn 移除了 LassoCV 的 n_alphas 参数,先用 lasso_path 生成 200 个 α 的等比对数网格
alphas_grid = lasso_path(X, y, n_alphas=200, max_iter=20000)[0]
# 与参考值对齐的标准 10 折划分(顺序切分,不打乱)
cv = KFold(n_splits=10)
model = LassoCV(alphas=alphas_grid, max_iter=20000, cv=cv, random_state=RNG, n_jobs=-1)
model.fit(X, y)

alphas = model.alphas_                      # 降序
mse_path = model.mse_path_                  # (n_alphas, 10)
mean_mse = mse_path.mean(axis=1)
se_mse = mse_path.std(axis=1, ddof=1) / np.sqrt(10)

i_min = int(np.argmin(mean_mse))
lambda_min = alphas[i_min]
se_at_min = se_mse[i_min]
threshold = mean_mse[i_min] + se_at_min

# 1SE 准则:在 MSE ≤ min+SE 的约束下取最大的 λ(最简约模型)
ok = np.where(mean_mse <= threshold)[0]
i_1se = int(ok[0])                          # alphas 降序,ok[0] 即最大 λ
lambda_1se = alphas[i_1se]

log_min, log_1se = np.log10(lambda_min), np.log10(lambda_1se)
print(f'lambda_min = {lambda_min:.6g}  (log10 = {log_min:.3f})')
print(f'lambda_1se = {lambda_1se:.6g}  (log10 = {log_1se:.3f})')

# λmin 下的系数(完整路径用同一 alpha 网格重新计算,便于画路径图)
coefs_all = np.zeros((len(alphas), X.shape[1]))   # (n_alphas, 40),与 alphas 对齐
coefs_path = lasso_path(X, y, alphas=alphas, max_iter=20000)[1]
coefs_all = coefs_path.T                    # lasso_path 返回 (n_features, n_alphas)

coef_min = model.coef_
nonzero = np.flatnonzero(coef_min != 0)
zeroed = np.flatnonzero(coef_min == 0)
print(f'λmin 下保留非零指标 {len(nonzero)}/40')
print('被压缩为 0 的指标:',
      ', '.join(features_en[j] for j in zeroed) or '(无)')

# ---------- 3. 图 1:Lasso 系数路径 ----------
fig, ax = plt.subplots(figsize=(9, 6.5))
cmap = plt.get_cmap('tab20')
for j in range(X.shape[1]):
    ax.plot(np.log10(alphas), coefs_all[:, j], lw=1.1,
            color=cmap(j % 20), alpha=0.85,
            label=features_en[j] if j < 20 else None)
ax.axvline(log_min, color='black', ls='--', lw=1.4,
           label=f'$\\log_{{10}}(\\lambda_{{\\min}})$ = {log_min:.3f}')
ax.axvline(log_1se, color='red', ls=':', lw=1.6,
           label=f'$\\log_{{10}}(\\lambda_{{1se}})$ = {log_1se:.3f}')
ax.set_xlabel(r'$\log_{10}(\lambda)$')
ax.set_ylabel('Regression coefficient (standardized)')
ax.set_title('Lasso Coefficient Profiles (40 flavor indicators)')
ax.axhline(0, color='grey', lw=0.6, alpha=0.6)
ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
fig.tight_layout()
fig.savefig('figures/fig1_lasso_paths.png', dpi=300)
plt.close(fig)

# ---------- 4. 图 2:CV 误差曲线 ----------
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(np.log10(alphas), mean_mse, yerr=se_mse,
            fmt='o-', ms=2.8, lw=1.1, elinewidth=0.8, capsize=1.6,
            color='steelblue', ecolor='steelblue', alpha=0.9)
ax.axvline(log_min, color='black', ls='--', lw=1.4,
           label=f'$\\lambda_{{\\min}}$: $\\log_{{10}}$ = {log_min:.3f}')
ax.axvline(log_1se, color='red', ls=':', lw=1.6,
           label=f'$\\lambda_{{1se}}$: $\\log_{{10}}$ = {log_1se:.3f}')
ax.annotate(f'$\\lambda_{{\\min}}$ = {lambda_min:.4g}\n'
            f'$\\lambda_{{1se}}$ = {lambda_1se:.4g}\n'
            f'MSE at $\\lambda_{{\\min}}$ = {mean_mse[i_min]:.4f}',
            xy=(log_min, mean_mse[i_min]), xytext=(log_min + 0.35, mean_mse[i_min] + 0.05),
            fontsize=9, bbox=dict(boxstyle='round', fc='whitesmoke', ec='grey', alpha=0.9),
            arrowprops=dict(arrowstyle='->', lw=0.8))
ax.set_xlabel(r'$\log_{10}(\lambda)$')
ax.set_ylabel('Cross-validated MSE (mean ± SE, 10-fold)')
ax.set_title('Lasso 10-fold Cross-Validation Error Curve')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('figures/fig2_cv_mse.png', dpi=300)
plt.close(fig)

# ---------- 5. 图 3:非零系数条形图 ----------
nz = nonzero[np.argsort(coef_min[nonzero])]
fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(nz) + 2), constrained_layout=True)
bars = ax.barh([features_en[j] for j in nz], coef_min[nz],
               color='steelblue', edgecolor='black', lw=0.4)
for b, v in zip(bars, coef_min[nz]):
    ax.text(v + (0.012 if v >= 0 else -0.012), b.get_y() + b.get_height() / 2,
            f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Lasso coefficient (at $\\lambda_{\\min}$, standardized scale)')
ax.set_title(f'Non-zero Coefficients Selected by Lasso ({len(nz)} of 40 indicators)')
ax.set_xlim(coef_min[nz].min() * 1.22, coef_min[nz].max() * 1.22)
fig.savefig('figures/fig3_coefficients.png', dpi=300)
plt.close(fig)

# ---------- 6. 报告数据 ----------
print('\n=== 非零系数(按系数从大到小) ===')
for j in sorted(nonzero, key=lambda j: -abs(coef_min[j])):
    print(f'{features_en[j]:<18s} {coef_min[j]:+.4f}')
print('\n完成:三张图已输出到 figures/')
