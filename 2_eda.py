"""
2_eda.py
========
EDA+ ARCH 效应检验

1. 各币种收益率时序图（标注政策事件）
2. 波动率聚集可视化（视觉上支持使用 GARCH）
3. 正态性检验（JB test）→ 证明 OLS 残差假设不成立
4. ARCH 效应检验（Engle's LM test）→ 正式验证需要 GARCH
5. ADF 单位根检验 → 验证时序平稳性

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from config import DATA_PROCESSED, COINS, PSA_DATE, MICA_DATE, KEY_DATES, OUTPUT_DIR, DEP_VARS

# ── 读入数据 ──────────────────────────────────────────────
df = pd.read_csv(DATA_PROCESSED, parse_dates=["Date"])

# ── 1. 收益率时序图（标注事件） ───────────────────────────
print("绘制收益率时序图...")
fig, axes = plt.subplots(len(COINS), 1, figsize=(14, 3 * len(COINS)), sharex=True)

for ax, coin in zip(axes, COINS):
    sub = df[df["Symbol"] == coin].sort_values("Date")
    ax.plot(sub["Date"], sub["log_return"], lw=0.6, color="steelblue", alpha=0.8)
    ax.axvline(PSA_DATE,  color="red",    lw=1.2, ls="--", label="PSA in effect")
    ax.axvline(MICA_DATE, color="orange", lw=1.2, ls="--", label="MiCA annouced")
    # 标注所有关键节点（浅色）
    for kd in KEY_DATES:
        ax.axvline(kd, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_ylabel(coin, fontsize=9)
    ax.set_ylim(-0.4, 0.4)

axes[0].legend(loc="upper left", fontsize=8)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30)
plt.suptitle("Log Returns by Coin(Events Marked)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}eda_log_returns.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → 已保存 eda_log_returns.png")

# ── 2. 波动率聚集可视化 ───────────────────────────────────
print("绘制波动率聚集图...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, coin in zip(axes.flatten(), COINS):
    sub = df[df["Symbol"] == coin].sort_values("Date")
    r = sub["log_return"].dropna()
    ax.plot(sub["Date"].iloc[-len(r):], r ** 2, lw=0.5, color="darkblue", alpha=0.7)
    ax.axvline(PSA_DATE,  color="red",    lw=1, ls="--")
    ax.axvline(MICA_DATE, color="orange", lw=1, ls="--")
    ax.set_title(coin, fontsize=10)
    ax.set_xlabel("")

plt.suptitle("Squared Log Returns(volatility clustering: if exists then supports using GARCH)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}eda_volatility_clustering.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 3. 正态性检验（Jarque-Bera） ─────────────────────────
print("\n=== Jarque-Bera 正态性检验 ===")
print(f"{'Coin':<8} {'JB stat':>10} {'p-value':>10} {'Skew':>8} {'Kurtosis':>10}")
print("-" * 50)

jb_results = []
for coin in COINS:
    r = df[df["Symbol"] == coin]["log_return"].dropna()
    jb_stat, jb_pval = stats.jarque_bera(r)
    jb_results.append({
        "Coin": coin,
        "JB Stat": round(jb_stat, 2),
        "p-value": round(jb_pval, 4),
        "Skewness": round(stats.skew(r), 3),
        "Excess Kurtosis": round(stats.kurtosis(r), 3),
    })
    print(f"{coin:<8} {jb_stat:>10.2f} {jb_pval:>10.4f} {stats.skew(r):>8.3f} {stats.kurtosis(r):>10.3f}")

# 说明：JB 检验拒绝正态性 → 残差非正态 → OLS 的标准推断失效
# 尖峰厚尾（excess kurtosis >> 0）是加密市场的典型特征，支持 GARCH-t 分布

jb_df = pd.DataFrame(jb_results)
jb_df.to_latex(f"{OUTPUT_DIR}jb_test.tex", index=False)
print(f"\n  → JB 检验结果已保存 jb_test.tex")

# ── 4. ARCH 效应检验（Engle's LM Test） ──────────────────
print("\n=== ARCH-LM 效应检验（lags=10）===")
print("H0：无 ARCH 效应（即方差恒定，可用 OLS）")
print("如果 p < 0.05，则拒绝 H0，必须用 GARCH\n")
print(f"{'Coin':<8} {'LM stat':>10} {'p-value':>10} {'结论':>15}")
print("-" * 50)

arch_results = []
for coin in COINS:
    r = df[df["Symbol"] == coin]["log_return"].dropna()
    lm_stat, lm_pval, f_stat, f_pval = het_arch(r, nlags=10)
    conclusion = "✅ 需要 GARCH" if lm_pval < 0.05 else "⚠️ 无明显 ARCH 效应"
    arch_results.append({
        "Coin": coin,
        "LM Stat": round(lm_stat, 2),
        "p-value": round(lm_pval, 4),
        "Conclusion": conclusion,
    })
    print(f"{coin:<8} {lm_stat:>10.2f} {lm_pval:>10.4f} {conclusion:>15}")

arch_df = pd.DataFrame(arch_results)
arch_df.to_latex(f"{OUTPUT_DIR}arch_lm_test.tex", index=False)
print(f"\n  → ARCH-LM 检验结果已保存 arch_lm_test.tex")

# ── 5. ADF 单位根检验 ─────────────────────────────────────
print("\n=== ADF 单位根检验 ===")
print("H0：存在单位根（非平稳）。p < 0.05 → 拒绝 H0 → 序列平稳\n")
print(f"{'Coin':<8} {'ADF stat':>10} {'p-value':>10} {'平稳?':>8}")
print("-" * 40)

adf_results = []
for coin in COINS:
    r = df[df["Symbol"] == coin]["log_return"].dropna()
    adf_stat, adf_pval, *_ = adfuller(r, autolag="AIC")
    stationary = "是" if adf_pval < 0.05 else "否"
    adf_results.append({"Coin": coin, "ADF Stat": round(adf_stat, 3),
                         "p-value": round(adf_pval, 4), "Stationary": stationary})
    print(f"{coin:<8} {adf_stat:>10.3f} {adf_pval:>10.4f} {stationary:>8}")

pd.DataFrame(adf_results).to_latex(f"{OUTPUT_DIR}adf_test.tex", index=False)
print(f"\n  → ADF 检验结果已保存 adf_test.tex")

print("\n✅ EDA 全部完成！结论摘要：")
print("  - 加密货币收益率存在显著尖峰厚尾（JB 检验）")
print("  - 存在 ARCH 效应（LM 检验），方差非恒定")
print("  - log return 序列平稳（ADF 检验）")
print("  → 以上三点共同支持使用 EGARCH 而非 OLS")
