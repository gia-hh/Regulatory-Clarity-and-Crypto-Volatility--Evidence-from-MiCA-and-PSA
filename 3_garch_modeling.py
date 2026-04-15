"""
3_garch_modeling.py
===================
EGARCH 建模：为每个币种选择最优 GARCH 族模型，
并提取条件波动率序列供后续分析使用。

流程：
1. 对每个币种跑 GARCH(1,1) vs EGARCH(1,1)，用 AIC/BIC 选优
2. 检验残差（Ljung-Box），确认模型充分捕捉了波动率动态
3. 绘制 News Impact Curve（NIC），可视化正负冲击的非对称性
4. 输出条件波动率序列，加回主数据集

【为什么用 EGARCH 而非 GARCH】
- GARCH(1,1)：正负冲击对波动率影响对称
- EGARCH(1,1)：允许非对称（γ 参数），对数形式天然保证方差为正
- 对于监管政策冲击，坏消息（政策收紧预期）和好消息（不确定性消除）
  对波动率的影响很可能不对称，EGARCH 能检验这一点
- 用 AIC/BIC 对比让模型选择有数据依据，而不是"默认用 GARCH"

【参数选择：为什么是 (1,1)】
- (1,1) 是金融时序的标准起点：大量实证表明对日频数据已足够
- 我们会对 (1,1), (1,2), (2,1) 做 AIC/BIC 对比，让数据说话
- α+β 接近 1 时考虑 IGARCH（波动率高度持续的情况）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from config import DATA_PROCESSED, COINS, OUTPUT_DIR, PSA_DATE, MICA_DATE

warnings.filterwarnings("ignore")

df = pd.read_csv(DATA_PROCESSED, parse_dates=["Date"])

# ── 存储所有币种的条件波动率 ──────────────────────────────
cond_vol_all = []

# ── 模型比较汇总表 ────────────────────────────────────────
comparison_rows = []

print("=" * 60)
print("GARCH 模型选择（AIC/BIC 对比）")
print("=" * 60)

for coin in COINS:
    sub = df[df["Symbol"] == coin].sort_values("Date").copy()
    r   = sub["log_return"].dropna() * 100   # arch 库惯例：用百分比收益率

    # 对齐索引
    sub = sub.loc[r.index]

    print(f"\n── {coin} ──────────────────────")

    # ── 候选模型 ─────────────────────────────────────────
    candidates = {
        "GARCH(1,1)":  arch_model(r, vol="Garch",  p=1, q=1, dist="t"),
        "GARCH(1,2)":  arch_model(r, vol="Garch",  p=1, q=2, dist="t"),
        "GARCH(2,1)":  arch_model(r, vol="Garch",  p=2, q=1, dist="t"),
        "EGARCH(1,1)": arch_model(r, vol="EGARCH", p=1, q=1, dist="t"),
        "EGARCH(1,2)": arch_model(r, vol="EGARCH", p=1, q=2, dist="t"),
    }

    fit_results = {}
    for name, spec in candidates.items():
        try:
            res = spec.fit(disp="off", show_warning=False)
            fit_results[name] = res
            print(f"  {name:<15}  AIC={res.aic:>8.2f}  BIC={res.bic:>8.2f}")
            comparison_rows.append({
                "Coin": coin, "Model": name,
                "AIC": round(res.aic, 2), "BIC": round(res.bic, 2),
            })
        except Exception as e:
            print(f"  {name:<15}  ❌ 拟合失败: {e}")

    # ── 选最优模型（最小 AIC） ────────────────────────────
    if not fit_results:
        print(f"  ⚠️ {coin} 所有模型拟合失败，跳过")
        continue

    best_name = min(fit_results, key=lambda k: fit_results[k].aic)
    best_fit  = fit_results[best_name]
    print(f"\n  ✅ 最优模型: {best_name}")

    # ── 残差诊断：Ljung-Box ───────────────────────────────
    std_resid = best_fit.resid / best_fit.conditional_volatility
    lb_resid  = acorr_ljungbox(std_resid.dropna(),   lags=[10, 20], return_df=True)
    lb_sq     = acorr_ljungbox(std_resid.dropna()**2, lags=[10, 20], return_df=True)

    print("  残差 Ljung-Box（标准化残差）:")
    print(f"    lag=10: stat={lb_resid['lb_stat'].iloc[0]:.2f}, p={lb_resid['lb_pvalue'].iloc[0]:.3f}")
    print(f"    lag=20: stat={lb_resid['lb_stat'].iloc[1]:.2f}, p={lb_resid['lb_pvalue'].iloc[1]:.3f}")
    print("  平方残差 Ljung-Box（剩余 ARCH 效应）:")
    print(f"    lag=10: stat={lb_sq['lb_stat'].iloc[0]:.2f},   p={lb_sq['lb_pvalue'].iloc[0]:.3f}")
    # p > 0.05 表示残差无自相关，模型已充分捕捉波动率动态

    # ── 保存条件波动率 ────────────────────────────────────
    cond_vol = best_fit.conditional_volatility / 100   # 还原为小数
    sub_aligned = sub.loc[cond_vol.index].copy()
    sub_aligned["cond_vol_garch"] = cond_vol.values
    sub_aligned["best_garch_model"] = best_name
    cond_vol_all.append(sub_aligned[["Symbol", "Date", "cond_vol_garch", "best_garch_model"]])

    # ── 绘制条件波动率 + 事件标注 ─────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(sub_aligned["Date"], sub_aligned["log_return"], lw=0.5, color="steelblue")
    ax1.set_ylabel("Log Return")
    ax1.axvline(PSA_DATE,  color="red",    ls="--", lw=1.2, label="PSA")
    ax1.axvline(MICA_DATE, color="orange", ls="--", lw=1.2, label="MiCA")
    ax1.legend(fontsize=8)

    ax2.plot(sub_aligned["Date"], cond_vol.values, lw=0.7, color="darkred")
    ax2.set_ylabel("Conditional Volatility (GARCH)")
    ax2.axvline(PSA_DATE,  color="red",    ls="--", lw=1.2)
    ax2.axvline(MICA_DATE, color="orange", ls="--", lw=1.2)

    plt.suptitle(f"{coin} — {best_name} Conditional Volatility", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}garch_{coin}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── News Impact Curve（仅 EGARCH） ────────────────────
    if "EGARCH" in best_name:
        params = best_fit.params
        omega  = params.get("omega", 0)
        alpha  = params.get("alpha[1]", 0)
        gamma  = params.get("gamma[1]", 0)   # 非对称项
        beta   = params.get("beta[1]", 0)

        eps_range = np.linspace(-4, 4, 200)
        # EGARCH NIC: log(σ²) = ω + α|ε| + γε + β*log(σ²_prev)
        log_var   = omega + alpha * np.abs(eps_range) + gamma * eps_range
        nic       = np.exp(log_var)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eps_range, nic, color="navy", lw=1.5)
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.fill_between(eps_range[eps_range < 0], nic[eps_range < 0],
                        alpha=0.15, color="red",   label="负冲击（坏消息）")
        ax.fill_between(eps_range[eps_range > 0], nic[eps_range > 0],
                        alpha=0.15, color="green", label="正冲击（好消息）")
        ax.set_xlabel("标准化冲击 ε")
        ax.set_ylabel("条件方差变化")
        ax.set_title(f"{coin} — News Impact Curve\nγ={gamma:.4f}（γ<0 表示负冲击放大波动率）")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}nic_{coin}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  γ（非对称参数）= {gamma:.4f}  {'← 负冲击放大波动率 ✓' if gamma < 0 else ''}")

# ── 汇总模型比较表 ────────────────────────────────────────
comp_df = pd.DataFrame(comparison_rows)
comp_df.to_latex(f"{OUTPUT_DIR}garch_model_comparison.tex", index=False)
print(f"\n  → 模型比较表已保存 garch_model_comparison.tex")

# ── 将条件波动率合并回主数据集 ───────────────────────────
if cond_vol_all:
    cond_vol_df = pd.concat(cond_vol_all, ignore_index=True)
    df_with_garch = df.merge(cond_vol_df, on=["Symbol", "Date"], how="left")
    df_with_garch.to_csv(f"{OUTPUT_DIR}data_with_garch.csv", index=False)
    print(f"  → 含条件波动率的数据已保存 data_with_garch.csv")

print("\n✅ GARCH 建模全部完成！")
