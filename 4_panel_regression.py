"""
4_panel_regression.py
=====================
面板回归分析 + 稳健性检验。

模块：
A. 基准回归：全样本面板 OLS（保留，作为与 GARCH 的对比基准）
B. 多窗口分析：±30/60/90 天，检验结果稳定性
C. 分阶段分析：公告期 vs 生效期（检验市场是否提前反应）
D. 分币种异质性：DOGE vs ETH vs XRP 的系数对比
E. Placebo Test：随机安慰剂检验（排除伪相关）

【核心逻辑说明】
- 基准结论：两个政策都使波动率显著下降
- 解释框架：不确定性消除假说（监管明确性 > 监管方向）
- 异质性：如果 DOGE（投机性）比 ETH（基本面）反应更强，
  支持"散户更受不确定性影响"的解释
- 安慰剂检验：在随机日期重跑模型，如果也显著，说明结果可能是巧合
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf
from config import (
    DATA_PROCESSED, PSA_DATE, MICA_DATE,
    COINS, DEP_VARS, CONTROLS_PSA, CONTROLS_MICA,
    SHORT_WINDOWS, OUTPUT_DIR,
)
from utils import merge_models, append_stats, save_latex, event_window

warnings.filterwarnings("ignore")

# ── 读入数据 ──────────────────────────────────────────────
df = pd.read_csv(DATA_PROCESSED, parse_dates=["Date"])
df.dropna(subset=["log_return"], inplace=True)

dep_vars   = list(DEP_VARS.values())
dep_labels = list(DEP_VARS.keys())

# ══════════════════════════════════════════════════════════
# A. 基准回归：全样本面板 OLS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("A. 基准面板回归（全样本）")
print("=" * 60)

for policy, event_date, dummy, controls in [
    ("PSA",  PSA_DATE,  "RegSin", CONTROLS_PSA),
    ("MiCA", MICA_DATE, "RegEU",  CONTROLS_MICA),
]:
    print(f"\n── {policy} ──")
    window_df = event_window(df, event_date, days=90)
    window_df = window_df.set_index(["Symbol", "Date"])

    models = {}
    for i, dep in enumerate(dep_vars, 1):
        if dep not in window_df.columns:
            continue
        y = window_df[dep].dropna()
        X = window_df.loc[y.index, [dummy] + controls]
        X = X.dropna()
        y = y.loc[X.index]

        try:
            m = PanelOLS(y, X, entity_effects=True, time_effects=False)
            models[f"({i})"] = m.fit(cov_type="clustered", cluster_entity=True)
        except Exception as e:
            print(f"  {dep}: 回归失败 — {e}")

    if models:
        summary = merge_models(models, model_type="panel")
        summary = append_stats(summary, models, model_type="panel")
        save_latex(summary, f"baseline_{policy}.tex",
                   caption=f"{policy} 基准回归（±90天窗口，实体固定效应）")
        print(f"  → baseline_{policy}.tex")

# ══════════════════════════════════════════════════════════
# B. 多窗口分析（±30 / ±60 / ±90 天）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("B. 多窗口分析")
print("=" * 60)

dep_main = DEP_VARS["vol"]   # 主因变量：价格波动率

for policy, event_date, dummy, controls in [
    ("PSA",  PSA_DATE,  "RegSin", CONTROLS_PSA),
    ("MiCA", MICA_DATE, "RegEU",  CONTROLS_MICA),
]:
    print(f"\n── {policy} ──")
    rows = []

    for win in SHORT_WINDOWS:
        w_df = event_window(df, event_date, days=win)
        w_df = w_df.set_index(["Symbol", "Date"])

        if dep_main not in w_df.columns:
            continue
        y = w_df[dep_main].dropna()
        X = w_df.loc[y.index, [dummy] + controls].dropna()
        y = y.loc[X.index]

        try:
            m   = PanelOLS(y, X, entity_effects=True).fit(
                      cov_type="clustered", cluster_entity=True)
            coef = m.params[dummy]
            se   = m.std_errors[dummy]
            pval = m.pvalues[dummy]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            rows.append({
                "Window": f"±{win}d",
                "Coefficient": f"{coef:.4f}{stars}",
                "Std. Error":  f"({se:.4f})",
                "N": int(m.nobs),
            })
            print(f"  ±{win}d:  β={coef:.4f}{stars}  se={se:.4f}  N={int(m.nobs)}")
        except Exception as e:
            print(f"  ±{win}d: 失败 — {e}")

    if rows:
        win_df = pd.DataFrame(rows)
        save_latex(win_df, f"multiwindow_{policy}.tex",
                   caption=f"{policy} — {dep_main} 多窗口稳健性检验")

# ══════════════════════════════════════════════════════════
# C. 公告期 vs 生效期（检验市场是否提前反应）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("C. 公告期 vs 生效期分析")
print("=" * 60)
print("假设：如果公告期系数显著，说明市场在事件前已定价不确定性")

for policy, ann_dummy, enf_dummy, controls, event_date in [
    ("PSA",  "PSA_announce",  "PSA_enforce",  CONTROLS_PSA,  PSA_DATE),
    ("MiCA", "MiCA_announce", "MiCA_enforce", CONTROLS_MICA, MICA_DATE),
]:
    print(f"\n── {policy} ──")
    w_df = event_window(df, event_date, days=90).set_index(["Symbol", "Date"])

    models = {}
    for i, dep in enumerate(dep_vars, 1):
        if dep not in w_df.columns or ann_dummy not in w_df.columns:
            continue
        regressors = [ann_dummy, enf_dummy] + controls
        cols_avail = [c for c in regressors if c in w_df.columns]
        y = w_df[dep].dropna()
        X = w_df.loc[y.index, cols_avail].dropna()
        y = y.loc[X.index]
        try:
            m = PanelOLS(y, X, entity_effects=True).fit(
                    cov_type="clustered", cluster_entity=True)
            models[f"({i})"] = m
        except Exception as e:
            print(f"  {dep}: 失败 — {e}")

    if models:
        summary = merge_models(models, model_type="panel")
        # 只保留公告/生效两行
        key_vars = [ann_dummy, enf_dummy]
        summary_key = summary[summary["Variable"].isin(key_vars)]
        save_latex(summary_key, f"announce_vs_enforce_{policy}.tex",
                   caption=f"{policy} — 公告期 vs 生效期系数对比")
        print(summary_key.to_string(index=False))

# ══════════════════════════════════════════════════════════
# D. 分币种异质性（修正原始代码的 bug）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("D. 分币种异质性分析")
print("=" * 60)
print("【原始代码 bug 修正】：原代码 df = df[df['Symbol']==coin] 在循环内")
print("会导致每次 df 缩小，第二个币种开始数据已被截断，此处已修正。\n")

dep_main = DEP_VARS["vol"]
coef_plot = {"PSA": {}, "MiCA": {}}

for coin in COINS:
    coin_df = df[df["Symbol"] == coin].copy()   # ← 修正：每次从完整 df 取

    for policy, event_date, dummy, controls in [
        ("PSA",  PSA_DATE,  "RegSin", CONTROLS_PSA),
        ("MiCA", MICA_DATE, "RegEU",  CONTROLS_MICA),
    ]:
        w_df = event_window(coin_df, event_date, days=90)

        if dep_main not in w_df.columns:
            continue
        regressors = [dummy] + controls
        cols_avail = [c for c in regressors if c in w_df.columns]
        y = w_df[dep_main].dropna()
        X = w_df.loc[y.index, cols_avail].dropna()
        y = y.loc[X.index]

        if len(y) < 30:
            print(f"  {coin} {policy}: 样本量不足，跳过")
            continue

        try:
            # 单币种用 OLS（无法做面板固定效应）
            X_const = X.copy()
            X_const.insert(0, "const", 1.0)
            import statsmodels.api as sm
            m = sm.OLS(y.values, X_const.values).fit(cov_type="HC3")
            idx = list(X_const.columns).index(dummy)
            coef = m.params[idx]
            se   = m.bse[idx]
            pval = m.pvalues[idx]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            coef_plot[policy][coin] = (coef, se)
            print(f"  {coin:<6} {policy}:  β={coef:.4f}{stars:<3}  se={se:.4f}")
        except Exception as e:
            print(f"  {coin} {policy}: 失败 — {e}")

# 异质性系数图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (policy, coin_coefs) in zip(axes, coef_plot.items()):
    if not coin_coefs:
        continue
    coins_  = list(coin_coefs.keys())
    coefs_  = [v[0] for v in coin_coefs.values()]
    ses_    = [v[1] for v in coin_coefs.values()]
    colors_ = ["#e74c3c" if c < 0 else "#2ecc71" for c in coefs_]

    ax.barh(coins_, coefs_, xerr=ses_, color=colors_, alpha=0.8,
            error_kw={"elinewidth": 1.5, "capsize": 4})
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"{policy} — 分币种政策系数（{dep_main[:15]}）", fontsize=10)
    ax.set_xlabel("系数（负值 = 波动率下降）")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}heterogeneity_coins.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → 异质性系数图已保存 heterogeneity_coins.png")

# ══════════════════════════════════════════════════════════
# E. Placebo Test（安慰剂检验）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("E. Placebo Test（随机安慰剂检验）")
print("=" * 60)
print("方法：随机抽取 500 个虚假事件日期，重跑面板回归，")
print("检验真实系数是否显著大于随机系数分布的尾部。")
print("如果真实系数落在随机分布之外 → 结果非偶然\n")

np.random.seed(42)
N_PLACEBO  = 500
dep_main   = DEP_VARS["vol"]

for policy, event_date, dummy, controls in [
    ("PSA",  PSA_DATE,  "RegSin", CONTROLS_PSA),
    ("MiCA", MICA_DATE, "RegEU",  CONTROLS_MICA),
]:
    print(f"── {policy} Placebo ──")

    # 真实系数（用 ±90 天窗口）
    real_coef = None
    w_df_real = event_window(df, event_date, days=90).copy()
    w_df_real = w_df_real.set_index(["Symbol", "Date"])
    if dep_main in w_df_real.columns:
        y = w_df_real[dep_main].dropna()
        X = w_df_real.loc[y.index, [dummy] + [c for c in controls if c in w_df_real.columns]].dropna()
        y = y.loc[X.index]
        try:
            m = PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
            real_coef = m.params[dummy]
        except:
            pass

    # 安慰剂日期范围（排除真实事件 ±180 天）
    date_min = df["Date"].min() + pd.Timedelta(days=90)
    date_max = df["Date"].max() - pd.Timedelta(days=90)
    all_dates = pd.date_range(date_min, date_max, freq="D")
    exclude   = pd.date_range(event_date - pd.Timedelta(days=180),
                              event_date + pd.Timedelta(days=180), freq="D")
    valid_dates = [d for d in all_dates if d not in exclude]

    if len(valid_dates) < N_PLACEBO:
        print(f"  ⚠️ 可用安慰剂日期不足（{len(valid_dates)}），跳过")
        continue

    placebo_dates  = np.random.choice(valid_dates, N_PLACEBO, replace=False)
    placebo_coefs  = []

    for fake_date in placebo_dates:
        fake_date = pd.Timestamp(fake_date)
        w_df_p = event_window(df, fake_date, days=90).copy()
        w_df_p["fake_dummy"] = (w_df_p["Date"] >= fake_date).astype(int)
        w_df_p = w_df_p.set_index(["Symbol", "Date"])

        if dep_main not in w_df_p.columns:
            continue
        ctrl_avail = [c for c in controls if c in w_df_p.columns]
        y = w_df_p[dep_main].dropna()
        X = w_df_p.loc[y.index, ["fake_dummy"] + ctrl_avail].dropna()
        y = y.loc[X.index]
        if len(y) < 20:
            continue
        try:
            m = PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
            placebo_coefs.append(m.params["fake_dummy"])
        except:
            continue

    if len(placebo_coefs) < 50:
        print(f"  ⚠️ 有效安慰剂回归不足（{len(placebo_coefs)}），跳过")
        continue

    placebo_coefs = np.array(placebo_coefs)
    pct_rank = np.mean(placebo_coefs < real_coef) if real_coef is not None else None

    print(f"  真实系数:       {real_coef:.4f}" if real_coef else "  真实系数: N/A")
    print(f"  安慰剂均值:     {placebo_coefs.mean():.4f}")
    print(f"  安慰剂标准差:   {placebo_coefs.std():.4f}")
    print(f"  百分位排名:     {pct_rank:.1%}" if pct_rank else "")
    print(f"  有效安慰剂数:   {len(placebo_coefs)}")

    # 绘制安慰剂分布图
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(placebo_coefs, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    if real_coef is not None:
        ax.axvline(real_coef, color="red", lw=2, ls="--",
                   label=f"真实系数 = {real_coef:.4f}")
    ax.axvline(np.percentile(placebo_coefs, 5),  color="gray", lw=1, ls=":")
    ax.axvline(np.percentile(placebo_coefs, 95), color="gray", lw=1, ls=":",
               label="5th/95th percentile")
    ax.set_xlabel("安慰剂系数")
    ax.set_ylabel("频率")
    ax.set_title(f"{policy} Placebo Test — {dep_main[:20]}\n"
                 f"真实系数百分位: {pct_rank:.1%}" if pct_rank else f"{policy} Placebo Test")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}placebo_{policy}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 安慰剂分布图已保存 placebo_{policy}.png")

print("\n✅ 全部回归分析完成！")
print(f"   输出文件均在 {OUTPUT_DIR} 目录下")
