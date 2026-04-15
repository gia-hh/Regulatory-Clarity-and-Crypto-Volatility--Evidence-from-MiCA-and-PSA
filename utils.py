"""
utils.py
========
公共工具函数，所有文件共用，避免重复代码。
"""

import os
import pandas as pd
import numpy as np
from config import OUTPUT_DIR


# ── 输出目录 ──────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 显著性星号 ────────────────────────────────────────────
def add_stars(pval: float) -> str:
    """
    返回显著性星号。
    *** p<0.01  ** p<0.05  * p<0.10
    """
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    return ""


# ── 回归结果 → DataFrame ──────────────────────────────────
def model_to_df(model, col_label: str, model_type: str = "ols") -> pd.DataFrame:
    """
    将 statsmodels OLS 或 linearmodels PanelOLS 的结果
    整理成两行格式（系数 + 括号标准误），方便横向合并输出。

    Parameters
    ----------
    model      : 已 fit 的模型对象
    col_label  : 列标签，如 "(1)"
    model_type : "ols" 或 "panel"
    """
    if model_type == "panel":
        params = model.params
        se     = model.std_errors
        pvals  = model.pvalues
    else:
        params = model.params
        se     = model.bse
        pvals  = model.pvalues

    rows = []
    for var in params.index:
        coef_str = f"{params[var]:.4f}{add_stars(pvals[var])}"
        se_str   = f"({se[var]:.4f})"
        rows.append({"Variable": var, f"{col_label}_coef": coef_str, f"{col_label}_se": se_str})

    return pd.DataFrame(rows)


# ── 多模型横向拼接 ────────────────────────────────────────
def merge_models(model_dict: dict, model_type: str = "ols") -> pd.DataFrame:
    """
    输入 {列标签: model} 字典，输出横向拼接的汇总表。
    自动过滤固定效应行（C(Symbol)、C(Date)、EntityEffects 等）。
    """
    merged = None
    for label, model in model_dict.items():
        df = model_to_df(model, label, model_type)
        merged = df if merged is None else pd.merge(merged, df, on="Variable", how="outer")

    # 过滤固定效应
    fe_prefixes = ("C(Symbol)", "C(Date)", "EntityEffects", "TimeEffects", "Intercept")
    mask = ~merged["Variable"].str.startswith(fe_prefixes)
    return merged[mask].reset_index(drop=True)


# ── 附加统计信息行 ────────────────────────────────────────
def append_stats(summary: pd.DataFrame, model_dict: dict, model_type: str = "ols") -> pd.DataFrame:
    """在汇总表末尾追加 Obs / R² / Adj.R² 行。"""
    stat_rows = {"Variable": ["Observations", "R²", "Adj. R²"]}

    for label, model in model_dict.items():
        if model_type == "panel":
            nobs   = int(model.nobs)
            r2     = model.rsquared
            r2_adj = model.rsquared  # linearmodels 无 adj R²，用 rsquared 代替
        else:
            nobs   = int(model.nobs)
            r2     = model.rsquared
            r2_adj = model.rsquared_adj

        stat_rows[f"{label}_coef"] = [str(nobs), f"{r2:.3f}", f"{r2_adj:.3f}"]
        stat_rows[f"{label}_se"]   = ["", "", ""]

    return pd.concat([summary, pd.DataFrame(stat_rows)], ignore_index=True)


# ── 保存 LaTeX ────────────────────────────────────────────
def save_latex(df: pd.DataFrame, filename: str, caption: str = "") -> None:
    """
    将 DataFrame 保存为 LaTeX 表格。
    文件写入 OUTPUT_DIR。
    """
    path = os.path.join(OUTPUT_DIR, filename)
    latex_str = df.to_latex(index=False, escape=False)

    if caption:
        latex_str = latex_str.replace(
            r"\begin{tabular}",
            f"\\caption{{{caption}}}\n\\begin{{tabular}}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  → 已保存：{path}")


# ── 事件窗口筛选 ──────────────────────────────────────────
def event_window(df: pd.DataFrame, event_date: pd.Timestamp, days: int) -> pd.DataFrame:
    """返回 event_date ± days 天内的子集（df 需含 Date 列）。"""
    lo = event_date - pd.Timedelta(days=days)
    hi = event_date + pd.Timedelta(days=days)
    return df[(df["Date"] >= lo) & (df["Date"] <= hi)].copy()
