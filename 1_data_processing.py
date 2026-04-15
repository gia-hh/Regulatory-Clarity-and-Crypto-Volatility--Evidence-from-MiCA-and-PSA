"""
1_data_processing.py
====================
数据清洗与特征构造。输入 merged_data.csv，输出 processed_data.csv。

主要做了什么：
1. 统一数据类型，删除缺失值
2. 计算 log return（更符合金融时序分析假设）
3. 构造 Running Variable（距离事件日期的天数），用于后续窗口分析和 RDD
4. 标注政策虚拟变量（分别针对 PSA 和 MiCA）
5. 把公告日和生效日分开标注（用于检验市场是否提前反应）
6. 计算 Realized Volatility（作为 GARCH 建模的检验基准）

【为什么这样做】
- Running Variable 是 RDD 的核心：原始代码用 abs(RegSin)<=window 筛窗口，
  但 RegSin 是 0/1 dummy，这个逻辑是错的。
  正确做法是用「距事件的天数」作为 running variable。
- 公告日 vs 生效日分开：如果市场在公告日就反应了，说明信息被提前定价，
  这是"不确定性消除"假说的直接支撑证据。
"""

import pandas as pd
import numpy as np
from config import (
    DATA_RAW, DATA_PROCESSED,
    PSA_DATE, MICA_DATE, KEY_DATES,
    COINS,
)

# ── 读入数据 ──────────────────────────────────────────────
print("读入数据...")
df = pd.read_csv(DATA_RAW, parse_dates=["Date"])
df.sort_values(["Symbol", "Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ── 基础类型修正 ──────────────────────────────────────────
numeric_cols = [
    "Close", "Open", "High", "Low", "Volume",
    "Lagged_Return", "Log_Cap", "Log_Volume",
    "BitcoinDominance", "SEPU", "European_News_Index",
    "7D_EWMA_Intraday_Price_Volatility", "σ_LIQt", "σ_Rt",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["Close", "Date", "Symbol"], inplace=True)
print(f"  清洗后剩余行数: {len(df)}")

# ── Log Return（对数收益率） ──────────────────────────────
# 比简单收益率更接近正态分布，是金融时序的标准做法
df["log_return"] = df.groupby("Symbol")["Close"].transform(
    lambda x: np.log(x / x.shift(1))
)

# ── Realized Volatility（已实现波动率，滚动22日标准差） ────
# 作为波动率的另一个度量，用于与 GARCH 条件波动率对比
df["realized_vol_22d"] = df.groupby("Symbol")["log_return"].transform(
    lambda x: x.rolling(22, min_periods=10).std() * np.sqrt(252)
)

# ── Running Variable：距各事件日期的天数 ──────────────────
# 正数 = 事件之后，负数 = 事件之前
# 【关键修正】原始代码用 RegSin(0/1) 做 window 筛选是错误的
df["days_to_PSA"]  = (df["Date"] - PSA_DATE).dt.days
df["days_to_MiCA"] = (df["Date"] - MICA_DATE).dt.days

# ── 政策虚拟变量 ──────────────────────────────────────────
df["RegSin"] = (df["Date"] >= PSA_DATE).astype(int)
df["RegEU"]  = (df["Date"] >= MICA_DATE).astype(int)

# ── 分阶段虚拟变量（公告期 vs 生效期） ───────────────────
# PSA 节点
PSA_ANNOUNCE = pd.Timestamp("2019-01-14")  # 草案提交国会（市场开始关注）
df["PSA_announce"] = (
    (df["Date"] >= PSA_ANNOUNCE) & (df["Date"] < PSA_DATE)
).astype(int)
df["PSA_enforce"] = (df["Date"] >= PSA_DATE).astype(int)

# MiCA 节点
MICA_VOTE    = pd.Timestamp("2023-04-20")  # 欧洲议会最终投票
df["MiCA_announce"] = (
    (df["Date"] >= MICA_VOTE) & (df["Date"] < MICA_DATE)
).astype(int)
df["MiCA_enforce"] = (df["Date"] >= MICA_DATE).astype(int)

# ── Liquidity（若原始数据有 High/Low/Volume） ─────────────
if all(c in df.columns for c in ["High", "Low", "Volume"]):
    spread = df["High"] - df["Low"]
    df["Liquidity"] = np.where(spread > 0, df["Volume"] / spread, np.nan)

# ── 输出 ──────────────────────────────────────────────────
df.to_csv(DATA_PROCESSED, index=False)
print(f"✅ 处理完成，已保存至 {DATA_PROCESSED}")
print(f"   包含列：{list(df.columns)}")
print(df[["Symbol", "Date", "log_return", "days_to_PSA",
          "days_to_MiCA", "RegSin", "RegEU",
          "PSA_announce", "MiCA_announce"]].head(10))
