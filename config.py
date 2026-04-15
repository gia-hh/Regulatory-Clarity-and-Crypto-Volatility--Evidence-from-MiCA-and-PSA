"""
config.py
=========
全局配置：所有文件共享的常量、路径、变量名。
修改这里一处，所有文件自动同步。
"""

import pandas as pd

# ── 事件日期 ─────────────────────────────────────────────
PSA_DATE  = pd.Timestamp("2020-01-28")   # 新加坡支付服务法正式生效
MICA_DATE = pd.Timestamp("2023-07-19")   # MiCA 正式发布于欧盟官方公报

# PSA 关键节点（用于 EDA 可视化标注）
KEY_DATES = pd.to_datetime([
    "2019-01-14",  # PSA 草案提交国会
    "2019-02-11",  # PSA 一读
    "2020-01-28",  # PSA 正式生效  ← 主事件
    "2020-09-24",  # MAS 首批牌照开放申请
    "2022-06-30",  # MiCA 欧洲议会通过
    "2022-10-05",  # MiCA 理事会批准
    "2023-04-20",  # MiCA 欧洲议会最终投票
    "2023-06-29",  # MiCA 签署成为法律
    "2023-07-19",  # MiCA 发布于欧盟官方公报  ← 主事件
    "2024-06-30",  # MiCA 加密资产服务商部分生效
    "2024-12-30",  # MiCA 全面生效
])

# ── 币种 ─────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]

# 用于回归的简称（与 merged_data.csv 中 Symbol 列一致）
COINS = ["BTC", "ETH", "LTC", "DOGE", "BNB", "XRP"]

# ── 变量名 ────────────────────────────────────────────────
# 因变量
DEP_VARS = {
    "vol":  "7D_EWMA_Intraday_Price_Volatility",   # 价格波动率
    "liq":  "σ_LIQt",                               # 流动性波动率
    "ret":  "σ_Rt",                                 # 收益率波动率
}

# PSA 回归控制变量
CONTROLS_PSA = [
    "SEPU",             # 新加坡经济政策不确定性
    "Lagged_Return",
    "Log_Volume",
    "Log_Cap",
    "BitcoinDominance",
]

# MiCA 回归控制变量
CONTROLS_MICA = [
    "European_News_Index",   # 欧洲新闻不确定性指数
    "Lagged_Return",
    "Log_Volume",
    "Log_Cap",
    "BitcoinDominance",
]

# ── 事件窗口 ──────────────────────────────────────────────
# 短期窗口（天数）：用于面板 OLS
SHORT_WINDOWS = [30, 60, 90]   # ±天数

# ── 文件路径 ──────────────────────────────────────────────
DATA_RAW      = "merged_data.csv"       # 原始面板数据（Symbol × Date）
DATA_PROCESSED = "processed_data.csv"  # 经 1_data_processing.py 处理后
OUTPUT_DIR    = "output/"               # 图表、LaTeX 输出目录
