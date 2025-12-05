import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import ruptures as rpt
import itertools
import warnings

from sklearn.linear_model import BayesianRidge   # <-- 新增：BLR

warnings.filterwarnings("ignore")

def auto_cp_by_pen_soft(signal, model="rbf", algo_cls=rpt.Pelt,
                        pen_grid=None, max_cps=10):
    """
    更温和的 auto cp：只用 Pelt 自己的 penalty，不再外加 k*log(n)
    在一组 pen_grid 里选“cost 最小且变点数不过多”的那组
    """
    n = len(signal)
    if pen_grid is None:
        base = np.log(n)
        pen_grid = [0.2*base, 0.5*base, 1*base, 2*base, 3*base]

    best_cost = np.inf
    best_bkps = None
    best_pen = None
    best_k = None

    for pen in pen_grid:
        algo = algo_cls(model=model).fit(signal)
        bkps = algo.predict(pen=pen)
        k = len(bkps) - 1
        if k > max_cps:
            continue

        cost = algo.cost.sum_of_costs(bkps)

        if cost < best_cost:
            best_cost = cost
            best_bkps = bkps
            best_pen = pen
            best_k = k

    return best_bkps, best_pen, best_k

def make_lagged_matrix(series, lag):
    """
    根据一维时间序列构造滞后特征:
    X_t = [r_{t-1}, ..., r_{t-lag}], y_t = r_t
    返回 X, y (均为 ndarray)
    """
    values = np.asarray(series)
    n = len(values)
    if n <= lag:
        raise ValueError("数据太短，无法构造 lag={} 的特征".format(lag))

    X = []
    y = []
    for t in range(lag, n):
        X.append(values[t-lag:t])
        y.append(values[t])

    X = np.array(X)
    y = np.array(y)
    return X, y

# =============================
# 1. 数据读取 & 基本预处理
# =============================
ticker = 'GOOGL'

df = pd.read_csv(
    './data/'+ticker+'_stock_data.csv',
    parse_dates=['Date'],
    index_col='Date'
)

print("原始列名：", df.columns)

# 取 Close 列并转为 float
price = df['Close'].astype(float).dropna().sort_index()

# 如果有重复日期，取最后一个
price = price[~price.index.duplicated(keep='last')]

# 将索引设置为交易日频率（Business Day），缺失日用前值填充
price = price.asfreq('B').ffill()
# 重采样为月度价格（和原策略保持一致）
price = price.resample('M').last()

# =============================
# 2. 使用对数收益率
# =============================
log_price = np.log(price)
returns = log_price.diff().dropna()  # 月度 log return

result = adfuller(returns)
print("ADF 统计量:", result[0])
print("p-value:", result[1])

# 简单画一下价格和收益率
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(price)
axes[0].set_title(ticker+' Close Price')
axes[1].plot(returns)
axes[1].set_title(ticker+' Log Returns (Monthly)')
plt.tight_layout()
plt.show()

# =============================
# 3. 变点检测（CP）——在收益率上找 regime 切换
# =============================
signal = returns.values.reshape(-1, 1)

bkps, best_pen, best_k = auto_cp_by_pen_soft(
    signal, model="rbf", algo_cls=rpt.Pelt, max_cps=6
)
print("最佳 penalty:", best_pen)
print("对应变点个数:", len(bkps) - 1)
print("检测到的变点位置（索引）:", bkps)

rpt.display(signal, bkps)
plt.title('Change Point Detection on Log Returns')
plt.show()

dates = returns.index
cp_dates = [dates[i-1] for i in bkps[:-1]]
print("变点对应的日期:", cp_dates)

# =============================
# 4. 在“最后一个 regime”上用 BLR 拟合（替换 ARIMA）
# =============================
# 最后一个 regime 起点：
last_cp_idx = 0 if len(bkps) == 1 else bkps[-2]
regime_returns = returns.iloc[last_cp_idx:]

print("最后一个 regime 样本区间：", regime_returns.index[0], " ~ ", regime_returns.index[-1])

# 简单拆分训练 / 测试集（例如最后 50 个点做测试）
test_size = 50
if len(regime_returns) <= test_size + 5:
    raise ValueError("最后一个regime样本太短，请减小 test_size 或调整 CP 设置。")

train_returns = regime_returns.iloc[:-test_size]
test_returns = regime_returns.iloc[-test_size:]

# ========== 4.1 选择 AR 阶数 p（用 BLR + 验证集 MSE） ==========
max_lag = 6  # 你可以调大一点，但月频数据不多，别太大
candidate_lags = range(1, max_lag + 1)

best_lag = None
best_mse = np.inf

for lag in candidate_lags:
    try:
        X, y = make_lagged_matrix(train_returns, lag)
    except ValueError:
        continue

    # 简单切一段作为验证集（最后 20%）
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    if len(X_val) == 0:
        continue

    blr = BayesianRidge()
    blr.fit(X_train, y_train)
    y_pred = blr.predict(X_val)
    mse = np.mean((y_val - y_pred) ** 2)

    if mse < best_mse:
        best_mse = mse
        best_lag = lag

print("选出的最优 AR 滞后阶数 p =", best_lag, "  验证集 MSE =", best_mse)

# ========== 4.2 用最优 lag 在整个 train_returns 上拟合 BLR ==========
X_full, y_full = make_lagged_matrix(train_returns, best_lag)
blr_final = BayesianRidge()
blr_final.fit(X_full, y_full)

# =============================
# 5. 用 BLR 预测，并构造简单交易信号
# =============================

# 为了模拟 multi-step 预测，像 ARIMA 一样递推预测 test_size 期
history = train_returns.copy()
forecast_vals = []

for t in range(len(test_returns)):
    # 1. 用当前历史数据构造 lag 特征
    X_hist, y_hist = make_lagged_matrix(history, best_lag)

    blr_step = BayesianRidge()
    blr_step.fit(X_hist, y_hist)

    # 2. 预测下一个点（也就是当前这个 test_returns.iloc[t]）
    last_lags = np.array(history.iloc[-best_lag:])
    X_new = last_lags.reshape(1, -1)
    y_hat = blr_step.predict(X_new)[0]

    forecast_vals.append(y_hat)

    # 3. 用“真实的” test_returns 来更新 history（重点是这里）
    new_point = pd.Series(
        [test_returns.iloc[t]],
        index=[test_returns.index[t]]
    )
    history = pd.concat([history, new_point])

forecast = pd.Series(forecast_vals, index=test_returns.index)


# 简单 trading rule：
signals = (forecast > 0).astype(int)   # >0 做多，否则空仓
strategy_returns = test_returns * signals
equity_curve = (1 + strategy_returns).cumprod()

# =============================
# 6. 可视化预测 & 策略表现（BLR+CP）
# =============================
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# (1) 预测 vs 实际收益
ax[0].plot(test_returns.index, test_returns, label='Real Returns')
ax[0].plot(forecast.index, forecast, label='BLR Forecast Returns', linestyle='--')
ax[0].axhline(0, linewidth=0.8)
ax[0].set_title('BLR Forecast vs Real Returns (Last Regime)')
ax[0].legend()

# (2) 策略净值曲线
ax[1].plot(equity_curve.index, equity_curve, label='BLR+CP Strategy Equity Curve')
ax[1].set_title('BLR+CP Trading Strategy Performance')
ax[1].legend()

plt.tight_layout()
plt.show()
