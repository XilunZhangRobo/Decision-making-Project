import os
from functools import lru_cache
from typing import Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.linear_model import BayesianRidge

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(iterable: Iterable, desc: str, enabled: bool):
    """Lightweight progress wrapper that falls back gracefully if tqdm is absent."""
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc)
    return iterable


def _auto_cp_by_pen_soft(signal: np.ndarray, model: str = "rbf", pen_grid=None, max_cps: int = 6) -> Tuple[list, float, int]:
    """Run PELT with a small penalty grid and pick lowest cost with <= max_cps."""
    n = len(signal)
    if pen_grid is None:
        base = np.log(n)
        pen_grid = [0.2 * base, 0.5 * base, 1 * base, 2 * base, 3 * base]

    best_cost = np.inf
    best_bkps = None
    best_pen = None
    best_k = None

    for pen in pen_grid:
        algo = rpt.Pelt(model=model).fit(signal)
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
    if best_bkps is None:
        best_bkps = [len(signal)]
    return best_bkps, best_pen, best_k


def _make_lagged_matrix(series: pd.Series, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series)
    n = len(values)
    if n <= lag:
        raise ValueError("Series too short for lag {}".format(lag))
    X, y = [], []
    for t in range(lag, n):
        X.append(values[t - lag : t])
        y.append(values[t])
    return np.array(X), np.array(y)


@lru_cache(maxsize=4)
def _compute_blr_cp_features_cached(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    return compute_blr_cp_features(df)


def compute_blr_cp_features(df: pd.DataFrame, cache_path: Optional[str] = None, show_progress: bool = False) -> pd.DataFrame:
    """
    Compute BLR forecast/regime features on (adjusted) close prices.
    Returns a DataFrame indexed like the input with columns:
    ['blr_forecast', 'regime_id', 'days_since_cp']

    If cache_path is provided and exists, it is loaded; otherwise compute and
    optionally save to cache_path.
    """
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, parse_dates=True, index_col=0)
        return cached

    price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    price = price.astype(float).dropna().sort_index()
    price = price[~price.index.duplicated(keep="last")]
    price = price.asfreq("B").ffill()

    monthly_price = price.resample("M").last()
    log_price = np.log(monthly_price)
    returns = log_price.diff().dropna()

    # change-point detection on monthly returns
    signal = returns.values.reshape(-1, 1)
    bkps, _, _ = _auto_cp_by_pen_soft(signal, model="rbf", max_cps=6)

    # regime labeling on monthly index
    regime_ids = np.zeros(len(returns), dtype=int)
    last = 0
    for idx, bkp in enumerate(bkps):
        regime_ids[last:bkp] = idx
        last = bkp
    regime_series = pd.Series(regime_ids, index=returns.index)

    # focus on last regime for BLR
    last_cp_idx = 0 if len(bkps) == 1 else bkps[-2]
    regime_returns = returns.iloc[last_cp_idx:]

    # choose lag via simple validation
    best_lag, best_mse = None, np.inf
    for lag in _progress(range(1, 7), desc="Selecting lag (BLR)", enabled=show_progress):
        try:
            X, y = _make_lagged_matrix(regime_returns, lag)
        except ValueError:
            continue
        split = int(len(X) * 0.8)
        if split == 0 or split >= len(X):
            continue
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        blr = BayesianRidge()
        blr.fit(X_train, y_train)
        mse = np.mean((y_val - blr.predict(X_val)) ** 2)
        if mse < best_mse:
            best_mse, best_lag = mse, lag
    if best_lag is None:
        best_lag = 3

    # recursive one-step forecasts over the last regime
    history = regime_returns.copy()
    forecasts = []
    for _ in _progress(range(len(regime_returns)), desc="Forecasting BLR", enabled=show_progress):
        X_hist, y_hist = _make_lagged_matrix(history, best_lag)
        blr_step = BayesianRidge().fit(X_hist, y_hist)
        last_lags = np.array(history.iloc[-best_lag:])
        y_hat = blr_step.predict(last_lags.reshape(1, -1))[0]
        forecasts.append(y_hat)
        history = pd.concat([history, pd.Series([history.iloc[len(forecasts)-1]], index=[history.index[len(forecasts)-1]])])

    forecast_series = pd.Series(forecasts, index=regime_returns.index)

    # align features back to daily price index
    feature_monthly = pd.DataFrame(
        {
            "blr_forecast": forecast_series.reindex(returns.index).ffill().fillna(0),
            "regime_id": regime_series,
        }
    )
    # days since last CP on monthly grid
    cp_positions = [returns.index[b - 1] for b in bkps if b <= len(returns)]
    last_cp_date = pd.Series(index=returns.index, dtype="datetime64[ns]")
    current = None
    for dt in returns.index:
        if dt in cp_positions:
            current = dt
        last_cp_date.loc[dt] = current
    # convert to Series so we can safely access the timedelta days
    days_since_cp = (feature_monthly.index.to_series() - last_cp_date).dt.days
    feature_monthly["days_since_cp"] = days_since_cp.fillna(0)

    features_daily = feature_monthly.reindex(price.index, method="ffill").fillna(0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        features_daily.to_csv(cache_path)

    return features_daily

