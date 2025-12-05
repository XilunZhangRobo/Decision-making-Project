import os

import numpy as np
import pandas as pd
from gymnasium import spaces

from gym_anytrading.features.blr_cp_features import compute_blr_cp_features
from .trading_env import TradingEnv, Actions, Positions


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with minimal dependencies."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral when undefined


class EnhancedStocksEnv(TradingEnv):
    """
    A higher-signal variant of StocksEnv with:
    - Richer observation space (technicals and volatility)
    - Symmetric reward (long/short) using percentage returns
    - Trade-cost penalty to discourage churn
    """

    def __init__(
        self,
        df,
        window_size: int = 60,
        frame_bound=None,
        render_mode=None,
        trade_fee_bid_percent: float = 0.001,
        trade_fee_ask_percent: float = 0.001,
        reward_trade_penalty: float = 0.0001,
        vol_window: int = 20,
    ):
        assert len(df) > window_size

        if frame_bound is None:
            frame_bound = (window_size, len(df))

        self.frame_bound = frame_bound
        self.trade_fee_bid_percent = trade_fee_bid_percent
        self.trade_fee_ask_percent = trade_fee_ask_percent
        self.reward_trade_penalty = reward_trade_penalty
        self.vol_window = vol_window

        # continuous action: target position weight in [-1, 1] (short to long)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        super().__init__(df, window_size, render_mode)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._position_weight = 0.0
        return obs, info

    def _process_data(self):
        df = self.df.copy()

        price_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(np.ones_like(price_series), index=price_series.index)

        log_ret = np.log(price_series / price_series.shift(1)).fillna(0)
        vol_win = log_ret.rolling(self.vol_window).std().fillna(0)
        sma_fast = price_series.rolling(10).mean()
        sma_slow = price_series.rolling(30).mean()
        sma_ratio = ((sma_fast - sma_slow) / sma_slow).fillna(0)
        rsi = _compute_rsi(price_series, period=14)
        vol_norm = (volume / volume.rolling(20).mean().replace(0, np.nan)).fillna(1) - 1

        prices = price_series.to_numpy()

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        prices_slice = prices[start:end]
        base_features = np.column_stack(
            [
                prices_slice,
                log_ret.to_numpy()[start:end],
                sma_ratio.to_numpy()[start:end],
                vol_win.to_numpy()[start:end],
                rsi.to_numpy()[start:end],
                vol_norm.to_numpy()[start:end],
            ]
        )

        # add model-based features (BLR forecast, regime_id, days_since_cp), cached if available
        cache_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "datasets", "data", "STOCKS_GOOGL_blr_features.csv")
        )
        extra = compute_blr_cp_features(df, cache_path=cache_path, show_progress=True)
        extra_slice = extra.iloc[start:end].to_numpy()

        feature_slice = np.column_stack([base_features, extra_slice])

        return prices_slice.astype(np.float32), feature_slice.astype(np.float32)

    def _update_profit(self, price_change, target_weight):
        """
        Mark-to-market equity with continuous position sizing.
        Fees applied on change in weight magnitude.
        """
        # price change applied to prior weight
        gross_factor = 1.0 + self._position_weight * price_change

        # fees on turnover
        turnover = abs(target_weight - self._position_weight)
        fee_factor = 1.0 - turnover * (self.trade_fee_bid_percent + self.trade_fee_ask_percent)

        new_profit = self._total_profit * max(gross_factor * fee_factor, self._profit_floor)
        self._position_weight = float(target_weight)
        return new_profit

    def step(self, action):
        self._truncated = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._truncated = True

        # clip and parse continuous action to target weight [-1, 1]
        arr = np.asarray(action).reshape(-1)
        target_weight = float(np.clip(arr[0], -1.0, 1.0))

        # price change since last tick
        prev_price = self.prices[self._current_tick - 1]
        price = self.prices[self._current_tick]
        price_change = (price - prev_price) / prev_price

        prev_profit = self._total_profit
        prev_weight = self._position_weight
        turnover = abs(target_weight - prev_weight)

        # update profit using prior weight and turnover fees; then set new weight
        new_profit = self._update_profit(price_change, target_weight)
        reward = float(new_profit - prev_profit) * 100.0  # scale reward for better learning
        if turnover > 0:
            reward -= self.reward_trade_penalty  # discourage churn

        self._total_profit = new_profit
        self._total_reward += reward

        # early stop if bankrupt
        if self._total_profit is not None and self._total_profit <= self._profit_floor:
            self._truncated = True

        # track pseudo position enum for rendering/history
        self._position = Positions.Long if self._position_weight >= 0 else Positions.Short
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, False, self._truncated, info

    def max_possible_profit(self):
        # Approximate using perfect foresight on direction with costs applied
        profit = 1.0
        for i in range(self._start_tick + 1, self._end_tick + 1):
            price_change = (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1]
            profit *= 1.0 + abs(price_change) * (1 - self.trade_fee_bid_percent - self.trade_fee_ask_percent)
        return profit

