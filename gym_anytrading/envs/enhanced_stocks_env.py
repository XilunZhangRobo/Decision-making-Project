import os

import numpy as np
import pandas as pd
import gymnasium as gym
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
        window_size: int = 30,  # Match baseline window size
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

        super().__init__(df, window_size, render_mode)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
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
        # Simple, proven features that actually help trading decisions:

        # 1. Price and momentum (what we had before)
        # 2. Add just ONE key trend indicator: short-term trend direction
        sma_10 = price_series.rolling(10).mean()
        sma_30 = price_series.rolling(30).mean()
        # Simple trend: positive when price > moving average (bullish)
        trend_direction = np.sign(price_series - sma_30).fillna(0)

        # Keep it simple: price, returns, mean-reversion, and trend
        base_features = np.column_stack([
            prices_slice / 1000.0,          # Normalized price
            np.clip(log_ret.to_numpy()[start:end], -0.05, 0.05),  # Clipped returns
            rsi.to_numpy()[start:end] / 100.0,  # Normalized RSI (0-1)
            trend_direction.to_numpy()[start:end],  # Trend direction (-1, 0, 1)
        ])

        # Simplified: remove complex BLR features that might be noisy
        # Just use basic technical indicators
        feature_slice = base_features

        return prices_slice.astype(np.float32), feature_slice.astype(np.float32)

    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def _calculate_reward(self, action):
        """
        Sharpe-ratio optimized reward: maximize returns while minimizing volatility.
        This should lead to more consistent, risk-adjusted performance.
        """
        step_reward = 0

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                # Sharpe-like reward: returns minus volatility penalty
                # This encourages steady gains over volatile swings
                returns = price_diff / last_trade_price  # Percentage return

                # Volatility penalty based on recent price swings
                recent_prices = self.prices[max(0, self._current_tick-10):self._current_tick+1]
                if len(recent_prices) > 1:
                    volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
                    volatility_penalty = volatility * 100.0
                else:
                    volatility_penalty = 0

                # Reward consistent performance, penalize volatility
                step_reward += (returns * 1000.0) - volatility_penalty

        return step_reward

    def step(self, action):
        self._truncated = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        # early stop if bankrupt
        if self._total_profit is not None and self._total_profit <= self._profit_floor:
            self._truncated = True

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit

