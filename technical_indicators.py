#!/usr/bin/env python3
"""
Technical Indicators Module
Comprehensive collection of technical indicators for trading strategy development.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Comprehensive technical indicators for trading analysis"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = TechnicalIndicators.ema(macd_line, signal)
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K and %D)"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        length = len(high)
        psar = np.zeros(length)
        uptrend = np.zeros(length, dtype=bool)
        af = np.zeros(length)
        ep = np.zeros(length)
        
        # Initialize
        psar[0] = low.iloc[0]
        uptrend[0] = True
        af[0] = acceleration
        ep[0] = high.iloc[0]
        
        for i in range(1, length):
            if uptrend[i-1]:
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                if low.iloc[i] <= psar[i]:
                    uptrend[i] = False
                    psar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af[i] = acceleration
                else:
                    uptrend[i] = True
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                if high.iloc[i] >= psar[i]:
                    uptrend[i] = True
                    psar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af[i] = acceleration
                else:
                    uptrend[i] = False
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(psar, index=high.index)
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        """Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = ((high.rolling(window=9).max() + low.rolling(window=9).min()) / 2)
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = ((high.rolling(window=26).max() + low.rolling(window=26).min()) / 2)
        
        # Senkou Span A: (Conversion Line + Base Line)/2, plotted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B: (52-period high + 52-period low)/2, plotted 26 periods ahead
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span: Close price plotted 26 periods back
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index (ADX) with +DI and -DI"""
        # Calculate True Range
        atr_values = TechnicalIndicators.atr(high, low, close, window)
        
        # Calculate directional movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                           np.maximum(low.shift(1) - low, 0), 0)
        
        dm_plus = pd.Series(dm_plus, index=high.index)
        dm_minus = pd.Series(dm_minus, index=high.index)
        
        # Calculate smoothed directional movement
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        
        # Calculate directional indicators
        di_plus = 100 * (dm_plus_smooth / atr_values)
        di_minus = 100 * (dm_minus_smooth / atr_values)
        
        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return adx, di_plus, di_minus
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def momentum(data: pd.Series, window: int = 10) -> pd.Series:
        """Price Momentum"""
        return data.diff(window)
    
    @staticmethod
    def rate_of_change(data: pd.Series, window: int = 10) -> pd.Series:
        """Rate of Change (ROC)"""
        return ((data - data.shift(window)) / data.shift(window)) * 100
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv_values = np.where(close > close.shift(1), volume,
                             np.where(close < close.shift(1), -volume, 0))
        return pd.Series(obv_values, index=close.index).cumsum()
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index (MFI)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_flow_sum = pd.Series(positive_flow, index=close.index).rolling(window=window).sum()
        negative_flow_sum = pd.Series(negative_flow, index=close.index).rolling(window=window).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        return 100 - (100 / (1 + money_ratio))
    
    @staticmethod
    def hull_moving_average(data: pd.Series, window: int) -> pd.Series:
        """Hull Moving Average"""
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        
        wma_half = data.rolling(window=half_window).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
        wma_full = data.rolling(window=window).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
        
        raw_hma = 2 * wma_half - wma_full
        return raw_hma.rolling(window=sqrt_window).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)


def demo_technical_indicators():
    """Demonstration of technical indicators"""
    print("=" * 60)
    print("TECHNICAL INDICATORS DEMO")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic OHLCV data
    close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
    high = close + np.abs(np.random.randn(100) * 0.5)
    low = close - np.abs(np.random.randn(100) * 0.5)
    volume = pd.Series(np.random.randint(1000, 10000, 100), index=dates)
    
    # Initialize indicators
    ti = TechnicalIndicators()
    
    # Calculate indicators
    rsi = ti.rsi(close)
    macd_line, macd_signal, macd_hist = ti.macd(close)
    bb_upper, bb_middle, bb_lower = ti.bollinger_bands(close)
    stoch_k, stoch_d = ti.stochastic_oscillator(high, low, close)
    williams = ti.williams_r(high, low, close)
    atr = ti.atr(high, low, close)
    cci = ti.cci(high, low, close)
    adx, di_plus, di_minus = ti.adx(high, low, close)
    
    # Display results
    print(f"RSI (last 5 values): {rsi.tail().round(2).tolist()}")
    print(f"MACD (last 5 values): {macd_line.tail().round(4).tolist()}")
    print(f"Bollinger Upper (last 5): {bb_upper.tail().round(2).tolist()}")
    print(f"Stochastic %K (last 5): {stoch_k.tail().round(2).tolist()}")
    print(f"Williams %R (last 5): {williams.tail().round(2).tolist()}")
    print(f"ATR (last 5 values): {atr.tail().round(2).tolist()}")
    print(f"CCI (last 5 values): {cci.tail().round(2).tolist()}")
    print(f"ADX (last 5 values): {adx.tail().round(2).tolist()}")
    
    print("\n✓ All technical indicators calculated successfully!")


if __name__ == "__main__":
    demo_technical_indicators()
