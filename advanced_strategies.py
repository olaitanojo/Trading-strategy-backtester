#!/usr/bin/env python3
"""
Advanced Trading Strategies Module
Sophisticated trading strategies using multiple technical indicators and advanced algorithms.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from backtester import TradingStrategy
from technical_indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')


class MeanReversionBollingerStrategy(TradingStrategy):
    """Enhanced Mean Reversion using Bollinger Bands with RSI confirmation"""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2, rsi_period: int = 14,
                 rsi_oversold: float = 30, rsi_overbought: float = 70):
        super().__init__(f"MeanRevBB_{bb_period}_{bb_std}_RSI_{rsi_period}")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using Bollinger Bands + RSI confirmation"""
        close = data['Close']
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(close, self.bb_period, self.bb_std)
        rsi = self.ti.rsi(close, self.rsi_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signals: Price touches lower band AND RSI is oversold
        buy_condition = (close <= bb_lower) & (rsi <= self.rsi_oversold)
        signals[buy_condition] = 1
        
        # Sell signals: Price touches upper band AND RSI is overbought
        sell_condition = (close >= bb_upper) & (rsi >= self.rsi_overbought)
        signals[sell_condition] = -1
        
        return signals


class MomentumStrategy(TradingStrategy):
    """Multi-timeframe momentum strategy with ADX filter"""
    
    def __init__(self, short_period: int = 10, long_period: int = 30, 
                 adx_period: int = 14, adx_threshold: float = 25):
        super().__init__(f"Momentum_{short_period}_{long_period}_ADX_{adx_threshold}")
        self.short_period = short_period
        self.long_period = long_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals with trend strength filter"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate momentum indicators
        short_momentum = self.ti.rate_of_change(close, self.short_period)
        long_momentum = self.ti.rate_of_change(close, self.long_period)
        
        # Calculate ADX for trend strength
        adx, di_plus, di_minus = self.ti.adx(high, low, close, self.adx_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Strong trend filter
        strong_trend = adx > self.adx_threshold
        
        # Buy: Positive momentum in both timeframes with strong trend
        buy_condition = (short_momentum > 0) & (long_momentum > 0) & strong_trend & (di_plus > di_minus)
        signals[buy_condition] = 1
        
        # Sell: Negative momentum in both timeframes with strong trend
        sell_condition = (short_momentum < 0) & (long_momentum < 0) & strong_trend & (di_minus > di_plus)
        signals[sell_condition] = -1
        
        return signals


class StochasticRSIStrategy(TradingStrategy):
    """Combined Stochastic and RSI strategy with divergence detection"""
    
    def __init__(self, stoch_k: int = 14, stoch_d: int = 3, rsi_period: int = 14,
                 stoch_oversold: float = 20, stoch_overbought: float = 80):
        super().__init__(f"StochRSI_{stoch_k}_{rsi_period}")
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.rsi_period = rsi_period
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using Stochastic + RSI combination"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate indicators
        stoch_k_val, stoch_d_val = self.ti.stochastic_oscillator(high, low, close, self.stoch_k, self.stoch_d)
        rsi = self.ti.rsi(close, self.rsi_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: Both oscillators in oversold region
        buy_condition = (stoch_k_val <= self.stoch_oversold) & (rsi <= 30)
        signals[buy_condition] = 1
        
        # Sell: Both oscillators in overbought region
        sell_condition = (stoch_k_val >= self.stoch_overbought) & (rsi >= 70)
        signals[sell_condition] = -1
        
        return signals


class IchimokuStrategy(TradingStrategy):
    """Ichimoku Cloud strategy with multiple confirmation signals"""
    
    def __init__(self):
        super().__init__("Ichimoku_Cloud")
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Ichimoku Cloud analysis"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate Ichimoku components
        ichimoku = self.ti.ichimoku_cloud(high, low, close)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signals: Multiple bullish confirmations
        price_above_cloud = close > ichimoku['senkou_span_a'].shift(-26)
        price_above_cloud_b = close > ichimoku['senkou_span_b'].shift(-26)
        tenkan_above_kijun = ichimoku['tenkan_sen'] > ichimoku['kijun_sen']
        chikou_above_price = ichimoku['chikou_span'] > close.shift(26)
        
        buy_condition = price_above_cloud & price_above_cloud_b & tenkan_above_kijun & chikou_above_price
        signals[buy_condition] = 1
        
        # Sell signals: Multiple bearish confirmations
        price_below_cloud = close < ichimoku['senkou_span_a'].shift(-26)
        price_below_cloud_b = close < ichimoku['senkou_span_b'].shift(-26)
        tenkan_below_kijun = ichimoku['tenkan_sen'] < ichimoku['kijun_sen']
        chikou_below_price = ichimoku['chikou_span'] < close.shift(26)
        
        sell_condition = price_below_cloud & price_below_cloud_b & tenkan_below_kijun & chikou_below_price
        signals[sell_condition] = -1
        
        return signals


class ParabolicSARMACDStrategy(TradingStrategy):
    """Parabolic SAR with MACD confirmation strategy"""
    
    def __init__(self, psar_accel: float = 0.02, psar_max: float = 0.2,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        super().__init__(f"PSAR_MACD_{psar_accel}_{macd_fast}_{macd_slow}")
        self.psar_accel = psar_accel
        self.psar_max = psar_max
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using PSAR trend + MACD momentum"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate indicators
        psar = self.ti.parabolic_sar(high, low, self.psar_accel, self.psar_max)
        macd_line, macd_signal_line, macd_histogram = self.ti.macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: Price above PSAR AND MACD bullish crossover
        psar_bullish = close > psar
        macd_bullish = macd_line > macd_signal_line
        
        buy_condition = psar_bullish & macd_bullish
        signals[buy_condition] = 1
        
        # Sell: Price below PSAR AND MACD bearish crossover
        psar_bearish = close < psar
        macd_bearish = macd_line < macd_signal_line
        
        sell_condition = psar_bearish & macd_bearish
        signals[sell_condition] = -1
        
        return signals


class VolatilityBreakoutStrategy(TradingStrategy):
    """Volatility breakout strategy using ATR and Bollinger Bands"""
    
    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0,
                 bb_period: int = 20, bb_std: float = 2.0):
        super().__init__(f"VolBreakout_ATR_{atr_period}_BB_{bb_period}")
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate breakout signals based on volatility measures"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate volatility indicators
        atr = self.ti.atr(high, low, close, self.atr_period)
        bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(close, self.bb_period, self.bb_std)
        
        # Dynamic levels based on ATR
        upper_breakout = close.shift(1) + (atr * self.atr_multiplier)
        lower_breakout = close.shift(1) - (atr * self.atr_multiplier)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: Price breaks above upper level AND above Bollinger upper band
        buy_condition = (close > upper_breakout) & (close > bb_upper)
        signals[buy_condition] = 1
        
        # Sell: Price breaks below lower level AND below Bollinger lower band
        sell_condition = (close < lower_breakout) & (close < bb_lower)
        signals[sell_condition] = -1
        
        return signals


class MultiIndicatorStrategy(TradingStrategy):
    """Sophisticated multi-indicator strategy with scoring system"""
    
    def __init__(self, ma_fast: int = 12, ma_slow: int = 26, rsi_period: int = 14,
                 bb_period: int = 20, adx_period: int = 14):
        super().__init__(f"MultiIndicator_{ma_fast}_{ma_slow}_{rsi_period}")
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.adx_period = adx_period
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using weighted scoring from multiple indicators"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate all indicators
        ma_fast = self.ti.ema(close, self.ma_fast)
        ma_slow = self.ti.ema(close, self.ma_slow)
        rsi = self.ti.rsi(close, self.rsi_period)
        bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(close, self.bb_period)
        adx, di_plus, di_minus = self.ti.adx(high, low, close, self.adx_period)
        macd_line, macd_signal_line, _ = self.ti.macd(close)
        
        # Create scoring system
        scores = pd.DataFrame(index=data.index)
        
        # Trend indicators (40% weight)
        scores['ma_trend'] = np.where(ma_fast > ma_slow, 1, -1) * 0.2
        scores['macd_trend'] = np.where(macd_line > macd_signal_line, 1, -1) * 0.2
        
        # Momentum indicators (30% weight)
        scores['rsi_momentum'] = np.where(rsi > 50, 1, -1) * 0.15
        scores['adx_direction'] = np.where(di_plus > di_minus, 1, -1) * 0.15
        
        # Mean reversion indicators (20% weight)
        scores['bb_position'] = np.where(close > bb_middle, 1, -1) * 0.2
        
        # Strength filter (10% weight)
        scores['trend_strength'] = np.where(adx > 25, 1, 0) * 0.1
        
        # Calculate total score
        total_score = scores.sum(axis=1)
        
        signals = pd.Series(0, index=data.index)
        signals[total_score > 0.6] = 1   # Strong bullish consensus
        signals[total_score < -0.6] = -1  # Strong bearish consensus
        
        return signals


class CommodityChannelStrategy(TradingStrategy):
    """CCI-based strategy with Williams %R confirmation"""
    
    def __init__(self, cci_period: int = 20, williams_period: int = 14,
                 cci_oversold: float = -100, cci_overbought: float = 100):
        super().__init__(f"CCI_Williams_{cci_period}_{williams_period}")
        self.cci_period = cci_period
        self.williams_period = williams_period
        self.cci_oversold = cci_oversold
        self.cci_overbought = cci_overbought
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using CCI and Williams %R"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate indicators
        cci = self.ti.cci(high, low, close, self.cci_period)
        williams = self.ti.williams_r(high, low, close, self.williams_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: CCI oversold AND Williams %R oversold (< -80)
        buy_condition = (cci < self.cci_oversold) & (williams < -80)
        signals[buy_condition] = 1
        
        # Sell: CCI overbought AND Williams %R overbought (> -20)
        sell_condition = (cci > self.cci_overbought) & (williams > -20)
        signals[sell_condition] = -1
        
        return signals


class HullMomentumStrategy(TradingStrategy):
    """Hull Moving Average with momentum confirmation"""
    
    def __init__(self, hull_period: int = 21, momentum_period: int = 10):
        super().__init__(f"Hull_Momentum_{hull_period}_{momentum_period}")
        self.hull_period = hull_period
        self.momentum_period = momentum_period
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using Hull MA trend and momentum"""
        close = data['Close']
        
        # Calculate Hull Moving Average
        hma = self.ti.hull_moving_average(close, self.hull_period)
        
        # Calculate momentum
        momentum = self.ti.momentum(close, self.momentum_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: Price above Hull MA AND positive momentum
        buy_condition = (close > hma) & (momentum > 0)
        signals[buy_condition] = 1
        
        # Sell: Price below Hull MA AND negative momentum
        sell_condition = (close < hma) & (momentum < 0)
        signals[sell_condition] = -1
        
        return signals


def demo_advanced_strategies():
    """Demonstration of advanced trading strategies"""
    print("=" * 60)
    print("ADVANCED TRADING STRATEGIES DEMO")
    print("=" * 60)
    
    # Generate sample OHLC data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    price = 100
    ohlc_data = []
    
    for _ in range(252):
        change = np.random.randn() * 0.02
        price = price * (1 + change)
        
        high = price * (1 + abs(np.random.randn()) * 0.01)
        low = price * (1 - abs(np.random.randn()) * 0.01)
        
        ohlc_data.append({
            'Open': price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': np.random.randint(1000, 10000)
        })
    
    data = pd.DataFrame(ohlc_data, index=dates)
    
    # Test strategies
    strategies = [
        MeanReversionBollingerStrategy(),
        MomentumStrategy(),
        StochasticRSIStrategy(),
        IchimokuStrategy(),
        ParabolicSARMACDStrategy(),
        VolatilityBreakoutStrategy(),
        MultiIndicatorStrategy(),
        CommodityChannelStrategy(),
        HullMomentumStrategy()
    ]
    
    print(f"\nTesting {len(strategies)} advanced strategies on {len(data)} days of data...\n")
    
    for strategy in strategies:
        try:
            signals = strategy.generate_signals(data)
            signal_count = len(signals[signals != 0])
            buy_signals = len(signals[signals == 1])
            sell_signals = len(signals[signals == -1])
            
            print(f"✓ {strategy.name}:")
            print(f"  Total Signals: {signal_count}")
            print(f"  Buy Signals: {buy_signals}")
            print(f"  Sell Signals: {sell_signals}")
            print()
            
        except Exception as e:
            print(f"✗ {strategy.name}: Error - {str(e)}")
            print()
    
    print("Advanced strategies demonstration completed!")


if __name__ == "__main__":
    demo_advanced_strategies()
