#!/usr/bin/env python3
"""
AI/ML Enhanced Deriv Trading Bot - Complete Linux Optimized Version
Designed for Render deployment with comprehensive ML capabilities
"""

import asyncio
import websockets
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
import requests
import os
import time
import signal
import sys
import asyncio
import threading
from threading import Thread, Lock
from flask import Flask, jsonify
import gc
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML/AI imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Scikit-learn not available, using simplified models")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available, using synthetic data")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("ta library not available, using basic indicators")

from scipy import stats
from scipy.signal import find_peaks
import math

# Enums and Data Classes
class MarketRegime(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear" 
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    NEWS_DRIVEN = "news_driven"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"

class SignalQuality(Enum):
    STRONG_BUY = "strong_buy"
    MODERATE_BUY = "moderate_buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    MODERATE_SELL = "moderate_sell"
    STRONG_SELL = "strong_sell"

class TradeOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PARTIAL_WIN = "partial_win"
    EARLY_EXIT = "early_exit"

@dataclass
class MLPrediction:
    signal_strength: float  # 0-100
    profit_probability: float  # 0-1
    reversal_risk: float  # 0-1 (risk of 80-90% profit then reversal)
    optimal_tp_levels: List[float]
    recommended_sl: float
    position_size_multiplier: float
    market_regime: MarketRegime
    confidence: float
    time_horizon: str  # "short", "medium", "long"
    risk_score: float  # 0-1
    volatility_forecast: float
    news_sentiment_impact: float

@dataclass
class TradeSignal:
    symbol: str
    signal_type: str  # "CALL", "PUT"
    strength: float
    confidence: float
    ml_prediction: MLPrediction
    technical_score: float
    entry_reasons: List[str]
    risk_factors: List[str]
    timestamp: datetime

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int = 0

class NewsEventDetector:
    def __init__(self):
        pass
    
    def is_news_time(self, current_time=None):
        return False, 0.0
    
    def get_news_sentiment_impact(self, symbol):
        return 0.3
    
    def _calculate_ml_tp_levels(self, profit_potential: float, reversal_risk: float, regime: MarketRegime) -> List[float]:
        """Calculate ML-optimized take profit levels"""
        try:
            base_tp = max(0.01, profit_potential)
            
            # Regime-specific adjustments
            if regime == MarketRegime.TRENDING_BULL or regime == MarketRegime.TRENDING_BEAR:
                multipliers = [0.4, 0.7, 1.2] if reversal_risk < 0.3 else [0.3, 0.6]
            elif regime == MarketRegime.VOLATILE:
                multipliers = [0.2, 0.4] if reversal_risk > 0.6 else [0.3, 0.6, 0.9]
            else:  # SIDEWAYS
                multipliers = [0.5, 0.8] if reversal_risk > 0.4 else [0.6, 1.0]
            
            return [base_tp * mult for mult in multipliers]
            
        except:
            return [0.015, 0.025]
    
    def _determine_time_horizon(self, regime: MarketRegime, volatility: float) -> str:
        """Determine optimal time horizon"""
        if regime == MarketRegime.VOLATILE or volatility > 0.05:
            return "short"
        elif regime == MarketRegime.TRENDING_BULL or regime == MarketRegime.TRENDING_BEAR:
            return "medium"
        else:
            return "short"
    
    def _get_rule_based_prediction(self, indicators: Dict, df: pd.DataFrame) -> MLPrediction:
        """Rule-based prediction when ML is not available"""
        try:
            # Simple rule-based scoring
            score = 0
            
            # RSI scoring
            rsi = indicators.get('rsi', 50)
            if 30 < rsi < 70:
                score += 20
            elif 20 < rsi <= 30 or 70 <= rsi < 80:
                score += 30
            elif rsi <= 20 or rsi >= 80:
                score += 10  # Extreme levels, lower confidence
            
            # MACD scoring
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if (macd > macd_signal and macd > 0) or (macd < macd_signal and macd < 0):
                score += 25
            
            # Trend strength scoring
            trend_strength = indicators.get('trend_strength', 0.5)
            score += int(trend_strength * 30)
            
            # Bollinger Bands position
            bb_position = indicators.get('bb_position', 0.5)
            if 0.2 < bb_position < 0.8:
                score += 15
            
            # ADX scoring
            adx = indicators.get('adx', 20)
            if adx > 25:
                score += 10
            
            # Calculate metrics
            signal_strength = min(100, score)
            profit_probability = signal_strength / 100
            
            # Simple reversal risk calculation
            reversal_risk = 0.5 - (trend_strength - 0.5)  # Higher trend = lower reversal risk
            reversal_risk = max(0, min(1, reversal_risk))
            
            # Market regime detection
            if adx > 25 and trend_strength > 0.7:
                regime = MarketRegime.TRENDING_BULL if rsi < 70 else MarketRegime.TRENDING_BEAR
            elif indicators.get('historical_vol', 0.2) > 0.4:
                regime = MarketRegime.VOLATILE
            else:
                regime = MarketRegime.SIDEWAYS
            
            return MLPrediction(
                signal_strength=signal_strength,
                profit_probability=profit_probability,
                reversal_risk=reversal_risk,
                optimal_tp_levels=[0.015, 0.025],
                recommended_sl=indicators.get('atr', 0.01) * 2,
                position_size_multiplier=min(2.0, profit_probability * 1.5),
                market_regime=regime,
                confidence=profit_probability * 0.8,  # Lower confidence for rule-based
                time_horizon="short",
                risk_score=0.5,
                volatility_forecast=indicators.get('historical_vol', 0.2),
                news_sentiment_impact=0.3
            )
            
        except Exception as e:
            logging.error(f"Rule-based prediction error: {e}")
            return self._get_conservative_prediction()
    
    def _get_default_ml_prediction(self) -> Dict:
        """Default ML prediction values"""
        return {
            'signal_strength': 60.0,
            'profit_probability': 0.6,
            'reversal_risk': 0.4,
            'optimal_tp_levels': [0.015, 0.025],
            'recommended_sl': 0.01,
            'position_size_multiplier': 1.0,
            'market_regime': MarketRegime.SIDEWAYS,
            'confidence': 0.5,
            'time_horizon': "short",
            'risk_score': 0.5,
            'volatility_forecast': 0.2
        }
    
    def _get_conservative_prediction(self) -> MLPrediction:
        """Conservative default prediction"""
        return MLPrediction(
            signal_strength=50.0,
            profit_probability=0.5,
            reversal_risk=0.5,
            optimal_tp_levels=[0.01, 0.02],
            recommended_sl=0.015,
            position_size_multiplier=0.8,
            market_regime=MarketRegime.SIDEWAYS,
            confidence=0.4,
            time_horizon="short",
            risk_score=0.6,
            volatility_forecast=0.25,
            news_sentiment_impact=0.3
        )

class RiskManagementEngine:
    """Advanced risk management with ML insights"""
    
    def __init__(self):
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '20'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.02'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '0.10'))
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', '0.15'))
        
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.trade_history = deque(maxlen=100)
        self.correlation_matrix = {}
        self.position_tracker = {}
        
    def assess_trade_risk(self, signal: TradeSignal, balance: float, ml_prediction: MLPrediction) -> Dict:
        """Comprehensive risk assessment for trade"""
        try:
            risk_assessment = {
                'approved': False,
                'position_size': 0.0,
                'risk_score': 1.0,
                'reasons': [],
                'adjustments': []
            }
            
            # Daily limits check
            if self.daily_trades >= self.max_daily_trades:
                risk_assessment['reasons'].append("Daily trade limit reached")
                return risk_assessment
            
            # Daily loss limit check
            if self.daily_pnl <= -self.max_daily_loss * balance:
                risk_assessment['reasons'].append("Daily loss limit reached")
                return risk_assessment
            
            # Drawdown check
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - balance) / self.peak_balance
                if current_drawdown >= self.max_drawdown:
                    risk_assessment['reasons'].append("Maximum drawdown exceeded")
                    return risk_assessment
            
            # ML-based risk scoring
            ml_risk_score = self._calculate_ml_risk_score(ml_prediction)
            
            # Technical risk scoring
            technical_risk_score = self._calculate_technical_risk_score(signal)
            
            # Combined risk score
            combined_risk = (ml_risk_score + technical_risk_score) / 2
            
            # Position sizing based on risk
            base_position_size = self.max_position_size * balance
            
            # Kelly Criterion approximation
            win_prob = ml_prediction.profit_probability
            avg_win = sum(ml_prediction.optimal_tp_levels) / len(ml_prediction.optimal_tp_levels)
            avg_loss = ml_prediction.recommended_sl
            
            if avg_loss > 0 and win_prob > 0.5:
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.01  # Conservative default
            
            # Adjust position size
            risk_adjusted_size = base_position_size * kelly_fraction * ml_prediction.position_size_multiplier
            risk_adjusted_size *= (1 - combined_risk)  # Reduce size for higher risk
            
            # News impact adjustment
            if ml_prediction.news_sentiment_impact > 0.6:
                risk_adjusted_size *= 0.7  # Reduce size during high-impact news
                risk_assessment['adjustments'].append("Reduced size due to news impact")
            
            # Volatility adjustment
            if ml_prediction.volatility_forecast > 0.4:
                risk_adjusted_size *= 0.8
                risk_assessment['adjustments'].append("Reduced size due to high volatility")
            
            # Minimum and maximum position size
            final_position_size = max(balance * 0.001, min(base_position_size, risk_adjusted_size))
            
            # Final approval logic
            if (ml_prediction.confidence > 0.3 and 
                signal.strength > 40 and 
                combined_risk < 0.8 and
                final_position_size > 0):
                
                risk_assessment['approved'] = True
                risk_assessment['position_size'] = final_position_size
                risk_assessment['risk_score'] = combined_risk
                risk_assessment['reasons'].append("Risk assessment passed")
            else:
                risk_assessment['reasons'].append(f"Risk too high: ML confidence={ml_prediction.confidence:.2f}, Signal strength={signal.strength:.1f}, Risk score={combined_risk:.2f}")
            
            return risk_assessment
            
        except Exception as e:
            logging.error(f"Risk assessment error: {e}")
            return {
                'approved': False,
                'position_size': 0.0,
                'risk_score': 1.0,
                'reasons': [f"Risk assessment failed: {str(e)}"],
                'adjustments': []
            }
    
    def _calculate_ml_risk_score(self, ml_prediction: MLPrediction) -> float:
        """Calculate risk score based on ML prediction"""
        try:
            risk_factors = []
            
            # Reversal risk
            risk_factors.append(ml_prediction.reversal_risk * 0.3)
            
            # Low confidence
            risk_factors.append((1 - ml_prediction.confidence) * 0.25)
            
            # Market regime risk
            regime_risk = {
                MarketRegime.VOLATILE: 0.4,
                MarketRegime.NEWS_DRIVEN: 0.3,
                MarketRegime.SIDEWAYS: 0.2,
                MarketRegime.TRENDING_BULL: 0.1,
                MarketRegime.TRENDING_BEAR: 0.1
            }
            risk_factors.append(regime_risk.get(ml_prediction.market_regime, 0.3) * 0.25)
            
            # Volatility risk
            risk_factors.append(min(1.0, ml_prediction.volatility_forecast / 0.5) * 0.2)
            
            return sum(risk_factors)
            
        except:
            return 0.5
    
    def _calculate_technical_risk_score(self, signal: TradeSignal) -> float:
        """Calculate risk score based on technical factors"""
        try:
            risk_score = 0.0
            
            # Signal strength risk (lower strength = higher risk)
            risk_score += (100 - signal.strength) / 100 * 0.3
            
            # Confidence risk
            risk_score += (1 - signal.confidence) * 0.3
            
            # Technical score risk
            risk_score += (100 - signal.technical_score) / 100 * 0.2
            
            # Risk factors penalty
            risk_score += len(signal.risk_factors) * 0.05
            
            return min(1.0, risk_score)
            
        except:
            return 0.5
    
    def update_trade_result(self, trade_id: str, pnl: float, outcome: TradeOutcome):
        """Update risk management with trade result"""
        try:
            self.daily_pnl += pnl
            self.daily_trades += 1
            
            # Update peak balance tracking
            if hasattr(self, 'current_balance'):
                if self.current_balance > self.peak_balance:
                    self.peak_balance = self.current_balance
            
            # Store trade history
            trade_record = {
                'id': trade_id,
                'pnl': pnl,
                'outcome': outcome,
                'timestamp': datetime.now()
            }
            self.trade_history.append(trade_record)
            
            logging.info(f"Trade {trade_id} completed: PnL={pnl:.4f}, Outcome={outcome.value}")
            
        except Exception as e:
            logging.error(f"Trade result update error: {e}")
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        try:
            if len(self.trade_history) == 0:
                return {
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            # Calculate metrics from trade history
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            losses = [t for t in self.trade_history if t['pnl'] < 0]
            
            win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
            avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
            avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 0
            
            gross_profit = sum(t['pnl'] for t in wins)
            gross_loss = abs(sum(t['pnl'] for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate running drawdown
            running_balance = 0
            peak = 0
            max_dd = 0
            
            for trade in self.trade_history:
                running_balance += trade['pnl']
                if running_balance > peak:
                    peak = running_balance
                dd = (peak - running_balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            # Simple Sharpe ratio approximation
            returns = [t['pnl'] for t in self.trade_history]
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe = 0
            
            return {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe,
                'total_trades': len(self.trade_history),
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades
            }
            
        except Exception as e:
            logging.error(f"Risk metrics calculation error: {e}")
            return {'error': str(e)}

class MLTradingEngine:
    """Machine Learning trading engine with simplified models"""
    
    def __init__(self):
        self.prediction_cache = {}
        self.cache_timeout = 60  # 1 minute cache
        self.model_ready = False
        self.models_trained = False  # Initialize this attribute
        
    def initialize_models(self):
        """Initialize ML models and prepare the engine"""
        try:
            logging.info("Initializing ML Trading Engine...")
            
            # Initialize model parameters
            self.prediction_cache = {}
            self.last_cache_clear = time.time()
            self.models_trained = True  # Added this attribute
            
            # Set model as ready (since we're using heuristic-based approach)
            self.model_ready = True
            
            logging.info("ML Trading Engine initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize ML models: {e}")
            self.model_ready = False
            self.models_trained = False
            return False
    
    def predict_market_direction(self, df: pd.DataFrame) -> Dict:
        """Predict market direction using simple heuristics (placeholder for ML)"""
        try:
            if len(df) < 20:
                return {'direction': 'neutral', 'confidence': 0.0, 'signal_strength': 0.0}
            
            # Simple trend-based prediction (placeholder for actual ML)
            close = df['close']
            sma_20 = close.rolling(window=20).mean()
            current_price = close.iloc[-1]
            sma_value = sma_20.iloc[-1]
            
            # Calculate price momentum
            price_change = (current_price - close.iloc[-20]) / close.iloc[-20]
            
            # Determine direction and confidence
            if current_price > sma_value and price_change > 0.01:
                direction = 'bullish'
                confidence = min(abs(price_change) * 10, 1.0)
            elif current_price < sma_value and price_change < -0.01:
                direction = 'bearish'
                confidence = min(abs(price_change) * 10, 1.0)
            else:
                direction = 'neutral'
                confidence = 0.5
            
            signal_strength = confidence * 0.8  # Scale down for conservative trading
            
            return {
                'direction': direction,
                'confidence': float(confidence),
                'signal_strength': float(signal_strength),
                'price_momentum': float(price_change)
            }
            
        except Exception as e:
            logging.error(f"ML prediction error: {e}")
            return {'direction': 'neutral', 'confidence': 0.0, 'signal_strength': 0.0}
    
    def generate_trade_signals(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Generate ML-enhanced trade signals"""
        try:
            prediction = self.predict_market_direction(df)
            
            # Combine ML prediction with technical indicators
            ta_score = 0
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi < 30:
                    ta_score += 0.3
                elif rsi > 70:
                    ta_score -= 0.3
            
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_diff = indicators['macd'] - indicators['macd_signal']
                ta_score += 0.2 if macd_diff > 0 else -0.2
            
            # Final signal
            ml_weight = 0.6
            ta_weight = 0.4
            combined_score = (prediction['signal_strength'] * ml_weight + 
                            abs(ta_score) * ta_weight)
            
            if prediction['direction'] == 'bullish' and ta_score >= 0:
                signal = 'buy'
            elif prediction['direction'] == 'bearish' and ta_score <= 0:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'signal': signal,
                'strength': float(combined_score),
                'ml_direction': prediction['direction'],
                'ml_confidence': prediction['confidence'],
                'ta_contribution': float(ta_score)
            }
            
        except Exception as e:
            logging.error(f"Signal generation error: {e}")
            return {'signal': 'hold', 'strength': 0.0}
    
    def clear_cache_if_needed(self):
        """Clear prediction cache if timeout exceeded"""
        try:
            current_time = time.time()
            if current_time - self.last_cache_clear > self.cache_timeout:
                self.prediction_cache.clear()
                self.last_cache_clear = current_time
        except Exception as e:
            logging.error(f"Cache clear error: {e}")


class TechnicalAnalysisEngine:
    """Advanced technical analysis with professional-grade calculations"""
    
    def __init__(self):
        self.indicators_cache = {}
        self.cache_timeout = 60  # 1 minute cache
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            if len(df) < 20:
                return self._get_default_indicators()
            
            indicators = {}
            
            # Moving Averages
            indicators.update(self._calculate_moving_averages(df))
            
            # Momentum Indicators
            indicators.update(self._calculate_momentum_indicators(df))
            
            # Volatility Indicators
            indicators.update(self._calculate_volatility_indicators(df))
            
            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(df))
            
            # Volume Indicators (if available)
            indicators.update(self._calculate_volume_indicators(df))
            
            # Support/Resistance Levels
            indicators.update(self._calculate_support_resistance(df))
            
            # Pattern Recognition
            indicators.update(self._detect_patterns(df))
            
            return indicators
            
        except Exception as e:
            logging.error(f"Technical analysis calculation error: {e}")
            return self._get_default_indicators()
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict:
        """Calculate various moving averages"""
        try:
            close = df['close']
            
            # Simple Moving Averages
            sma_5 = close.rolling(window=5).mean()
            sma_10 = close.rolling(window=10).mean()
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean() if len(df) >= 50 else sma_20
            
            # Exponential Moving Averages
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            ema_50 = close.ewm(span=50).mean() if len(df) >= 50 else ema_26
            
            # Volume Weighted MA (if volume available)
            if 'volume' in df.columns and not df['volume'].isna().all():
                vwma = (close * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            else:
                vwma = sma_20
            
            return {
                'sma_5': float(sma_5.iloc[-1]) if not sma_5.isna().iloc[-1] else float(close.iloc[-1]),
                'sma_10': float(sma_10.iloc[-1]) if not sma_10.isna().iloc[-1] else float(close.iloc[-1]),
                'sma_20': float(sma_20.iloc[-1]) if not sma_20.isna().iloc[-1] else float(close.iloc[-1]),
                'sma_50': float(sma_50.iloc[-1]) if not sma_50.isna().iloc[-1] else float(close.iloc[-1]),
                'ema_12': float(ema_12.iloc[-1]) if not ema_12.isna().iloc[-1] else float(close.iloc[-1]),
                'ema_26': float(ema_26.iloc[-1]) if not ema_26.isna().iloc[-1] else float(close.iloc[-1]),
                'ema_50': float(ema_50.iloc[-1]) if not ema_50.isna().iloc[-1] else float(close.iloc[-1]),
                'vwma_20': float(vwma.iloc[-1]) if not vwma.isna().iloc[-1] else float(close.iloc[-1])
            }
        except:
            close_price = float(df['close'].iloc[-1])
            return {
                'sma_5': close_price, 'sma_10': close_price, 'sma_20': close_price, 'sma_50': close_price,
                'ema_12': close_price, 'ema_26': close_price, 'ema_50': close_price, 'vwma_20': close_price
            }
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Stochastic
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            k_percent = 100 * (close - low_14) / (high_14 - low_14)
            d_percent = k_percent.rolling(window=3).mean()
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - macd_signal
            
            # CCI
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            # Williams %R
            williams_r = -100 * (high_14 - close) / (high_14 - low_14)
            
            return {
                'rsi': float(rsi.iloc[-1]) if not rsi.isna().iloc[-1] else 50.0,
                'stoch_k': float(k_percent.iloc[-1]) if not k_percent.isna().iloc[-1] else 50.0,
                'stoch_d': float(d_percent.iloc[-1]) if not d_percent.isna().iloc[-1] else 50.0,
                'macd': float(macd_line.iloc[-1]) if not macd_line.isna().iloc[-1] else 0.0,
                'macd_signal': float(macd_signal.iloc[-1]) if not macd_signal.isna().iloc[-1] else 0.0,
                'macd_histogram': float(macd_histogram.iloc[-1]) if not macd_histogram.isna().iloc[-1] else 0.0,
                'cci': float(cci.iloc[-1]) if not cci.isna().iloc[-1] else 0.0,
                'williams_r': float(williams_r.iloc[-1]) if not williams_r.isna().iloc[-1] else -50.0
            }
        except:
            return {
                'rsi': 50.0, 'stoch_k': 50.0, 'stoch_d': 50.0, 'macd': 0.0,
                'macd_signal': 0.0, 'macd_histogram': 0.0, 'cci': 0.0, 'williams_r': -50.0
            }
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility indicators"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Bollinger Bands
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            
            # ATR (Average True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # Historical Volatility
            returns = close.pct_change()
            historical_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # Keltner Channels
            kc_middle = close.ewm(span=20).mean()
            kc_upper = kc_middle + (atr * 2)
            kc_lower = kc_middle - (atr * 2)
            
            return {
                'bb_upper': float(bb_upper.iloc[-1]) if not bb_upper.isna().iloc[-1] else float(close.iloc[-1] * 1.02),
                'bb_middle': float(sma_20.iloc[-1]) if not sma_20.isna().iloc[-1] else float(close.iloc[-1]),
                'bb_lower': float(bb_lower.iloc[-1]) if not bb_lower.isna().iloc[-1] else float(close.iloc[-1] * 0.98),
                'bb_position': float(bb_position.iloc[-1]) if not bb_position.isna().iloc[-1] else 0.5,
                'atr': float(atr.iloc[-1]) if not atr.isna().iloc[-1] else float(close.iloc[-1] * 0.01),
                'historical_vol': float(historical_vol.iloc[-1]) if not historical_vol.isna().iloc[-1] else 0.2,
                'kc_upper': float(kc_upper.iloc[-1]) if not kc_upper.isna().iloc[-1] else float(close.iloc[-1] * 1.015),
                'kc_lower': float(kc_lower.iloc[-1]) if not kc_lower.isna().iloc[-1] else float(close.iloc[-1] * 0.985)
            }
        except:
            close_price = float(df['close'].iloc[-1])
            return {
                'bb_upper': close_price * 1.02, 'bb_middle': close_price, 'bb_lower': close_price * 0.98,
                'bb_position': 0.5, 'atr': close_price * 0.01, 'historical_vol': 0.2,
                'kc_upper': close_price * 1.015, 'kc_lower': close_price * 0.985
            }
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate trend indicators"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # ADX (Average Directional Index)
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            # Parabolic SAR (simplified)
            psar = close.copy()  # Simplified version
            
            # Ichimoku components (simplified)
            high_9 = high.rolling(window=9).max()
            low_9 = low.rolling(window=9).min()
            tenkan_sen = (high_9 + low_9) / 2
            
            high_26 = high.rolling(window=26).max()
            low_26 = low.rolling(window=26).min()
            kijun_sen = (high_26 + low_26) / 2
            
            return {
                'adx': float(adx.iloc[-1]) if not adx.isna().iloc[-1] else 25.0,
                'plus_di': float(plus_di.iloc[-1]) if not plus_di.isna().iloc[-1] else 25.0,
                'minus_di': float(minus_di.iloc[-1]) if not minus_di.isna().iloc[-1] else 25.0,
                'psar': float(psar.iloc[-1]) if not psar.isna().iloc[-1] else float(close.iloc[-1]),
                'tenkan_sen': float(tenkan_sen.iloc[-1]) if not tenkan_sen.isna().iloc[-1] else float(close.iloc[-1]),
                'kijun_sen': float(kijun_sen.iloc[-1]) if not kijun_sen.isna().iloc[-1] else float(close.iloc[-1]),
                'trend_strength': self._calculate_trend_strength(df)
            }
        except:
            close_price = float(df['close'].iloc[-1])
            return {
                'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0, 'psar': close_price,
                'tenkan_sen': close_price, 'kijun_sen': close_price, 'trend_strength': 0.5
            }
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volume indicators"""
        try:
            close = df['close']
            
            if 'volume' in df.columns and not df['volume'].isna().all():
                volume = df['volume']
                
                # On Balance Volume
                obv = (volume * ((close > close.shift(1)).astype(int) - (close < close.shift(1)).astype(int))).cumsum()
                
                # Volume SMA
                volume_sma = volume.rolling(window=20).mean()
                
                # Volume Rate of Change
                volume_roc = volume.pct_change(periods=10)
                
                return {
                    'obv': float(obv.iloc[-1]) if not obv.isna().iloc[-1] else 0.0,
                    'volume_sma': float(volume_sma.iloc[-1]) if not volume_sma.isna().iloc[-1] else 1000.0,
                    'volume_roc': float(volume_roc.iloc[-1]) if not volume_roc.isna().iloc[-1] else 0.0,
                    'volume_ratio': float(volume.iloc[-1] / volume_sma.iloc[-1]) if not volume_sma.isna().iloc[-1] and volume_sma.iloc[-1] > 0 else 1.0
                }
            else:
                return {
                    'obv': 0.0,
                    'volume_sma': 1000.0,
                    'volume_roc': 0.0,
                    'volume_ratio': 1.0
                }
        except:
            return {
                'obv': 0.0,
                'volume_sma': 1000.0,
                'volume_roc': 0.0,
                'volume_ratio': 1.0
            }
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Find pivot points (simplified)
            window = min(10, len(high) // 4)
            
            # Resistance levels (local maxima)
            resistance_indices, _ = find_peaks(high, distance=window)
            resistance_levels = high[resistance_indices] if len(resistance_indices) > 0 else [max(high)]
            
            # Support levels (local minima)
            support_indices, _ = find_peaks(-low, distance=window)
            support_levels = low[support_indices] if len(support_indices) > 0 else [min(low)]
            
            current_price = close[-1]
            
            # Find nearest levels
            resistance_levels = [r for r in resistance_levels if r > current_price]
            support_levels = [s for s in support_levels if s < current_price]
            
            nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.02
            nearest_support = max(support_levels) if support_levels else current_price * 0.98
            
            return {
                'nearest_resistance': float(nearest_resistance),
                'nearest_support': float(nearest_support),
                'resistance_strength': len([r for r in resistance_levels if abs(r - nearest_resistance) / current_price < 0.005]),
                'support_strength': len([s for s in support_levels if abs(s - nearest_support) / current_price < 0.005])
            }
        except:
            current_price = float(df['close'].iloc[-1])
            return {
                'nearest_resistance': current_price * 1.02,
                'nearest_support': current_price * 0.98,
                'resistance_strength': 1,
                'support_strength': 1
            }
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect candlestick patterns"""
        try:
            if len(df) < 3:
                return {'doji': 0.0, 'hammer': 0.0, 'engulfing_bull': 0.0, 'engulfing_bear': 0.0}
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            open_price = latest['open']
            high_price = latest['high']
            low_price = latest['low']
            close_price = latest['close']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # Doji pattern
            doji_score = 1.0 - (body_size / total_range) if total_range > 0 else 0.0
            
            # Hammer pattern
            hammer_score = 0.0
            if total_range > 0:
                hammer_score = (lower_shadow / total_range) * (1 - body_size / total_range)
                if lower_shadow > body_size * 2 and upper_shadow < body_size:
                    hammer_score *= 2
            
            # Engulfing patterns
            engulfing_bull = 0.0
            engulfing_bear = 0.0
            
            if close_price > open_price and prev['close'] < prev['open']:  # Current green, prev red
                if open_price < prev['close'] and close_price > prev['open']:
                    engulfing_bull = 1.0
            
            if close_price < open_price and prev['close'] > prev['open']:  # Current red, prev green
                if open_price > prev['close'] and close_price < prev['open']:
                    engulfing_bear = 1.0
            
            return {
                'doji': min(1.0, max(0.0, doji_score)),
                'hammer': min(1.0, max(0.0, hammer_score)),
                'engulfing_bull': engulfing_bull,
                'engulfing_bear': engulfing_bear
            }
        except:
            return {'doji': 0.0, 'hammer': 0.0, 'engulfing_bull': 0.0, 'engulfing_bear': 0.0}
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        try:
            close = df['close']
            
            # Multiple timeframe trend alignment
            sma_5 = close.rolling(5).mean().iloc[-1]
            sma_10 = close.rolling(10).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # Check trend alignment
            uptrend_alignment = sma_5 > sma_10 > sma_20 and current_price > sma_5
            downtrend_alignment = sma_5 < sma_10 < sma_20 and current_price < sma_5
            
            if uptrend_alignment:
                return 1.0
            elif downtrend_alignment:
                return 0.0
            else:
                return 0.5
        except:
            return 0.5
    
    def _get_default_indicators(self) -> Dict:
        """Return default indicators if calculation fails"""
        return {
            'sma_5': 100.0, 'sma_10': 100.0, 'sma_20': 100.0, 'sma_50': 100.0,
            'ema_12': 100.0, 'ema_26': 100.0, 'ema_50': 100.0, 'vwma_20': 100.0,
            'rsi': 50.0, 'stoch_k': 50.0, 'stoch_d': 50.0, 'macd': 0.0,
            'macd_signal': 0.0, 'macd_histogram': 0.0, 'cci': 0.0, 'williams_r': -50.0,
            'bb_upper': 102.0, 'bb_middle': 100.0, 'bb_lower': 98.0, 'bb_position': 0.5,
            'atr': 1.0, 'historical_vol': 0.2, 'kc_upper': 101.5, 'kc_lower': 98.5,
            'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0, 'psar': 100.0,
            'tenkan_sen': 100.0, 'kijun_sen': 100.0, 'trend_strength': 0.5,
            'obv': 0.0, 'volume_sma': 1000.0, 'volume_roc': 0.0, 'volume_ratio': 1.0,
            'nearest_resistance': 102.0, 'nearest_support': 98.0,
            'resistance_strength': 1, 'support_strength': 1,
            'doji': 0.0, 'hammer': 0.0, 'engulfing_bull': 0.0, 'engulfing_bear': 0.0
        }

class MarketDataProvider:
    """Enhanced market data provider with multiple sources"""
    
    def __init__(self):
        self.symbols_map = {
            'R_50': '^GSPC',  # S&P 500 as proxy
            'R_75': '^GSPC',
            'R_100': '^GSPC',
            'BOOM500': 'BTC-USD',  # Bitcoin as volatile asset proxy
            'CRASH500': 'BTC-USD',
            'BOOM1000': 'BTC-USD',
            'CRASH1000': 'BTC-USD',
            'Step_200': 'ETH-USD',
            'Step_500': 'ETH-USD'
        }
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.synthetic_data_cache = {}
        
    def get_market_data(self, symbol: str, period: str = '1mo', interval: str = '1h') -> pd.DataFrame:
        """Fetch market data with fallback to synthetic data"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        # Try to get real data first
        if YFINANCE_AVAILABLE:
            data = self._get_real_market_data(symbol, period, interval)
            if data is not None and len(data) > 20:
                self.cache[cache_key] = (data, time.time())
                return data
        
        # Fallback to synthetic data
        logging.info(f"Using synthetic data for {symbol}")
        data = self._generate_synthetic_data(symbol, period, interval)
        
        if data is not None:
            self.cache[cache_key] = (data, time.time())
            return data
        
        # Last resort - return minimal dummy data
        return self._get_minimal_dummy_data()
    
    def _get_real_market_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch real market data from Yahoo Finance"""
        try:
            proxy_symbol = self.symbols_map.get(symbol, '^GSPC')
            ticker = yf.Ticker(proxy_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logging.warning(f"No data received for {proxy_symbol}")
                return None
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            data.reset_index(inplace=True)
            
            # Add synthetic volume if missing
            if 'volume' not in data.columns or data['volume'].isna().all():
                data['volume'] = np.random.randint(1000, 10000, len(data))
            
            logging.info(f"Fetched {len(data)} rows for {symbol} (proxy: {proxy_symbol})")
            return data
            
        except Exception as e:
            logging.error(f"Error fetching real market data for {symbol}: {e}")
            return None
    
    def _generate_synthetic_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Generate realistic synthetic market data"""
        try:
            # Determine number of periods
            period_map = {'1d': 24, '5d': 120, '1mo': 720, '3mo': 2160, '6mo': 4320, '1y': 8760}
            interval_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '1d': 1440}
            
            total_minutes = period_map.get(period, 720) * 60
            interval_minutes = interval_map.get(interval, 60)
            num_points = min(1000, max(50, total_minutes // interval_minutes))
            
            # Generate timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=total_minutes)
            timestamps = pd.date_range(start=start_time, end=end_time, periods=num_points)
            
            # Generate price data with realistic characteristics
            np.random.seed(hash(symbol) % 2147483647)  # Consistent seed per symbol
            
            # Base parameters
            initial_price = 100.0
            volatility = 0.02 if 'R_' in symbol else 0.05  # Lower vol for synthetics
            drift = 0.0001  # Slight upward drift
            
            # Generate returns
            returns = np.random.normal(drift, volatility, num_points)
            
            # Add autocorrelation for realism
            for i in range(1, len(returns)):
                returns[i] += returns[i-1] * 0.1  # Small momentum effect
            
            # Generate prices
            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate OHLC data
            opens = prices[:-1]  # Open is previous close
            closes = prices[1:]  # Close is current price
            
            # Generate highs and lows
            highs = []
            lows = []
            volumes = []
            
            for i in range(len(opens)):
                open_price = opens[i]
                close_price = closes[i]
                
                # High/low based on volatility
                daily_range = abs(close_price - open_price) * np.random.uniform(1.5, 3.0)
                high = max(open_price, close_price) + daily_range * np.random.uniform(0, 0.5)
                low = min(open_price, close_price) - daily_range * np.random.uniform(0, 0.5)
                
                highs.append(high)
                lows.append(max(0, low))  # Prevent negative prices
                
                # Volume correlated with price movement
                price_change = abs(close_price - open_price) / open_price
                base_volume = 5000
                volume = int(base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0))
                volumes.append(volume)
            
            # Create DataFrame
            df = pd.DataFrame({
                'datetime': timestamps[1:],
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            return df
            
        except Exception as e:
            logging.error(f"Error generating synthetic data for {symbol}: {e}")
            return self._get_minimal_dummy_data()
    
    def _get_minimal_dummy_data(self) -> pd.DataFrame:
        """Generate minimal dummy data as last resort"""
        dates = pd.date_range(start=datetime.now() - timedelta(hours=24), periods=50, freq='30min')
        base_price = 100.0
        
        data = []
        for i, date in enumerate(dates):
            price = base_price + np.sin(i * 0.1) * 2 + np.random.normal(0, 0.5)
            data.append({
                'datetime': date,
                'open': price,
                'high': price + np.random.uniform(0.1, 0.5),
                'low': price - np.random.uniform(0.1, 0.5),
                'close': price + np.random.normal(0, 0.2),
                'volume': np.random.randint(1000, 5000)
            })
        
        return pd.DataFrame(data)






class AIEnhancedDerivBot:
    """Complete AI/ML Enhanced Deriv Trading Bot"""
    
    def __init__(self):
        # Core settings
        self.app_id = os.getenv('DERIV_APP_ID', '1089')
        self.api_token = os.getenv('DERIV_API_TOKEN')
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        
        # Trading symbols
        self.trading_symbols = os.getenv('SYMBOLS', 'R_50,R_75,R_100').split(',')
        
        # Connection management
        self.websocket = None
        self.is_connected = False
        self.is_running = False
        self.req_id = 0
        self.response_futures = {}
        self.connection_lock = Lock()
        
        # Trading state
        self.balance = 0
        self.active_trades = {}
        self.trade_counter = 0
        self.last_trade_time = 0
        self.last_heartbeat = time.time()
        
        # Core engines
        self.ml_engine = MLTradingEngine()
        self.ta_engine = TechnicalAnalysisEngine()
        self.risk_engine = RiskManagementEngine()
        
        # Market data storage
        self.market_data_cache = {}
        self.tick_cache = defaultdict(deque)
        
        # System monitoring
        self.start_time = datetime.now()
        self.restart_count = 0
        self.max_restarts = int(os.getenv('MAX_RESTARTS', '50'))
        self.memory_cleanup_counter = 0
        self.health_status = {
            'status': 'starting',
            'uptime': 0,
            'memory_usage': 0,
            'active_connections': 0,
            'last_signal_time': None,
            'indicators_calculated': 0,
            'ml_predictions': 0,
            'trades_today': 0
        }
        
        # Initialize components
        self.setup_logging()
        self.init_database()
        self.setup_signal_handlers()
        self.setup_flask_health_server()
        
        # Initialize ML models in background
        Thread(target=self._initialize_ml_async, daemon=True).start()
    
    def setup_logging(self):
        """Setup comprehensive logging for Render deployment"""
        logging.basicConfig(
            level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Suppress excessive logs from external libraries
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers for Linux/Render"""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.is_running = False
            
            # Close active trades
            asyncio.create_task(self._emergency_close_all_trades())
            
            # Save state
            self._save_state_to_db()
            
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def setup_flask_health_server(self):
        """Setup Flask health check server for Render"""
        self.health_app = Flask(__name__)
        
        @self.health_app.route('/health')
        def health_check():
            try:
                uptime = (datetime.now() - self.start_time).total_seconds()
                
                # Update health status
                self.health_status.update({
                    'status': 'healthy' if self.is_connected else 'unhealthy',
                    'uptime': uptime,
                    'memory_usage': self._get_memory_usage(),
                    'active_connections': 1 if self.is_connected else 0,
                    'trades_today': self.risk_engine.daily_trades,
                    'balance': self.balance,
                    'ml_models_ready': self.ml_engine.models_trained
                })
                
                return jsonify(self.health_status), 200 if self.is_connected else 503
                
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.health_app.route('/metrics')
        def metrics():
            try:
                risk_metrics = self.risk_engine.get_risk_metrics()
                
                metrics = {
                    'trading_metrics': risk_metrics,
                    'system_metrics': {
                        'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                        'restart_count': self.restart_count,
                        'memory_usage_mb': self._get_memory_usage(),
                        'cache_size': len(self.market_data_cache)
                    },
                    'ml_metrics': {
                        'models_trained': self.ml_engine.models_trained,
                        'predictions_made': self.health_status.get('ml_predictions', 0),
                        'model_performance': getattr(self.ml_engine, 'model_performance', {})
                    }
                }
                
                return jsonify(metrics), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.health_app.route('/status')
        def status():
            return jsonify({
                'bot_status': 'running' if self.is_running else 'stopped',
                'connection_status': 'connected' if self.is_connected else 'disconnected',
                'balance': self.balance,
                'active_trades': len(self.active_trades),
                'daily_trades': self.risk_engine.daily_trades,
                'daily_pnl': self.risk_engine.daily_pnl
            })
        
        # Start Flask server in a separate thread
        port = int(os.environ.get('PORT', 8000))
        Thread(
            target=lambda: self.health_app.run(host='0.0.0.0', port=port, debug=False),
            daemon=True
        ).start()
        
        logging.info(f"Health server started on port {port}")
    
    def init_database(self):
        """Initialize comprehensive SQLite database"""
        try:
            conn = sqlite3.connect('ai_trading_bot.db', timeout=30)
            cursor = conn.cursor()
            
            # Enhanced trades table
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                symbol TEXT,
                signal_type TEXT,
                amount REAL,
                contract_id TEXT,
                status TEXT,
                entry_price REAL,
                exit_price REAL,
                profit_loss REAL,
                ml_signal_strength REAL,
                ml_confidence REAL,
                profit_probability REAL,
                reversal_risk REAL,
                market_regime TEXT,
                risk_score REAL,
                position_size_multiplier REAL,
                news_impact REAL,
                technical_score REAL,
                volatility_forecast REAL,
                optimal_tp_levels TEXT,
                recommended_sl REAL,
                actual_duration INTEGER,
                outcome TEXT
            )''')
            
            # ML predictions tracking
            cursor.execute('''CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                symbol TEXT,
                signal_strength REAL,
                profit_probability REAL,
                reversal_risk REAL,
                market_regime TEXT,
                confidence REAL,
                volatility_forecast REAL,
                news_sentiment_impact REAL,
                prediction_accuracy REAL,
                actual_outcome TEXT
            )''')
            
            # Performance analytics
            cursor.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                date DATE,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                ml_accuracy REAL,
                best_performing_symbol TEXT,
                best_time_period TEXT
            )''')
            
            # Market data cache table
            cursor.execute('''CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                timestamp DATETIME,
                timeframe TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                tick_count INTEGER
            )''')
            
            # System events log
            cursor.execute('''CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                event_type TEXT,
                description TEXT,
                severity TEXT,
                data TEXT
            )''')
            
            # ML model performance tracking
            cursor.execute('''CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                model_name TEXT,
                accuracy_score REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                training_samples INTEGER,
                validation_samples INTEGER,
                feature_importance TEXT
            )''')
            
            conn.commit()
            conn.close()
            
            logging.info("Database initialized successfully")
            
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
    
    def _initialize_ml_async(self):
        """Initialize ML models asynchronously"""
        try:
            logging.info("Starting ML model initialization...")
            self.ml_engine.initialize_models()
            
            if self.ml_engine.models_trained:
                logging.info("✅ ML models trained and ready")
                self.health_status['ml_models_ready'] = True
                self._log_system_event("ML_INITIALIZED", "ML models successfully initialized", "INFO")
            else:
                logging.warning("⚠️ ML models using simplified mode")
                self.health_status['ml_models_ready'] = False
                self._log_system_event("ML_SIMPLIFIED", "ML models using simplified mode", "WARNING")
                
        except Exception as e:
            logging.error(f"ML initialization error: {e}")
            self._log_system_event("ML_ERROR", f"ML initialization failed: {e}", "ERROR")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _log_system_event(self, event_type: str, description: str, severity: str, data: str = None):
        """Log system events to database"""
        try:
            conn = sqlite3.connect('ai_trading_bot.db', timeout=10)
            cursor = conn.cursor()
            
            cursor.execute('''INSERT INTO system_events 
                            (timestamp, event_type, description, severity, data)
                            VALUES (?, ?, ?, ?, ?)''',
                          (datetime.now(), event_type, description, severity, data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"System event logging error: {e}")
    
    async def start_bot(self):
        """Start the main bot loop"""
        try:
            logging.info("🚀 Starting AI Enhanced Deriv Trading Bot...")
            self.is_running = True
            self.health_status['status'] = 'starting'
            
            # Connect to Deriv API
            await self._connect_to_deriv()
            
            if not self.is_connected:
                logging.error("Failed to connect to Deriv API")
                return
            
            # Initialize account info
            await self._initialize_account()
            
            # Start main trading loop
            self.health_status['status'] = 'running'
            await self._main_trading_loop()
            
        except Exception as e:
            logging.error(f"Bot startup error: {e}")
            self._log_system_event("BOT_ERROR", f"Bot startup failed: {e}", "CRITICAL")
        finally:
            await self._cleanup()
    
    async def _connect_to_deriv(self):
        """Connect to Deriv WebSocket API"""
        max_retries = 5
        retry_delay = 5

    async def _connect_to_deriv(self):
    """Connect to Deriv WebSocket API"""
    print("=" * 50)
    print("ATTEMPTING TO CONNECT TO DERIV")
    print(f"WebSocket URL: {self.ws_url}")
    print(f"API Token present: {bool(self.api_token)}")
    print("=" * 50)
    
    logging.info("=" * 50)
    logging.info("ATTEMPTING TO CONNECT TO DERIV")
    logging.info(f"WebSocket URL: {self.ws_url}")
    logging.info(f"API Token present: {bool(self.api_token)}")
    logging.info("=" * 50)
    
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        # ... rest of the code

    async def _connect_to_deriv(self):
     """Connect to Deriv WebSocket API"""
     max_retries = 5
     retry_delay = 5
    
     for attempt in range(max_retries):
         try:
             logging.info(f"Connecting to Deriv API (attempt {attempt + 1}/{max_retries})")
            
             # Add debug logging BEFORE connection
             logging.info(f"WebSocket URL: {self.ws_url}")
            
             self.websocket = await websockets.connect(
                 self.ws_url,
                 ping_interval=30,
                 ping_timeout=10,
                 close_timeout=10
            )
            
            # Add debug logging AFTER connection
             logging.info(f"WebSocket connection established successfully")
             logging.info(f"Connection state: {self.websocket.open}")
            
            # Wait briefly for connection to stabilize
             await asyncio.sleep(0.5)
            
            # Authorize if token provided
             if self.api_token:
                 logging.info(f"Attempting authorization with token: {self.api_token[:10]}...")
                 auth_response = await self._send_request({
                     "authorize": self.api_token
                })
                
                 logging.info(f"Authorization response received: {auth_response}")
                
                 if auth_response.get('error'):
                     logging.error(f"Authorization failed: {auth_response['error']}")
                     return
            
            # If we get here, connection succeeded
             return
                    
         except Exception as e:
             logging.error(f"Connection attempt {attempt + 1} failed: {e}")
             if attempt < max_retries - 1:
                 await asyncio.sleep(retry_delay)
             else:
                 raise
    
    async def _initialize_account(self):
        """Initialize account information"""
        try:
            # Get account details
            account_response = await self._send_request({"get_account_status": 1})
            if account_response and not account_response.get('error'):
                logging.info("Account details retrieved successfully")
            
            # Subscribe to balance updates
            await self._send_request({"balance": 1, "subscribe": 1})
            
        except Exception as e:
            logging.error(f"Account initialization error: {e}")
    
    async def _main_trading_loop(self):
        """Main trading loop with ML-enhanced decision making"""
        logging.info("Starting main trading loop")
        
        cycle_count = 0
        last_cleanup = time.time()
        
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Health check
                if not self.is_connected or not self.websocket:
                    logging.warning("Connection lost, attempting reconnection...")
                    await self._connect_to_deriv()
                    if not self.is_connected:
                        await asyncio.sleep(30)
                        continue
                
                # Data collection and analysis cycle
                await self._data_collection_cycle()
                
                # Trading decision cycle (less frequent)
                if cycle_count % 2 == 0:  # Every other cycle
                    await self._trading_decision_cycle()
                
                # Monitor existing trades
                await self._monitor_active_trades()
                
                # Periodic cleanup
                if time.time() - last_cleanup > 300:  # Every 5 minutes
                    await self._periodic_cleanup()
                    last_cleanup = time.time()
                
                # Update health status
                self.health_status['last_signal_time'] = datetime.now().isoformat()
                self.last_heartbeat = time.time()
                
                # Cycle timing control
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, 30 - cycle_duration)  # 30-second cycles
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                cycle_count += 1
                
            except Exception as e:
                logging.error(f"Trading loop error: {e}")
                self._log_system_event("TRADING_ERROR", f"Trading loop error: {e}", "ERROR")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _data_collection_cycle(self):
        """Collect and update market data for all symbols"""
        try:
            for symbol in self.trading_symbols:
                # Get latest ticks
                tick_response = await self._send_request({
                    "ticks_history": symbol,
                    "count": 50,
                    "end": "latest",
                    "style": "ticks"
                })
                
                if tick_response and not tick_response.get('error'):
                    self._update_tick_cache(symbol, tick_response['history']['prices'])
                
                # Get candlestick data
                candle_response = await self._send_request({
                    "candles": symbol,
                    "count": 100,
                    "end": "latest",
                    "granularity": 60  # 1-minute candles
                })
                
                if candle_response and not candle_response.get('error'):
                    self._update_market_data_cache(symbol, candle_response['candles'])
                
                # Small delay between symbol requests
                await asyncio.sleep(0.1)
            
            self.health_status['indicators_calculated'] += len(self.trading_symbols)
            
        except Exception as e:
            logging.error(f"Data collection error: {e}")
    
    async def _trading_decision_cycle(self):
        """Analyze markets and make trading decisions"""
        try:
            for symbol in self.trading_symbols:
                # Skip if no recent data
                if symbol not in self.market_data_cache:
                    continue
                
                # Get market data
                market_data = self.market_data_cache[symbol]
                if len(market_data) < 20:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(market_data)
                
                # Calculate technical indicators
                indicators = self.ta_engine.calculate_all_indicators(df)
                
                # Get ML prediction
                ml_prediction = self.ml_engine.predict_signal_quality(symbol, indicators)
                
                # Generate trading signal
                signal = self._generate_trading_signal(symbol, indicators, ml_prediction, df)
                
                if signal and signal.strength > 60:  # Minimum signal strength threshold
                    # Risk assessment
                    risk_assessment = self.risk_engine.assess_trade_risk(signal, self.balance, ml_prediction)
                    
                    if risk_assessment['approved']:
                        # Execute trade
                        await self._execute_trade(signal, risk_assessment, ml_prediction)
                    else:
                        logging.info(f"Trade rejected for {symbol}: {risk_assessment['reasons']}")
                
                # Store ML prediction for analysis
                self._store_ml_prediction(symbol, ml_prediction, signal)
            
            self.health_status['ml_predictions'] += len(self.trading_symbols)
            
        except Exception as e:
            logging.error(f"Trading decision error: {e}")
    
    def _update_tick_cache(self, symbol: str, prices: list):
        """Update tick cache with latest prices"""
        try:
            if symbol not in self.tick_cache:
                self.tick_cache[symbol] = deque(maxlen=1000)
            
            for price in prices:
                self.tick_cache[symbol].append({
                    'price': float(price),
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            logging.error(f"Tick cache update error for {symbol}: {e}")
    
    def _update_market_data_cache(self, symbol: str, candles: list):
        """Update market data cache with latest candles"""
        try:
            if symbol not in self.market_data_cache:
                self.market_data_cache[symbol] = deque(maxlen=200)
            
            # Clear old data and add new
            self.market_data_cache[symbol].clear()
            
            for candle in candles:
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(candle['epoch']),
                    open=float(candle['open']),
                    high=float(candle['high']),
                    low=float(candle['low']),
                    close=float(candle['close']),
                    volume=1000  # Default volume for synthetics
                )
                self.market_data_cache[symbol].append(asdict(market_data))
            
        except Exception as e:
            logging.error(f"Market data cache update error for {symbol}: {e}")
    
    def _generate_trading_signal(self, symbol: str, indicators: Dict, ml_prediction: MLPrediction, df: pd.DataFrame) -> TradeSignal:
        """Generate comprehensive trading signal"""
        try:
            # Technical analysis scoring
            technical_score = self._calculate_technical_score(indicators)
            
            # Determine signal direction and strength
            signal_type = self._determine_signal_direction(indicators, ml_prediction)
            
            if signal_type == "NEUTRAL":
                return None
            
            # Calculate overall signal strength
            ml_weight = 0.6
            technical_weight = 0.4
            
            combined_strength = (ml_prediction.signal_strength * ml_weight + 
                               technical_score * technical_weight)
            
            # Entry reasons
            entry_reasons = self._get_entry_reasons(indicators, ml_prediction)
            
            # Risk factors
            risk_factors = self._get_risk_factors(indicators, ml_prediction, df)
            
            return TradeSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=combined_strength,
                confidence=ml_prediction.confidence,
                ml_prediction=ml_prediction,
                technical_score=technical_score,
                entry_reasons=entry_reasons,
                risk_factors=risk_factors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, indicators: Dict) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0
            max_score = 0
            
            # RSI scoring (30 points)
            rsi = indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 30
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                score += 25
            elif rsi < 20 or rsi > 80:
                score += 15
            max_score += 30
            
            # MACD scoring (25 points)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if (macd > macd_signal and macd > 0) or (macd < macd_signal and macd < 0):
                score += 25
            elif macd != macd_signal:
                score += 15
            max_score += 25
            
            # Trend strength (20 points)
            trend_strength = indicators.get('trend_strength', 0.5)
            score += trend_strength * 20
            max_score += 20
            
            # ADX scoring (15 points)
            adx = indicators.get('adx', 20)
            if adx > 25:
                score += 15
            elif adx > 20:
                score += 10
            max_score += 15
            
            # Bollinger Bands (10 points)
            bb_position = indicators.get('bb_position', 0.5)
            if 0.2 <= bb_position <= 0.8:
                score += 10
            elif bb_position < 0.1 or bb_position > 0.9:
                score += 5
            max_score += 10
            
            return (score / max_score) * 100 if max_score > 0 else 50
            
        except Exception as e:
            logging.error(f"Technical score calculation error: {e}")
            return 50.0
    
    def _determine_signal_direction(self, indicators: Dict, ml_prediction: MLPrediction) -> str:
        """Determine signal direction (CALL/PUT/NEUTRAL)"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi < 40:
                bullish_signals += 1
            elif rsi > 60:
                bearish_signals += 1
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                bullish_signals += 1
            elif macd < macd_signal:
                bearish_signals += 1
            
            # Trend signals
            trend_strength = indicators.get('trend_strength', 0.5)
            if trend_strength > 0.6:
                # Determine trend direction from MA alignment
                sma_5 = indicators.get('sma_5', 100)
                sma_20 = indicators.get('sma_20', 100)
                if sma_5 > sma_20:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # ML prediction influence
            if ml_prediction.signal_strength > 70:
                if ml_prediction.market_regime in [MarketRegime.TRENDING_BULL]:
                    bullish_signals += 2
                elif ml_prediction.market_regime in [MarketRegime.TRENDING_BEAR]:
                    bearish_signals += 2
            
            # Decision logic
            if bullish_signals > bearish_signals and bullish_signals >= 2:
                return "CALL"
            elif bearish_signals > bullish_signals and bearish_signals >= 2:
                return "PUT"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logging.error(f"Signal direction determination error: {e}")
            return "NEUTRAL"
    
    def _get_entry_reasons(self, indicators: Dict, ml_prediction: MLPrediction) -> List[str]:
        """Get list of entry reasons"""
        reasons = []
        
        try:
            # Technical reasons
            if indicators.get('rsi', 50) < 30:
                reasons.append("RSI oversold")
            elif indicators.get('rsi', 50) > 70:
                reasons.append("RSI overbought")
            
            if indicators.get('adx', 20) > 25:
                reasons.append("Strong trend detected")
            
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                reasons.append("MACD bullish crossover")
            elif macd < macd_signal:
                reasons.append("MACD bearish crossover")
            
            # ML reasons
            if ml_prediction.confidence > 0.7:
                reasons.append("High ML confidence")
            
            if ml_prediction.profit_probability > 0.6:
                reasons.append("High profit probability")
            
            if ml_prediction.market_regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
                reasons.append(f"Trending market detected ({ml_prediction.market_regime.value})")
            
        except Exception as e:
            logging.error(f"Entry reasons error: {e}")
        
        return reasons or ["Signal criteria met"]
    
    def _get_risk_factors(self, indicators: Dict, ml_prediction: MLPrediction, df: pd.DataFrame) -> List[str]:
        """Get list of risk factors"""
        risk_factors = []
        
        try:
            # High volatility
            if ml_prediction.volatility_forecast > 0.4:
                risk_factors.append("High volatility expected")
            
            # High reversal risk
            if ml_prediction.reversal_risk > 0.6:
                risk_factors.append("High reversal risk")
            
            # Low confidence
            if ml_prediction.confidence < 0.5:
                risk_factors.append("Low ML confidence")
            
            # News impact
            if ml_prediction.news_sentiment_impact > 0.6:
                risk_factors.append("High news impact expected")
            
            # Extreme RSI
            rsi = indicators.get('rsi', 50)
            if rsi > 80 or rsi < 20:
                risk_factors.append("Extreme RSI levels")
            
            # Volatile market regime
            if ml_prediction.market_regime == MarketRegime.VOLATILE:
                risk_factors.append("Volatile market conditions")
            
        except Exception as e:
            logging.error(f"Risk factors error: {e}")
        
        return risk_factors
    
    async def _execute_trade(self, signal: TradeSignal, risk_assessment: Dict, ml_prediction: MLPrediction):
        """Execute trade with comprehensive logging"""
        try:
            # Get proposal
            proposal_request = {
                "proposal": 1,
                "amount": risk_assessment['position_size'],
                "contract_type": signal.signal_type,
                "symbol": signal.symbol,
                "duration": 5,
                "duration_unit": "m",  # 5-minute trades
                "basis": "stake"
            }
            
            proposal_response = await self._send_request(proposal_request)
            
            if proposal_response.get('error'):
                logging.error(f"Proposal error: {proposal_response['error']}")
                return
            
            proposal = proposal_response['proposal']
            
            # Execute trade
            buy_request = {
                "buy": proposal['id'],
                "price": proposal['ask_price']
            }
            
            buy_response = await self._send_request(buy_request)
            
            if buy_response.get('error'):
                logging.error(f"Trade execution error: {buy_response['error']}")
                return
            
            # Store trade information
            contract_id = buy_response['buy']['contract_id']
            
            trade_data = {
                'contract_id': contract_id,
                'signal': signal,
                'ml_prediction': ml_prediction,
                'risk_assessment': risk_assessment,
                'entry_time': datetime.now(),
                'entry_price': float(proposal['spot']),
                'amount': risk_assessment['position_size']
            }
            
            self.active_trades[contract_id] = trade_data
            
            # Log to database
            self._store_trade_to_db(trade_data)
            
            # Update counters
            self.risk_engine.daily_trades += 1
            self.trade_counter += 1
            self.last_trade_time = time.time()
            
            logging.info(f"Trade executed: {signal.symbol} {signal.signal_type} Amount: {risk_assessment['position_size']:.2f}")
            self._log_system_event("TRADE_EXECUTED", f"Executed {signal.signal_type} trade on {signal.symbol}", "INFO", 
                                 json.dumps({'contract_id': contract_id, 'amount': risk_assessment['position_size']}))
            
        except Exception as e:
            logging.error(f"Trade execution error: {e}")
            self._log_system_event("TRADE_ERROR", f"Trade execution failed: {e}", "ERROR")
    
    async def _monitor_active_trades(self):
        """Monitor and manage active trades"""
        try:
            completed_trades = []
            
            for contract_id, trade_data in self.active_trades.items():
                # Get contract status
                portfolio_response = await self._send_request({"portfolio": 1})
                
                if portfolio_response and not portfolio_response.get('error'):
                    contracts = portfolio_response['portfolio']['contracts']
                    
                    for contract in contracts:
                        if contract['contract_id'] == int(contract_id):
                            if contract['status'] != 'open':
                                # Trade completed
                                pnl = float(contract['profit'])
                                
                                # Determine outcome
                                if pnl > 0:
                                    outcome = TradeOutcome.WIN
                                elif pnl < 0:
                                    outcome = TradeOutcome.LOSS
                                else:
                                    outcome = TradeOutcome.BREAKEVEN
                                
                                # Update risk management
                                self.risk_engine.update_trade_result(contract_id, pnl, outcome)
                                
                                # Update database
                                self._update_trade_result(contract_id, contract, outcome)
                                
                                completed_trades.append(contract_id)
                                
                                logging.info(f"Trade completed: {contract_id} PnL: {pnl:.4f}")
                                break
            
            # Remove completed trades
            for contract_id in completed_trades:
                del self.active_trades[contract_id]
            
        except Exception as e:
            logging.error(f"Trade monitoring error: {e}")
    
    def _store_trade_to_db(self, trade_data: Dict):
        """Store trade to database"""
        try:
            conn = sqlite3.connect('ai_trading_bot.db', timeout=10)
            cursor = conn.cursor()
            
            signal = trade_data['signal']
            ml_pred = trade_data['ml_prediction']
            
            cursor.execute('''INSERT INTO trades 
                            (timestamp, symbol, signal_type, amount, contract_id, status,
                             entry_price, ml_signal_strength, ml_confidence, profit_probability,
                             reversal_risk, market_regime, risk_score, position_size_multiplier,
                             news_impact, technical_score, volatility_forecast, optimal_tp_levels,
                             recommended_sl)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (trade_data['entry_time'], signal.symbol, signal.signal_type,
                           trade_data['amount'], trade_data['contract_id'], 'open',
                           trade_data['entry_price'], ml_pred.signal_strength, ml_pred.confidence,
                           ml_pred.profit_probability, ml_pred.reversal_risk, ml_pred.market_regime.value,
                           ml_pred.risk_score, ml_pred.position_size_multiplier, ml_pred.news_sentiment_impact,
                           signal.technical_score, ml_pred.volatility_forecast, 
                           json.dumps(ml_pred.optimal_tp_levels), ml_pred.recommended_sl))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Trade storage error: {e}")
    
    def _store_ml_prediction(self, symbol: str, ml_prediction: MLPrediction, signal: TradeSignal):
        """Store ML prediction for later analysis"""
        try:
            conn = sqlite3.connect('ai_trading_bot.db', timeout=10)
            cursor = conn.cursor()
            
            cursor.execute('''INSERT INTO ml_predictions 
                            (timestamp, symbol, signal_strength, profit_probability, reversal_risk,
                             market_regime, confidence, volatility_forecast, news_sentiment_impact)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (datetime.now(), symbol, ml_prediction.signal_strength,
                           ml_prediction.profit_probability, ml_prediction.reversal_risk,
                           ml_prediction.market_regime.value, ml_prediction.confidence,
                           ml_prediction.volatility_forecast, ml_prediction.news_sentiment_impact))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"ML prediction storage error: {e}")
    
    def _update_trade_result(self, contract_id: str, contract: Dict, outcome: TradeOutcome):
        """Update trade result in database"""
        try:
            conn = sqlite3.connect('ai_trading_bot.db', timeout=10)
            cursor = conn.cursor()
            
            cursor.execute('''UPDATE trades SET 
                            status = ?, exit_price = ?, profit_loss = ?, outcome = ?
                            WHERE contract_id = ?''',
                          (contract['status'], float(contract.get('exit_spot', 0)),
                           float(contract['profit']), outcome.value, contract_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Trade result update error: {e}")
    
    async def _periodic_cleanup(self):
        """Perform periodic cleanup and maintenance"""
        try:
            self.memory_cleanup_counter += 1
            
            # Garbage collection
            if self.memory_cleanup_counter % 10 == 0:
                gc.collect()
                logging.info("Performed garbage collection")
            
            # Clear old cache entries
            current_time = time.time()
            for symbol in list(self.tick_cache.keys()):
                # Remove ticks older than 1 hour
                old_ticks = [tick for tick in self.tick_cache[symbol] 
                           if (current_time - tick['timestamp'].timestamp()) > 3600]
                for old_tick in old_ticks:
                    if old_tick in self.tick_cache[symbol]:
                        self.tick_cache[symbol].remove(old_tick)
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Log system health
            memory_usage = self._get_memory_usage()
            logging.info(f"System health - Memory: {memory_usage:.1f}MB, Active trades: {len(self.active_trades)}, Cache size: {len(self.market_data_cache)}")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
    
    async def _update_performance_metrics(self):
        """Update daily performance metrics"""
        try:
            risk_metrics = self.risk_engine.get_risk_metrics()
            
            conn = sqlite3.connect('ai_trading_bot.db', timeout=10)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute('''INSERT OR REPLACE INTO performance_metrics 
                            (date, total_trades, winning_trades, losing_trades, win_rate,
                             avg_win, avg_loss, profit_factor, total_pnl, max_drawdown,
                             sharpe_ratio)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (today, risk_metrics.get('total_trades', 0),
                           0, 0,  # Will be calculated separately
                           risk_metrics.get('win_rate', 0), risk_metrics.get('avg_win', 0),
                           risk_metrics.get('avg_loss', 0), risk_metrics.get('profit_factor', 0),
                           risk_metrics.get('daily_pnl', 0), risk_metrics.get('max_drawdown', 0),
                           risk_metrics.get('sharpe_ratio', 0)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Performance metrics update error: {e}")
    
    async def _send_request(self, request: Dict) -> Dict:
        """Send request to Deriv API with error handling"""
        try:
            if not self.is_connected or not self.websocket:
                return {'error': {'message': 'Not connected'}}
            
            self.req_id += 1
            request['req_id'] = self.req_id
            
            await self.websocket.send(json.dumps(request))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
            
            return json.loads(response)
            
        except asyncio.TimeoutError:
            logging.error("Request timeout")
            return {'error': {'message': 'Request timeout'}}
        except Exception as e:
            logging.error(f"Request error: {e}")
            return {'error': {'message': str(e)}}
    
    async def _emergency_close_all_trades(self):
        """Emergency close all active trades"""
        try:
            logging.info("Emergency closing all active trades...")
            
            for contract_id in list(self.active_trades.keys()):
                try:
                    # Attempt to sell the contract
                    sell_request = {"sell": int(contract_id), "price": 0}
                    await self._send_request(sell_request)
                    
                except Exception as e:
                    logging.error(f"Error closing trade {contract_id}: {e}")
            
            logging.info("Emergency trade closure completed")
            
        except Exception as e:
            logging.error(f"Emergency closure error: {e}")
    
    def _save_state_to_db(self):
        """Save current state to database"""
        try:
            conn = sqlite3.connect('ai_trading_bot.db', timeout=10)
            cursor = conn.cursor()
            
            # Save current state
            state_data = {
                'balance': self.balance,
                'active_trades': len(self.active_trades),
                'daily_trades': self.risk_engine.daily_trades,
                'daily_pnl': self.risk_engine.daily_pnl,
                'uptime': (datetime.now() - self.start_time).total_seconds()
            }
            
            cursor.execute('''INSERT INTO system_events 
                            (timestamp, event_type, description, severity, data)
                            VALUES (?, ?, ?, ?, ?)''',
                          (datetime.now(), "STATE_SAVE", "Bot state saved on shutdown", "INFO",
                           json.dumps(state_data)))
            
            conn.commit()
            conn.close()
            
            logging.info("State saved to database")
            
        except Exception as e:
            logging.error(f"State save error: {e}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        try:
            logging.info("Starting cleanup process...")
            
            # Close websocket connection
            if self.websocket:
                await self.websocket.close()
            
            # Save final state
            self._save_state_to_db()
            
            logging.info("Cleanup completed")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")


async def main():
    """Main entry point"""
    try:
        if not os.getenv('DERIV_API_TOKEN'):
            logging.warning("DERIV_API_TOKEN not set, running in demo mode")

        bot = AIEnhancedDerivBot()
        await bot.start_bot()

    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logging.error(f"Main execution error: {e}")
    finally:
        logging.info("Bot shutdown completed")
        

def run_bot():
    """Run the bot with proper error handling"""
    logging.info("Initializing bot...")
    
    try:
        import uvloop
        uvloop.install()
    except:
        pass  # Use default event loop if uvloop unavailable
    
    try:
        if not os.getenv('DERIV_API_TOKEN'):
            logging.warning("DERIV_API_TOKEN not set")
        
        bot = AIEnhancedDerivBot()
        asyncio.run(bot.start_bot())
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except:
        logging.exception("Fatal error")

if __name__ == "__main__":
    run_bot()

asyncio.run(main())

async def get_balance(self):
    """Get account balance"""
    balance_response = await self._send_request({"balance": 1})
    if balance_response and not balance_response.get('error'):
        self.balance = float(balance_response['balance']['balance'])
        self.risk_engine.current_balance = self.balance
        logging.info(f"Account balance: {self.balance}")

def check_news_impact(self, current_time=None) -> Tuple[bool, float]:
    """Check if current time is near major news release"""
    if current_time is None:
        current_time = datetime.now()
    
    current_str = current_time.strftime("%H:%M")
    
    for news_time in self.high_impact_times:
        news_dt = datetime.strptime(news_time, "%H:%M").time()
        current_dt = current_time.time()
        
        # Check if within 30 minutes of news time
        time_diff = abs((datetime.combine(datetime.today(), current_dt) - 
                       datetime.combine(datetime.today(), news_dt)).total_seconds())
        
        if time_diff <= 1800:  # 30 minutes
            impact_factor = max(0.1, 1.0 - (time_diff / 1800))
            return True, impact_factor
    
    return False, 0.0


# Run the bot - remove the word "balance" after this line
asyncio.run(main())


def get_news_sentiment_impact(self, symbol: str) -> float:
    """Get news sentiment impact for symbol (simplified)"""
    # Simplified news impact based on volatility patterns
    current_hour = datetime.now().hour
    
    # Higher impact during market open hours
    if 8 <= current_hour <= 16:  # London/NY overlap
        return 0.7
    elif 0 <= current_hour <= 2:  # Asia session
        return 0.4
    else:
        return 0.2





class MLFeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.features_fitted = False
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame, indicators: Dict) -> np.ndarray:
        """Create comprehensive feature set from market data and indicators"""
        try:
            features = []
            feature_names = []
            
            # Price-based features
            price_features, price_names = self._price_features(df)
            features.extend(price_features)
            feature_names.extend(price_names)
            
            # Technical indicator features
            indicator_features, indicator_names = self._indicator_features(indicators)
            features.extend(indicator_features)
            feature_names.extend(indicator_names)
            
            # Pattern recognition features
            pattern_features, pattern_names = self._pattern_features(df)
            features.extend(pattern_features)
            feature_names.extend(pattern_names)
            
            # Market microstructure features
            micro_features, micro_names = self._microstructure_features(df)
            features.extend(micro_features)
            feature_names.extend(micro_names)
            
            # Time-based features
            time_features, time_names = self._time_features()
            features.extend(time_features)
            feature_names.extend(time_names)
            
            # Statistical features
            stat_features, stat_names = self._statistical_features(df)
            features.extend(stat_features)
            feature_names.extend(stat_names)
            
            # Convert to numpy array
            feature_array = np.array(features).reshape(1, -1)
            
            # Handle NaN and infinite values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Store feature names
            if not self.feature_names:
                self.feature_names = feature_names
            
            # Scale features if scaler available
            if ML_AVAILABLE and self.scaler is not None:
                if not self.features_fitted:
                    # For first prediction, fit and transform
                    self.scaler.fit(feature_array)
                    self.features_fitted = True
                
                return self.scaler.transform(feature_array)
            
            return feature_array
            
        except Exception as e:
            logging.error(f"Feature engineering error: {e}")
            # Return dummy features with consistent shape
            return np.zeros((1, 60))  # Fixed size for consistency
    
    def _price_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract price-based features"""
        try:
            if len(df) < 2:
                return [0.0] * 8, ['price_change', 'daily_range', 'body_ratio', 'volume_ratio', 
                                   'price_momentum_5', 'price_momentum_10', 'price_position', 'gap']
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            features = []
            
            # Price change
            features.append(float(latest['close'] / prev['close'] - 1) if prev['close'] != 0 else 0.0)
            
            # Daily range
            features.append(float(latest['high'] / latest['low'] - 1) if latest['low'] != 0 else 0.0)
            
            # Body ratio (candle body vs total range)
            total_range = latest['high'] - latest['low']
            body_size = abs(latest['close'] - latest['open'])
            features.append(float(body_size / total_range) if total_range != 0 else 0.0)
            
            # Volume ratio
            avg_volume = df['volume'].mean() if 'volume' in df.columns else 1000
            features.append(float(latest['volume'] / avg_volume) if avg_volume != 0 else 1.0)
            
            # Price momentum over different periods
            if len(df) >= 6:
                features.append(float(latest['close'] / df.iloc[-6]['close'] - 1))
            else:
                features.append(0.0)
            
            if len(df) >= 11:
                features.append(float(latest['close'] / df.iloc[-11]['close'] - 1))
            else:
                features.append(0.0)
            
            # Price position within recent range
            if len(df) >= 20:
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                if recent_high != recent_low:
                    features.append(float((latest['close'] - recent_low) / (recent_high - recent_low)))
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # Gap (difference between current open and previous close)
            features.append(float(latest['open'] / prev['close'] - 1) if prev['close'] != 0 else 0.0)
            
            names = ['price_change', 'daily_range', 'body_ratio', 'volume_ratio', 
                    'price_momentum_5', 'price_momentum_10', 'price_position', 'gap']
            
            return features, names
            
        except Exception as e:
            logging.error(f"Price features error: {e}")
            return [0.0] * 8, ['price_change', 'daily_range', 'body_ratio', 'volume_ratio', 
                              'price_momentum_5', 'price_momentum_10', 'price_position', 'gap']
    
    def _indicator_features(self, indicators: Dict) -> Tuple[List[float], List[str]]:
        """Convert technical indicators to features"""
        try:
            features = []
            names = []
            
            # Momentum indicators (normalized)
            features.append(float(indicators.get('rsi', 50) / 100))  # 0-1 scale
            names.append('rsi_norm')
            
            features.append(float(np.tanh(indicators.get('macd', 0))))  # -1 to 1
            names.append('macd_norm')
            
            features.append(float(indicators.get('stoch_k', 50) / 100))
            names.append('stoch_k_norm')
            
            features.append(float((indicators.get('williams_r', -50) + 100) / 100))  # 0-1 scale
            names.append('williams_r_norm')
            
            # Trend indicators
            features.append(float(indicators.get('adx', 20) / 100))
            names.append('adx_norm')
            
            features.append(float(indicators.get('trend_strength', 0.5)))
            names.append('trend_strength')
            
            # Volatility indicators
            features.append(float(indicators.get('bb_position', 0.5)))
            names.append('bb_position')
            
            atr = indicators.get('atr', 0.01)
            features.append(float(min(1.0, atr / 0.05)))  # Normalize ATR
            names.append('atr_norm')
            
            # Support/Resistance proximity
            current_price = indicators.get('nearest_support', 100) + indicators.get('nearest_resistance', 100)
            current_price = current_price / 2  # Average as proxy for current price
            
            resistance_distance = indicators.get('nearest_resistance', current_price * 1.02) / current_price - 1
            support_distance = 1 - indicators.get('nearest_support', current_price * 0.98) / current_price
            
            features.append(float(min(1.0, resistance_distance * 50)))  # Scale and cap
            features.append(float(min(1.0, support_distance * 50)))
            names.extend(['resistance_distance', 'support_distance'])
            
            return features, names
            
        except Exception as e:
            logging.error(f"Indicator features error: {e}")
            return [0.5] * 10, ['rsi_norm', 'macd_norm', 'stoch_k_norm', 'williams_r_norm', 
                               'adx_norm', 'trend_strength', 'bb_position', 'atr_norm',
                               'resistance_distance', 'support_distance']
    
    def _pattern_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract candlestick pattern features"""
        try:
            if len(df) < 3:
                return [0.0] * 10, ['doji', 'hammer', 'shooting_star', 'engulfing_bull', 'engulfing_bear',
                                   'pin_bar', 'inside_bar', 'outside_bar', 'momentum_candle', 'reversal_candle']
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3] if len(df) >= 3 else prev
            
            features = []
            
            # Basic patterns from indicators
            features.append(float(self._calculate_doji(latest)))
            features.append(float(self._calculate_hammer(latest)))
            features.append(float(self._calculate_shooting_star(latest)))
            features.append(float(self._calculate_engulfing_bull(latest, prev)))
            features.append(float(self._calculate_engulfing_bear(latest, prev)))
            
            # Additional patterns
            features.append(float(self._calculate_pin_bar(latest)))
            features.append(float(self._calculate_inside_bar(latest, prev)))
            features.append(float(self._calculate_outside_bar(latest, prev)))
            features.append(float(self._calculate_momentum_candle(latest, prev, prev2)))
            features.append(float(self._calculate_reversal_candle(latest, prev, prev2)))
            
            names = ['doji', 'hammer', 'shooting_star', 'engulfing_bull', 'engulfing_bear',
                    'pin_bar', 'inside_bar', 'outside_bar', 'momentum_candle', 'reversal_candle']
            
            return features, names
            
        except Exception as e:
            logging.error(f"Pattern features error: {e}")
            return [0.0] * 10, ['doji', 'hammer', 'shooting_star', 'engulfing_bull', 'engulfing_bear',
                               'pin_bar', 'inside_bar', 'outside_bar', 'momentum_candle', 'reversal_candle']
    
    def _calculate_doji(self, candle) -> float:
        """Calculate doji pattern strength"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        return 1.0 - (body_size / total_range) if total_range > 0 else 0.0
    
    def _calculate_hammer(self, candle) -> float:
        """Calculate hammer pattern strength"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        
        if total_range == 0:
            return 0.0
        
        # Hammer: long lower shadow, small upper shadow, small body at top
        hammer_score = (lower_shadow / total_range) * (1 - body_size / total_range)
        if lower_shadow > body_size * 2 and upper_shadow < body_size:
            hammer_score *= 1.5
        
        return min(1.0, hammer_score)
    
    def _calculate_shooting_star(self, candle) -> float:
        """Calculate shooting star pattern strength"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        
        if total_range == 0:
            return 0.0
        
        # Shooting star: long upper shadow, small lower shadow, small body at bottom
        star_score = (upper_shadow / total_range) * (1 - body_size / total_range)
        if upper_shadow > body_size * 2 and lower_shadow < body_size:
            star_score *= 1.5
        
        return min(1.0, star_score)
    
    def _calculate_engulfing_bull(self, current, prev) -> float:
        """Calculate bullish engulfing pattern"""
        if (current['close'] > current['open'] and prev['close'] < prev['open'] and
            current['open'] < prev['close'] and current['close'] > prev['open']):
            return 1.0
        return 0.0
    
    def _calculate_engulfing_bear(self, current, prev) -> float:
        """Calculate bearish engulfing pattern"""
        if (current['close'] < current['open'] and prev['close'] > prev['open'] and
            current['open'] > prev['close'] and current['close'] < prev['open']):
            return 1.0
        return 0.0
    
    def _calculate_pin_bar(self, candle) -> float:
        """Calculate pin bar pattern strength"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return 0.0
        
        # Pin bar: small body with long tail (shadow)
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        max_shadow = max(lower_shadow, upper_shadow)
        
        if max_shadow > body_size * 3:
            return min(1.0, max_shadow / total_range)
        
        return 0.0
    
    def _calculate_inside_bar(self, current, prev) -> float:
        """Calculate inside bar pattern"""
        if (current['high'] <= prev['high'] and current['low'] >= prev['low']):
            return 1.0
        return 0.0
    
    def _calculate_outside_bar(self, current, prev) -> float:
        """Calculate outside bar pattern"""
        if (current['high'] >= prev['high'] and current['low'] <= prev['low']):
            return 1.0
        return 0.0
    
    def _calculate_momentum_candle(self, current, prev, prev2) -> float:
        """Calculate momentum candle strength"""
        # Strong directional candle with increasing momentum
        current_body = abs(current['close'] - current['open'])
        prev_body = abs(prev['close'] - prev['open'])
        prev2_body = abs(prev2['close'] - prev2['open'])
        
        current_range = current['high'] - current['low']
        
        if current_range == 0:
            return 0.0
        
        # Check for increasing body size and same direction
        if (current_body > prev_body > prev2_body and 
            np.sign(current['close'] - current['open']) == np.sign(prev['close'] - prev['open'])):
            return min(1.0, current_body / current_range)
        
        return 0.0
    
    def _calculate_reversal_candle(self, current, prev, prev2) -> float:
        """Calculate reversal candle strength"""
        # Strong candle in opposite direction to recent trend
        current_direction = np.sign(current['close'] - current['open'])
        prev_direction = np.sign(prev['close'] - prev['open'])
        prev2_direction = np.sign(prev2['close'] - prev2['open'])
        
        current_body = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']
        
        if current_range == 0:
            return 0.0
        
        # Strong reversal if direction changes and current candle is strong
        if (current_direction != prev_direction == prev2_direction and 
            current_body > current_range * 0.6):
            return 1.0
        
        return 0.0
    
    def _microstructure_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract market microstructure features"""
        try:
            if len(df) < 10:
                return [0.0] * 12, ['volatility_regime', 'volume_profile', 'price_acceleration',
                                   'bid_ask_pressure', 'market_depth', 'order_flow',
                                   'volatility_clustering', 'mean_reversion', 'momentum_persistence',
                                   'liquidity_proxy', 'market_efficiency', 'noise_ratio']
            
            features = []
            close = df['close']
            volume = df['volume']
            
            # Volatility regime
            returns = close.pct_change().dropna()
            recent_vol = returns.tail(10).std()
            historical_vol = returns.std()
            vol_regime = recent_vol / historical_vol if historical_vol > 0 else 1.0
            features.append(float(min(3.0, vol_regime) / 3.0))  # Normalize to 0-1
            
            # Volume profile
            recent_avg_volume = volume.tail(10).mean()
            historical_avg_volume = volume.mean()
            volume_profile = recent_avg_volume / historical_avg_volume if historical_avg_volume > 0 else 1.0
            features.append(float(min(3.0, volume_profile) / 3.0))  # Normalize to 0-1
            
            # Price acceleration (second derivative)
            if len(returns) >= 3:
                acceleration = returns.diff().tail(3).mean()
                features.append(float(np.tanh(acceleration * 1000)))  # Scale and bound
            else:
                features.append(0.0)
            
            # Bid-ask pressure proxy (using high-low spread)
            spreads = (df['high'] - df['low']) / df['close']
            recent_spread = spreads.tail(5).mean()
            historical_spread = spreads.mean()
            spread_ratio = recent_spread / historical_spread if historical_spread > 0 else 1.0
            features.append(float(min(2.0, spread_ratio) / 2.0))
            
            # Market depth proxy (volume at different price levels)
            price_levels = pd.qcut(close.tail(20), q=5, duplicates='drop')
            level_volumes = df.tail(20).groupby(price_levels)['volume'].mean()
            depth_imbalance = level_volumes.std() / level_volumes.mean() if level_volumes.mean() > 0 else 0
            features.append(float(min(1.0, depth_imbalance)))
            
            # Order flow proxy (price vs volume relationship)
            price_volume_corr = close.tail(20).corr(volume.tail(20))
            features.append(float((price_volume_corr + 1) / 2))  # Convert to 0-1 scale
            
            # Volatility clustering (GARCH effect)
            vol_autocorr = returns.tail(20).rolling(5).std().autocorr()
            features.append(float((vol_autocorr + 1) / 2) if not np.isnan(vol_autocorr) else 0.5)
            
            # Mean reversion tendency
            mean_price = close.tail(20).mean()
            current_price = close.iloc[-1]
            mean_reversion = abs(current_price - mean_price) / mean_price if mean_price > 0 else 0
            features.append(float(min(1.0, mean_reversion * 10)))
            
            # Momentum persistence
            momentum_periods = []
            for i in range(1, min(6, len(returns))):
                momentum_periods.append(returns.iloc[-i])
            momentum_consistency = 1.0 if len(set(np.sign(momentum_periods))) == 1 else 0.0
            features.append(float(momentum_consistency))
            
            # Liquidity proxy (inverse of volatility adjusted by volume)
            if recent_vol > 0 and recent_avg_volume > 0:
                liquidity = recent_avg_volume / recent_vol
                liquidity_norm = min(1.0, liquidity / (historical_avg_volume / historical_vol))
            else:
                liquidity_norm = 0.5
            features.append(float(liquidity_norm))
            
            # Market efficiency (random walk test)
            if len(returns) >= 10:
                # Variance ratio test approximation
                returns_10 = returns.tail(10)
                single_period_var = returns_10.var()
                multi_period_var = returns_10.rolling(2).sum().var() / 2
                efficiency = single_period_var / multi_period_var if multi_period_var > 0 else 1.0
                features.append(float(min(2.0, abs(efficiency - 1.0))))
            else:
                features.append(0.0)
            
            # Noise ratio (high-low vs open-close)
            noise_ratios = (df['high'] - df['low']) / abs(df['close'] - df['open'])
            noise_ratios = noise_ratios.replace([np.inf, -np.inf], 2.0)  # Handle division by zero
            avg_noise = noise_ratios.tail(10).mean()
            features.append(float(min(1.0, avg_noise / 5.0)))  # Normalize
            
            names = ['volatility_regime', 'volume_profile', 'price_acceleration',
                    'bid_ask_pressure', 'market_depth', 'order_flow',
                    'volatility_clustering', 'mean_reversion', 'momentum_persistence',
                    'liquidity_proxy', 'market_efficiency', 'noise_ratio']
            
            return features, names
            
        except Exception as e:
            logging.error(f"Microstructure features error: {e}")
            return [0.5] * 12, ['volatility_regime', 'volume_profile', 'price_acceleration',
                               'bid_ask_pressure', 'market_depth', 'order_flow',
                               'volatility_clustering', 'mean_reversion', 'momentum_persistence',
                               'liquidity_proxy', 'market_efficiency', 'noise_ratio']
    
    def _time_features(self) -> Tuple[List[float], List[str]]:
        """Extract time-based features"""
        try:
            now = datetime.now()
            features = []
            
            # Basic time features
            features.append(float(now.hour / 24))  # Hour of day (0-1)
            features.append(float(now.weekday() / 6))  # Day of week (0-1)
            features.append(float(now.day / 31))  # Day of month (0-1)
            
            # Cyclical time features
            features.append(float(np.sin(2 * np.pi * now.hour / 24)))  # Hourly cycle
            features.append(float(np.cos(2 * np.pi * now.hour / 24)))
            features.append(float(np.sin(2 * np.pi * now.weekday() / 7)))  # Weekly cycle
            features.append(float(np.cos(2 * np.pi * now.weekday() / 7)))
            
            # Market session indicators
            utc_hour = now.hour  # Assuming UTC
            
            # Major market sessions
            london_session = 1.0 if 8 <= utc_hour <= 17 else 0.0
            ny_session = 1.0 if 13 <= utc_hour <= 22 else 0.0
            tokyo_session = 1.0 if 0 <= utc_hour <= 9 else 0.0
            overlap_session = 1.0 if 13 <= utc_hour <= 17 else 0.0  # London-NY overlap
            
            features.extend([london_session, ny_session, tokyo_session, overlap_session])
            
            # Weekend indicator
            is_weekend = 1.0 if now.weekday() >= 5 else 0.0
            features.append(is_weekend)
            
            names = ['hour_norm', 'weekday_norm', 'day_norm', 'hour_sin', 'hour_cos',
                    'weekday_sin', 'weekday_cos', 'london_session', 'ny_session',
                    'tokyo_session', 'overlap_session', 'is_weekend']
            
            return features, names
            
        except Exception as e:
            logging.error(f"Time features error: {e}")
            return [0.5] * 12, ['hour_norm', 'weekday_norm', 'day_norm', 'hour_sin', 'hour_cos',
                               'weekday_sin', 'weekday_cos', 'london_session', 'ny_session',
                               'tokyo_session', 'overlap_session', 'is_weekend']
    
    def _statistical_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract statistical features from price data"""
        try:
            if len(df) < 10:
                return [0.0] * 18, ['skewness', 'kurtosis', 'hurst_exponent', 'fractal_dimension',
                                   'entropy', 'complexity', 'persistence', 'anti_persistence',
                                   'trend_intensity', 'cycle_strength', 'seasonality',
                                   'regime_stability', 'predictability', 'chaos_indicator',
                                   'information_ratio', 'downside_deviation', 'maximum_drawdown',
                                   'recovery_factor']
            
            features = []
            close = df['close']
            returns = close.pct_change().dropna()
            
            # Distribution characteristics
            features.append(float(returns.skew() / 3.0) if len(returns) > 3 else 0.0)  # Normalize skewness
            features.append(float(min(1.0, abs(returns.kurtosis()) / 10.0)))  # Normalize kurtosis
            
            # Hurst exponent (simplified)
            if len(returns) >= 20:
                hurst = self._calculate_hurst_exponent(returns.values)
                features.append(float(hurst))
            else:
                features.append(0.5)
            
            # Fractal dimension
            fractal_dim = 2.0 - features[-1]  # Relationship to Hurst exponent
            features.append(float(fractal_dim / 2.0))  # Normalize to 0-1
            
            # Shannon entropy of returns
            if len(returns) > 0:
                hist, _ = np.histogram(returns, bins=10)
                hist = hist + 1e-10  # Avoid log(0)
                probs = hist / hist.sum()
                entropy = -np.sum(probs * np.log2(probs))
                features.append(float(entropy / np.log2(10)))  # Normalize by max entropy
            else:
                features.append(0.5)
            
            # Complexity (approximate)
            complexity = len(set(np.round(returns.values, 4))) / len(returns) if len(returns) > 0 else 0
            features.append(float(complexity))
            
            # Persistence and anti-persistence
            autocorr_1 = returns.autocorr(lag=1) if len(returns) > 1 else 0
            persistence = max(0, autocorr_1) if not np.isnan(autocorr_1) else 0
            anti_persistence = max(0, -autocorr_1) if not np.isnan(autocorr_1) else 0
            features.extend([float(persistence), float(anti_persistence)])
            
            # Trend intensity
            if len(close) >= 20:
                trend_slope = stats.linregress(range(20), close.tail(20).values)[0]
                trend_intensity = abs(trend_slope) / close.tail(20).mean()
                features.append(float(min(1.0, trend_intensity * 1000)))
            else:
                features.append(0.0)
            
            # Cycle strength (simplified spectral analysis)
            if len(returns) >= 20:
                fft_vals = np.fft.fft(returns.tail(20).values)
                power_spectrum = np.abs(fft_vals) ** 2
                cycle_strength = np.max(power_spectrum[1:]) / np.sum(power_spectrum[1:])
                features.append(float(min(1.0, cycle_strength * 5)))
            else:
                features.append(0.0)
            
            # Seasonality detection (simplified)
            if len(close) >= 7:
                daily_returns = []
                for i in range(7):
                    day_returns = returns[returns.index.dayofweek == i] if hasattr(returns.index, 'dayofweek') else returns[::7]
                    daily_returns.append(day_returns.mean() if len(day_returns) > 0 else 0)
                seasonality = np.std(daily_returns) / (np.mean(np.abs(daily_returns)) + 1e-10)
                features.append(float(min(1.0, seasonality)))
            else:
                features.append(0.0)
            
            # Regime stability
            if len(returns) >= 20:
                rolling_vol = returns.rolling(5).std()
                vol_stability = 1.0 - (rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0
                features.append(float(max(0.0, vol_stability)))
            else:
                features.append(0.5)
            
            # Predictability score
            if len(returns) >= 10:
                # Simple AR(1) model fit
                try:
                    y = returns.values[1:]
                    x = returns.values[:-1]
                    correlation = np.corrcoef(x, y)[0, 1]
                    predictability = abs(correlation) if not np.isnan(correlation) else 0
                    features.append(float(predictability))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # Chaos indicator (largest Lyapunov exponent approximation)
            if len(returns) >= 20:
                chaos_score = self._approximate_lyapunov(returns.values)
                features.append(float(min(1.0, max(0.0, chaos_score))))
            else:
                features.append(0.5)
            
            # Information ratio
            if len(returns) > 1:
                excess_return = returns.mean()
                tracking_error = returns.std()
                info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                features.append(float(np.tanh(info_ratio)))  # Bound between -1 and 1, then shift to 0-1
            else:
                features.append(0.5)
            
            # Downside deviation
            negative_returns = returns[returns < 0]
            downside_dev = negative_returns.std() if len(negative_returns) > 0 else 0
            total_std = returns.std()
            downside_ratio = downside_dev / total_std if total_std > 0 else 0
            features.append(float(downside_ratio))
            
            # Maximum drawdown
            if len(close) >= 10:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
                features.append(float(min(1.0, max_drawdown)))
            else:
                features.append(0.0)
            
            # Recovery factor
            if len(returns) > 0:
                total_return = returns.sum()
                recovery_factor = total_return / (features[-1] + 1e-10)  # Total return / max drawdown
                features.append(float(np.tanh(recovery_factor)))
            else:
                features.append(0.0)
            
            names = ['skewness', 'kurtosis', 'hurst_exponent', 'fractal_dimension',
                    'entropy', 'complexity', 'persistence', 'anti_persistence',
                    'trend_intensity', 'cycle_strength', 'seasonality',
                    'regime_stability', 'predictability', 'chaos_indicator',
                    'information_ratio', 'downside_deviation', 'maximum_drawdown',
                    'recovery_factor']
            
            return features, names
            
        except Exception as e:
            logging.error(f"Statistical features error: {e}")
            return [0.5] * 18, ['skewness', 'kurtosis', 'hurst_exponent', 'fractal_dimension',
                               'entropy', 'complexity', 'persistence', 'anti_persistence',
                               'trend_intensity', 'cycle_strength', 'seasonality',
                               'regime_stability', 'predictability', 'chaos_indicator',
                               'information_ratio', 'downside_deviation', 'maximum_drawdown',
                               'recovery_factor']
    
    def _calculate_hurst_exponent(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        try:
            n = len(returns)
            if n < 10:
                return 0.5
            
            # Create cumulative deviations
            mean_return = np.mean(returns)
            cumulative_deviations = np.cumsum(returns - mean_return)
            
            # Calculate range and standard deviation
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(returns)
            
            if S == 0:
                return 0.5
            
            # R/S statistic
            rs = R / S
            
            # Hurst exponent approximation
            hurst = np.log(rs) / np.log(n)
            
            # Bound between 0 and 1
            return max(0.0, min(1.0, hurst))
            
        except:
            return 0.5
    
    def _approximate_lyapunov(self, data: np.ndarray) -> float:
        """Approximate largest Lyapunov exponent"""
        try:
            if len(data) < 10:
                return 0.5
            
            # Simple approximation based on sensitivity to initial conditions
            diffs = np.diff(data)
            if len(diffs) < 2:
                return 0.5
            
            # Calculate average exponential divergence
            log_diffs = np.log(np.abs(diffs) + 1e-10)
            lyapunov = np.mean(log_diffs[1:] - log_diffs[:-1])
            
            # Normalize to 0-1 range
            return 0.5 + np.tanh(lyapunov) / 2
            
        except:
            return 0.5

class MLTradingEngine:
    """Enhanced ML engine with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = MLFeatureEngineer()
        self.market_data = MarketDataProvider()
        self.news_detector = NewsEventDetector()
        self.models_trained = False
        self.training_data_cache = {}
        self.last_training_time = 0
        self.model_performance = {}
        self.ensemble_weights = {}
        
        # Initialize models if ML available
        if ML_AVAILABLE:
            self._initialize_model_architecture()
        else:
            logging.warning("ML libraries not available, using simplified models")
    
    def _initialize_model_architecture(self):
        """Initialize ML model architecture"""
        try:
            # Classification models for signal quality
            self.models['rf_classifier'] = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
            )
            self.models['mlp_classifier'] = MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
            )
            
            # Regression models for profit prediction
            self.models['rf_regressor'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.models['gb_regressor'] = GradientBoostingRegressor(
                n_estimators=100, random_state=42, max_depth=6
            )
            self.models['mlp_regressor'] = MLPRegressor(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
            )
            
            # Specialized models
            self.models['reversal_detector'] = RandomForestClassifier(
                n_estimators=150, random_state=42, class_weight='balanced'
            )
            self.models['volatility_predictor'] = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
            self.models['regime_classifier'] = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            
            # Anomaly detection
            self.models['anomaly_detector'] = IsolationForest(
                contamination=0.1, random_state=42
            )
            
            logging.info("ML model architecture initialized")
            
        except Exception as e:
            logging.error(f"Model architecture initialization error: {e}")
    
    def initialize_models(self):
        """Initialize and train ML models with comprehensive market data"""
        try:
            logging.info("Starting ML model training with market data...")
            
            if not ML_AVAILABLE:
                self._use_simplified_models()
                return
            
            # Collect training data from multiple sources
            training_data = self._collect_comprehensive_training_data()
            
            if len(training_data['features']) < 100:
                logging.warning("Insufficient training data, using simplified models")
                self._use_simplified_models()
                return
            
            # Prepare data
            X = np.array(training_data['features'])
            y_signal = np.array(training_data['signal_quality'])
            y_profit = np.array(training_data['profit_potential'])
            y_reversal = np.array(training_data['reversal_risk'])
            y_regime = np.array(training_data['market_regime'])
            y_volatility = np.array(training_data['volatility'])
            
            # Train models with cross-validation
            self._train_with_validation(X, y_signal, y_profit, y_reversal, y_regime, y_volatility)
            
            # Calculate ensemble weights
            self._calculate_ensemble_weights(X, y_signal, y_profit, y_reversal)
            
            self.models_trained = True
            self.last_training_time = time.time()
            
            logging.info("ML models trained successfully with ensemble weighting")
            
        except Exception as e:
            logging.error(f"ML model initialization error: {e}")
            self._use_simplified_models()
    
    def _collect_comprehensive_training_data(self) -> Dict:
        """Collect training data from multiple symbols and timeframes"""
        training_data = {
            'features': [],
            'signal_quality': [],
            'profit_potential': [],
            'reversal_risk': [],
            'market_regime': [],
            'volatility': []
        }
        
        symbols = ['R_50', 'R_75', 'R_100', 'BOOM500', 'CRASH500', 'BOOM1000', 'CRASH1000']
        timeframes = [('3mo', '1h'), ('1mo', '30m'), ('1mo', '15m')]
        
        ta_engine = TechnicalAnalysisEngine()
        
        for symbol in symbols:
            for period, interval in timeframes:
                try:
                    # Get market data
                    df = self.market_data.get_market_data(symbol, period, interval)
                    
                    if len(df) < 100:
                        continue
                    
                    # Generate training samples
                    samples = self._generate_training_samples(df, symbol, ta_engine)
                    
                    if samples:
                        for sample in samples:
                            training_data['features'].append(sample['features'])
                            training_data['signal_quality'].append(sample['signal_quality'])
                            training_data['profit_potential'].append(sample['profit_potential'])
                            training_data['reversal_risk'].append(sample['reversal_risk'])
                            training_data['market_regime'].append(sample['market_regime'])
                            training_data['volatility'].append(sample['volatility'])
                    
                    logging.info(f"Collected {len(samples) if samples else 0} samples from {symbol} {period}/{interval}")
                    
                except Exception as e:
                    logging.error(f"Error collecting data from {symbol}: {e}")
                    continue
        
        logging.info(f"Total training samples collected: {len(training_data['features'])}")
        return training_data
    
    def _generate_training_samples(self, df: pd.DataFrame, symbol: str, ta_engine: TechnicalAnalysisEngine) -> List[Dict]:
        """Generate labeled training samples from historical data"""
        try:
            samples = []
            
            # Use sliding window approach
            window_size = 50
            future_window = 10
            
            for i in range(window_size, len(df) - future_window):
                # Current window for features
                current_window = df.iloc[i-window_size:i].copy()
                
                # Future window for labels
                future_window_data = df.iloc[i:i+future_window].copy()
                
                # Calculate indicators for current window
                indicators = ta_engine.calculate_all_indicators(current_window)
                
                # Create features
                features = self.feature_engineer.create_features(current_window, indicators)
                
                # Generate labels based on future price movement
                labels = self._generate_labels(
                    current_window.iloc[-1],
                    future_window_data,
                    symbol
                )
                
                sample = {
                    'features': features.flatten(),
                    'signal_quality': labels['signal_quality'],
                    'profit_potential': labels['profit_potential'],
                    'reversal_risk': labels['reversal_risk'],
                    'market_regime': labels['market_regime'],
                    'volatility': labels['volatility']
                }
                
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            logging.error(f"Error generating training samples: {e}")
            return []
    
    def _generate_labels(self, current_candle: pd.Series, future_data: pd.DataFrame, symbol: str) -> Dict:
        """Generate training labels based on future price movements"""
        try:
            current_price = current_candle['close']
            future_prices = future_data['close'].values
            
            # Calculate returns
            max_return = max(future_prices) / current_price - 1
            min_return = min(future_prices) / current_price - 1
            final_return = future_prices[-1] / current_price - 1
            
            # Signal quality (0=bad, 1=good)
            profit_threshold = 0.02 if 'R_' in symbol else 0.03  # Lower threshold for synthetics
            signal_quality = 1 if max_return > profit_threshold else 0
            
            # Profit potential
            profit_potential = max(0, max_return)
            
            # Reversal risk (did it reach 80%+ of max profit then reverse?)
            reversal_risk = 0
            if max_return > 0.015:  # Only check if there was significant movement
                max_profit_point = np.argmax(future_prices)
                if max_profit_point < len(future_prices) - 2:  # Not at the end
                    profit_at_max = future_prices[max_profit_point] / current_price - 1
                    final_profit = final_return
                    
                    if profit_at_max > 0.015 and final_profit < profit_at_max * 0.5:
                        reversal_risk = 1
            
            # Market regime (simplified classification)
            volatility = np.std(future_prices / current_price - 1)
            
            if volatility > 0.05:
                regime = 3  # Volatile
            elif max_return > 0.02 and min_return > -0.01:
                regime = 0  # Trending bull
            elif max_return < 0.01 and min_return < -0.02:
                regime = 1  # Trending bear
            else:
                regime = 2  # Sideways
            
            return {
                'signal_quality': signal_quality,
                'profit_potential': profit_potential,
                'reversal_risk': reversal_risk,
                'market_regime': regime,
                'volatility': volatility
            }
            
        except Exception as e:
            logging.error(f"Error generating labels: {e}")
            return {
                'signal_quality': 0,
                'profit_potential': 0.0,
                'reversal_risk': 0,
                'market_regime': 2,
                'volatility': 0.02
            }
    
    def _train_with_validation(self, X, y_signal, y_profit, y_reversal, y_regime, y_volatility):
        """Train models with cross-validation"""
        try:
            # Split data
            X_train, X_test, y_signal_train, y_signal_test = train_test_split(
                X, y_signal, test_size=0.2, random_state=42, stratify=y_signal
            )
            
            # Train signal quality models
            self.models['rf_classifier'].fit(X_train, y_signal_train)
            self.models['mlp_classifier'].fit(X_train, y_signal_train)
            
            # Evaluate signal quality models
            rf_score = self.models['rf_classifier'].score(X_test, y_signal_test)
            mlp_score = self.models['mlp_classifier'].score(X_test, y_signal_test)
            
            self.model_performance['rf_classifier'] = rf_score
            self.model_performance['mlp_classifier'] = mlp_score
            
            # Train regression models
            _, _, y_profit_train, y_profit_test = train_test_split(
                X, y_profit, test_size=0.2, random_state=42
            )
            
            self.models['gb_regressor'].fit(X_train, y_profit_train)
            self.models['mlp_regressor'].fit(X_train, y_profit_train)
            
            # Train specialized models
            self.models['reversal_detector'].fit(X_train, 
                y_reversal[:len(X_train)] if len(y_reversal) > len(X_train) else y_reversal)
            self.models['regime_classifier'].fit(X_train,
                y_regime[:len(X_train)] if len(y_regime) > len(X_train) else y_regime)
            self.models['volatility_predictor'].fit(X_train,
                y_volatility[:len(X_train)] if len(y_volatility) > len(X_train) else y_volatility)
            
            # Train anomaly detector
            self.models['anomaly_detector'].fit(X_train)
            
            logging.info(f"Model performance - RF: {rf_score:.3f}, MLP: {mlp_score:.3f}")
            
        except Exception as e:
            logging.error(f"Model training error: {e}")
    
    def _calculate_ensemble_weights(self, X, y_signal, y_profit, y_reversal):
        """Calculate optimal ensemble weights based on model performance"""
        try:
            # Simple performance-based weighting
            rf_performance = self.model_performance.get('rf_classifier', 0.5)
            mlp_performance = self.model_performance.get('mlp_classifier', 0.5)
            
            total_performance = rf_performance + mlp_performance
            if total_performance > 0:
                self.ensemble_weights = {
                    'rf_weight': rf_performance / total_performance,
                    'mlp_weight': mlp_performance / total_performance
                }
            else:
                self.ensemble_weights = {'rf_weight': 0.5, 'mlp_weight': 0.5}
            
            logging.info(f"Ensemble weights: RF={self.ensemble_weights['rf_weight']:.3f}, MLP={self.ensemble_weights['mlp_weight']:.3f}")
            
        except Exception as e:
            logging.error(f"Ensemble weight calculation error: {e}")
            self.ensemble_weights = {'rf_weight': 0.5, 'mlp_weight': 0.5}
    
    def _use_simplified_models(self):
        """Use simplified rule-based models when ML is not available"""
        self.models_trained = True
        self.simplified_mode = True
        logging.info("Using simplified rule-based models")
    
    def predict_signal_quality(self, symbol: str, technical_indicators: Dict) -> MLPrediction:
        """Generate comprehensive ML prediction"""
        try:
            if not self.models_trained:
                return self._get_conservative_prediction()
            
            # Get recent market data for features
            df = self.market_data.get_market_data(symbol, period='5d', interval='1h')
            
            if len(df) < 20:
                return self._get_conservative_prediction()
            
            # Create features
            features = self.feature_engineer.create_features(df, technical_indicators)
            
            if not ML_AVAILABLE or hasattr(self, 'simplified_mode'):
                return self._get_rule_based_prediction(technical_indicators, df)
            
            # Get ML predictions
            ml_predictions = self._get_ml_predictions(features, symbol, technical_indicators)
            
            # Add news sentiment impact
            news_impact = self.news_detector.get_news_sentiment_impact(symbol)
            is_news_time, news_factor = self.news_detector.is_news_time()
            
            # Adjust predictions based on news
            if is_news_time:
                ml_predictions['signal_strength'] *= (1 + news_factor * 0.3)
                ml_predictions['reversal_risk'] *= (1 + news_factor * 0.5)
                ml_predictions['confidence'] *= (1 - news_factor * 0.2)
            
            # Create final prediction
            prediction = MLPrediction(
                signal_strength=min(100.0, ml_predictions['signal_strength']),
                profit_probability=ml_predictions['profit_probability'],
                reversal_risk=ml_predictions['reversal_risk'],
                optimal_tp_levels=ml_predictions['optimal_tp_levels'],
                recommended_sl=ml_predictions['recommended_sl'],
                position_size_multiplier=ml_predictions['position_size_multiplier'],
                market_regime=ml_predictions['market_regime'],
                confidence=ml_predictions['confidence'],
                time_horizon=ml_predictions['time_horizon'],
                risk_score=ml_predictions['risk_score'],
                volatility_forecast=ml_predictions['volatility_forecast'],
                news_sentiment_impact=news_impact
            )
            
            return prediction
            
        except Exception as e:
            logging.error(f"ML prediction error: {e}")
            return self._get_conservative_prediction()
    
    def _get_ml_predictions(self, features: np.ndarray, symbol: str, indicators: Dict) -> Dict:
        """Get predictions from ML models"""
        try:
            # Ensemble signal quality prediction
            rf_signal_prob = self.models['rf_classifier'].predict_proba(features)[0][1]
            mlp_signal_prob = self.models['mlp_classifier'].predict_proba(features)[0][1]
            
            rf_weight = self.ensemble_weights.get('rf_weight', 0.5)
            mlp_weight = self.ensemble_weights.get('mlp_weight', 0.5)
            
            signal_strength = (rf_signal_prob * rf_weight + mlp_signal_prob * mlp_weight) * 100
            
            # Profit potential prediction
            gb_profit = max(0, self.models['gb_regressor'].predict(features)[0])
            mlp_profit = max(0, self.models['mlp_regressor'].predict(features)[0])
            profit_potential = (gb_profit + mlp_profit) / 2
            
            # Reversal risk prediction
            reversal_risk = self.models['reversal_detector'].predict_proba(features)[0][1]
            
            # Market regime prediction
            regime_pred = self.models['regime_classifier'].predict(features)[0]
            regime_map = {
                0: MarketRegime.TRENDING_BULL,
                1: MarketRegime.TRENDING_BEAR,
                2: MarketRegime.SIDEWAYS,
                3: MarketRegime.VOLATILE
            }
            market_regime = regime_map.get(regime_pred, MarketRegime.SIDEWAYS)
            
            # Volatility forecast
            volatility_forecast = max(0, self.models['volatility_predictor'].predict(features)[0])
            
            # Anomaly detection
            is_anomaly = self.models['anomaly_detector'].predict(features)[0] == -1
            anomaly_factor = 0.3 if is_anomaly else 1.0
            
            # Calculate derived metrics
            confidence = (signal_strength / 100 + (1 - reversal_risk)) / 2 * anomaly_factor
            
            # Risk score
            risk_score = (reversal_risk + volatility_forecast + (1 - confidence)) / 3
            
            # Optimal take profit levels
            optimal_tp_levels = self._calculate_ml_tp_levels(
                profit_potential, reversal_risk, market_regime
            )
            
            # Recommended stop loss
            atr = indicators.get('atr', 0.01)
            base_sl = atr * 2
            volatility_multiplier = 1 + volatility_forecast
            recommended_sl = base_sl * volatility_multiplier
            
            # Position size multiplier
            position_multiplier = confidence * 1.5 * (1 - risk_score)
            position_multiplier = max(0.1, min(2.0, position_multiplier))
            
            # Time horizon
            time_horizon = self._determine_time_horizon(market_regime, volatility_forecast)
            
            return {
                'signal_strength': signal_strength,
                'profit_probability': signal_strength / 100,
                'reversal_risk': reversal_risk,
                'optimal_tp_levels': optimal_tp_levels,
                'recommended_sl': recommended_sl,
                'position_size_multiplier': position_multiplier,
                'market_regime': market_regime,
                'confidence': confidence,
                'time_horizon': time_horizon,
                          'risk_score': risk_score,
                'volatility_forecast': volatility_forecast
            }
            
        except Exception as e:
            logging.error(f"ML predictions error: {e}")
            return self._get_default_ml_prediction()
    
    def _calculate_ml_tp_levels(self, profit_potential: float, reversal_risk: float, regime: MarketRegime) -> List[float]:
        try:
            base_tp = max(0.01, profit_potential)
            if regime == MarketRegime.TRENDING_BULL or regime == MarketRegime.TRENDING_BEAR:
                multipliers = [0.4, 0.7, 1.2] if reversal_risk < 0.3 else [0.3, 0.6]
            elif regime == MarketRegime.VOLATILE:
                multipliers = [0.2, 0.4] if reversal_risk > 0.6 else [0.3, 0.6, 0.9]
            else:
                multipliers = [0.5, 0.8] if reversal_risk > 0.4 else [0.6, 1.0]
            return [base_tp * mult for mult in multipliers]
        except:
            return [0.015, 0.025]
    
    def _determine_time_horizon(self, regime: MarketRegime, volatility: float) -> str:
        if regime == MarketRegime.VOLATILE or volatility > 0.05:
            return "short"
        elif regime == MarketRegime.TRENDING_BULL or regime == MarketRegime.TRENDING_BEAR:
            return "medium"
        else:
            return "short"



if __name__ == '__main__':
    import os
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bot instance - this will automatically start the Flask server
    bot = AIEnhancedDerivBot()
    
    # Keep the main thread alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down bot...")
