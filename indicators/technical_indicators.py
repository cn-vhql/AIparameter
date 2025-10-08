#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
技术指标计算模块
基于TA-Lib实现常见技术指标的计算
支持MACD、RSI、KDJ、布林带等指标
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self):
        self.indicators_info = {
            "MACD": {
                "params": ["fastperiod", "slowperiod", "signalperiod"],
                "defaults": [12, 26, 9],
                "description": "移动平均收敛发散指标"
            },
            "RSI": {
                "params": ["timeperiod"],
                "defaults": [14],
                "description": "相对强弱指数"
            },
            "KDJ": {
                "params": ["fastk_period", "slowk_period", "slowd_period"],
                "defaults": [9, 3, 3],
                "description": "随机指标"
            },
            "布林带": {
                "params": ["timeperiod", "nbdevup", "nbdevdn"],
                "defaults": [20, 2, 2],
                "description": "布林带指标"
            },
            "均线": {
                "params": ["timeperiod"],
                "defaults": [20],
                "description": "移动平均线"
            },
            "乖离率": {
                "params": ["timeperiod"],
                "defaults": [20],
                "description": "乖离率指标"
            }
        }
    
    def _ensure_float64(self, data):
        """确保数据为float64类型（TA-Lib要求）"""
        return np.asarray(data, dtype=np.float64)
    
    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str, 
                          params: Dict[str, int] = None) -> pd.DataFrame:
        """
        计算指定技术指标
        """
        if indicator_name not in self.indicators_info:
            logger.error(f"不支持的指标: {indicator_name}")
            return df
        
        # 使用默认参数如果未提供
        if params is None:
            param_names = self.indicators_info[indicator_name]["params"]
            default_values = self.indicators_info[indicator_name]["defaults"]
            params = dict(zip(param_names, default_values))
        
        try:
            if indicator_name == "MACD":
                return self._calculate_macd(df, **params)
            elif indicator_name == "RSI":
                return self._calculate_rsi(df, **params)
            elif indicator_name == "KDJ":
                return self._calculate_kdj(df, **params)
            elif indicator_name == "布林带":
                return self._calculate_bollinger_bands(df, **params)
            elif indicator_name == "均线":
                return self._calculate_moving_average(df, **params)
            elif indicator_name == "乖离率":
                return self._calculate_bias(df, **params)
            else:
                return df
                
        except Exception as e:
            logger.error(f"计算指标 {indicator_name} 时发生错误: {str(e)}")
            return df
    
    def _calculate_macd(self, df: pd.DataFrame, fastperiod: int = 12, 
                       slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """计算MACD指标"""
        # 确保数据为float64类型
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, 
            fastperiod=fastperiod, 
            slowperiod=slowperiod, 
            signalperiod=signalperiod
        )
        
        # 添加到DataFrame
        df = df.copy()
        df[f'macd_{fastperiod}_{slowperiod}_{signalperiod}'] = macd
        df[f'macd_signal_{fastperiod}_{slowperiod}_{signalperiod}'] = macd_signal
        df[f'macd_hist_{fastperiod}_{slowperiod}_{signalperiod}'] = macd_hist
        
        # 生成交易信号
        df['macd_signal'] = 0
        df.loc[macd > macd_signal, 'macd_signal'] = 1  # 金叉买入信号
        df.loc[macd < macd_signal, 'macd_signal'] = -1  # 死叉卖出信号
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        # 确保数据为float64类型
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算RSI
        rsi = talib.RSI(close_prices, timeperiod=timeperiod)
        
        # 添加到DataFrame
        df = df.copy()
        df[f'rsi_{timeperiod}'] = rsi
        
        # 生成交易信号
        df['rsi_signal'] = 0
        df.loc[rsi < 30, 'rsi_signal'] = 1  # 超卖买入信号
        df.loc[rsi > 70, 'rsi_signal'] = -1  # 超买卖出信号
        
        return df
    
    def _calculate_kdj(self, df: pd.DataFrame, fastk_period: int = 9,
                      slowk_period: int = 3, slowd_period: int = 3) -> pd.DataFrame:
        """计算KDJ指标"""
        # 确保数据为float64类型
        high_prices = self._ensure_float64(df['high'].values)
        low_prices = self._ensure_float64(df['low'].values)
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算KDJ
        slowk, slowd = talib.STOCH(
            high_prices, low_prices, close_prices,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )
        
        # 计算J值
        j_values = 3 * slowk - 2 * slowd
        
        # 添加到DataFrame
        df = df.copy()
        df[f'k_{fastk_period}_{slowk_period}_{slowd_period}'] = slowk
        df[f'd_{fastk_period}_{slowk_period}_{slowd_period}'] = slowd
        df[f'j_{fastk_period}_{slowk_period}_{slowd_period}'] = j_values
        
        # 生成交易信号
        df['kdj_signal'] = 0
        # K线上穿D线且J值小于20为买入信号
        buy_condition = (slowk > slowd) & (slowk.shift(1) <= slowd.shift(1)) & (j_values < 20)
        # K线下穿D线且J值大于80为卖出信号
        sell_condition = (slowk < slowd) & (slowk.shift(1) >= slowd.shift(1)) & (j_values > 80)
        
        df.loc[buy_condition, 'kdj_signal'] = 1
        df.loc[sell_condition, 'kdj_signal'] = -1
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, timeperiod: int = 20, 
                                 nbdevup: int = 2, nbdevdn: int = 2) -> pd.DataFrame:
        """计算布林带指标"""
        # 确保数据为float64类型
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算布林带
        upperband, middleband, lowerband = talib.BBANDS(
            close_prices,
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=0
        )
        
        # 添加到DataFrame
        df = df.copy()
        df[f'bb_upper_{timeperiod}_{nbdevup}'] = upperband
        df[f'bb_middle_{timeperiod}_{nbdevup}'] = middleband
        df[f'bb_lower_{timeperiod}_{nbdevdn}'] = lowerband

        # 生成交易信号
        df['bb_signal'] = 0
        # 价格跌破下轨为买入信号
        df.loc[close_prices < lowerband, 'bb_signal'] = 1
        # 价格突破上轨为卖出信号
        df.loc[close_prices > upperband, 'bb_signal'] = -1
        
        return df
    
    def _calculate_moving_average(self, df: pd.DataFrame, timeperiod: int = 20) -> pd.DataFrame:
        """计算移动平均线"""
        # 确保数据为float64类型
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算移动平均线
        ma = talib.SMA(close_prices, timeperiod=timeperiod)
        
        # 添加到DataFrame
        df = df.copy()
        df[f'ma_{timeperiod}'] = ma
        
        # 生成交易信号（价格上穿/下穿均线）
        df['ma_signal'] = 0
        df.loc[close_prices > ma, 'ma_signal'] = 1  # 价格上穿均线
        df.loc[close_prices < ma, 'ma_signal'] = -1  # 价格下穿均线
        
        return df
    
    def _calculate_bias(self, df: pd.DataFrame, timeperiod: int = 20) -> pd.DataFrame:
        """计算乖离率"""
        # 确保数据为float64类型
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算移动平均线
        ma = talib.SMA(close_prices, timeperiod=timeperiod)
        
        # 计算乖离率
        bias = (close_prices - ma) / ma * 100
        
        # 添加到DataFrame
        df = df.copy()
        df[f'bias_{timeperiod}'] = bias
        
        # 生成交易信号
        df['bias_signal'] = 0
        df.loc[bias < -6, 'bias_signal'] = 1  # 负乖离过大，买入信号
        df.loc[bias > 6, 'bias_signal'] = -1  # 正乖离过大，卖出信号
        
        return df
    
    def get_available_indicators(self) -> List[str]:
        """获取支持的指标列表"""
        return list(self.indicators_info.keys())
    
    def get_indicator_info(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """获取指标的参数信息"""
        return self.indicators_info.get(indicator_name)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有支持的指标"""
        result_df = df.copy()
        for indicator in self.get_available_indicators():
            result_df = self.calculate_indicator(result_df, indicator)
        return result_df

# 测试函数
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    close_prices = np.random.normal(100, 10, len(dates)).cumsum() + 1000
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': close_prices * 0.99,
        'high': close_prices * 1.02,
        'low': close_prices * 0.98,
        'close': close_prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    })
    test_df.set_index('date', inplace=True)
    
    # 测试指标计算
    indicator_calc = TechnicalIndicators()
    
    # 测试MACD
    macd_df = indicator_calc.calculate_indicator(test_df, "MACD")
    print("MACD计算完成")
    print(macd_df[['close', 'macd_12_26_9', 'macd_signal_12_26_9']].tail())
    
    # 测试RSI
    rsi_df = indicator_calc.calculate_indicator(test_df, "RSI")
    print("\nRSI计算完成")
    print(rsi_df[['close', 'rsi_14', 'rsi_signal']].tail())
    
    print(f"\n支持的指标: {indicator_calc.get_available_indicators()}")