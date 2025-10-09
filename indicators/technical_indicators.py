#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
技术指标计算模块
基于TA-Lib实现常见技术指标的计算
支持13种技术指标及其参数优化
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
                "params": ["fast_period", "slow_period", "signal_period"],
                "defaults": [12, 26, 9],
                "param_mapping": {
                    "fast_period": "fastperiod",
                    "slow_period": "slowperiod",
                    "signal_period": "signalperiod"
                },
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
            },
            "ATR": {
                "params": ["timeperiod"],
                "defaults": [14],
                "description": "平均真实波幅指标"
            },
            "OBV": {
                "params": [],
                "defaults": [],
                "description": "能量潮指标"
            },
            "威廉指标": {
                "params": ["timeperiod"],
                "defaults": [14],
                "description": "威廉超买超卖指标"
            },
            "CCI": {
                "params": ["timeperiod"],
                "defaults": [20],
                "description": "商品通道指数"
            },
            "ADX": {
                "params": ["timeperiod"],
                "defaults": [14],
                "description": "平均方向指数"
            },
            "动量指标": {
                "params": ["timeperiod"],
                "defaults": [10],
                "description": "动量振荡指标"
            },
            "抛物线SAR": {
                "params": ["acceleration", "maximum"],
                "defaults": [0.02, 0.2],
                "description": "抛物线停损指标"
            }
        }
    
    def _ensure_float64(self, data):
        """确保数据为float64类型（TA-Lib要求）"""
        return np.asarray(data, dtype=np.float64)
    
    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str, 
                          params: Dict[str, int] = None) -> pd.DataFrame:
        """
        计算指定技术指标
        
        Args:
            df: 包含价格数据的DataFrame
            indicator_name: 指标名称
            params: 指标参数字典
            
        Returns:
            pd.DataFrame: 添加了技术指标和信号的DataFrame
        """
        if indicator_name not in self.indicators_info:
            logger.error(f"不支持的指标: {indicator_name}")
            return df
        
        # 使用默认参数如果未提供
        indicator_info = self.indicators_info[indicator_name]
        if params is None:
            param_names = indicator_info["params"]
            default_values = indicator_info["defaults"]
            params = dict(zip(param_names, default_values))
            
        try:
            # 如果存在参数映射，转换参数名
            if "param_mapping" in indicator_info:
                mapped_params = {
                    indicator_info["param_mapping"].get(k, k): v 
                    for k, v in params.items()
                }
            else:
                mapped_params = params.copy()
            
            # 确保DataFrame是副本
            df = df.copy()
            
            try:
                # 计算指标
                if indicator_name == "MACD":
                    result_df = self._calculate_macd(df, **mapped_params)
                elif indicator_name == "RSI":
                    result_df = self._calculate_rsi(df, **mapped_params)
                elif indicator_name == "KDJ":
                    result_df = self._calculate_kdj(df, **mapped_params)
                elif indicator_name == "布林带":
                    result_df = self._calculate_bollinger_bands(df, **mapped_params)
                elif indicator_name == "均线":
                    result_df = self._calculate_moving_average(df, **mapped_params)
                elif indicator_name == "乖离率":
                    result_df = self._calculate_bias(df, **mapped_params)
                elif indicator_name == "ATR":
                    result_df = self._calculate_atr(df, **mapped_params)
                elif indicator_name == "OBV":
                    result_df = self._calculate_obv(df)
                elif indicator_name == "威廉指标":
                    result_df = self._calculate_williams(df, **mapped_params)
                elif indicator_name == "CCI":
                    result_df = self._calculate_cci(df, **mapped_params)
                elif indicator_name == "ADX":
                    result_df = self._calculate_adx(df, **mapped_params)
                elif indicator_name == "动量指标":
                    result_df = self._calculate_momentum(df, **mapped_params)
                elif indicator_name == "抛物线SAR":
                    result_df = self._calculate_sar(df, **mapped_params)
                else:
                    return df
                
                # 确保信号列名统一
                if 'signal' not in result_df.columns:
                    logger.info(f"为 {indicator_name} 创建默认信号列")
                    result_df['signal'] = 0
                elif any(col for col in result_df.columns if col.endswith('_signal') and col != 'signal'):
                    # 如果存在其他信号列，将其重命名为统一的'signal'列
                    old_signal_col = next(col for col in result_df.columns if col.endswith('_signal'))
                    result_df['signal'] = result_df[old_signal_col]
                    result_df = result_df.drop(columns=[old_signal_col])
                    
                logger.info(f"{indicator_name} 指标计算完成，信号数量：{len(result_df[result_df['signal'] != 0])}")
                return result_df
                
            except Exception as e:
                logger.error(f"计算 {indicator_name} 时发生错误: {str(e)}")
                df['signal'] = 0
                return df
                
        except Exception as e:
            logger.error(f"计算指标 {indicator_name} 时发生错误: {str(e)}")
            return df
    
    def _calculate_macd(self, df: pd.DataFrame, fastperiod: int = 12, 
                       slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """计算MACD指标"""
        try:
            result_df = df.copy()
            
            # 确保数据为float64类型
            close_prices = self._ensure_float64(df['close'].values)
            
            # 计算MACD
            macd, macdsignal, macdhist = talib.MACD(
                close_prices, 
                fastperiod=int(fastperiod), 
                slowperiod=int(slowperiod), 
                signalperiod=int(signalperiod)
            )
            
            # 添加MACD指标数据
            result_df[f'macd_{fastperiod}_{slowperiod}_{signalperiod}'] = macd
            result_df[f'macd_signal_line_{fastperiod}_{slowperiod}_{signalperiod}'] = macdsignal
            result_df[f'macd_hist_{fastperiod}_{slowperiod}_{signalperiod}'] = macdhist
            
            # 生成交易信号
            result_df['signal'] = 0
            
            # 计算交叉信号
            macd_series = pd.Series(macd, index=df.index)
            signal_series = pd.Series(macdsignal, index=df.index)

            # 金叉：MACD从下方穿越信号线
            golden_cross = (macd_series > signal_series) & (macd_series.shift(1) <= signal_series.shift(1))
            # 死叉：MACD从上方穿越信号线
            death_cross = (macd_series < signal_series) & (macd_series.shift(1) >= signal_series.shift(1))

            # 设置信号
            result_df.loc[golden_cross, 'signal'] = 1
            result_df.loc[death_cross, 'signal'] = -1
            
            logger.info(f"MACD信号生成: 金叉数={sum(golden_cross)}, 死叉数={sum(death_cross)}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"MACD计算错误: {str(e)}")
            df['signal'] = 0
            return df
    
    def _calculate_rsi(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算RSI
        rsi = talib.RSI(close_prices, timeperiod=timeperiod)
        
        # 添加到DataFrame
        df = df.copy()
        df[f'rsi_{timeperiod}'] = rsi
        
        # 生成交易信号 - 将rsi转换为pandas Series以使用shift方法
        rsi_series = pd.Series(rsi, index=df.index)
        df['signal'] = 0
        df.loc[(rsi_series < 30) & (rsi_series.shift(1) >= 30), 'signal'] = 1  # 超卖买入
        df.loc[(rsi_series > 70) & (rsi_series.shift(1) <= 70), 'signal'] = -1  # 超买卖出
        
        return df
    
    def _calculate_kdj(self, df: pd.DataFrame, fastk_period: int = 9,
                      slowk_period: int = 3, slowd_period: int = 3) -> pd.DataFrame:
        """计算KDJ指标"""
        try:
            result_df = df.copy()
            
            # 计算KDJ
            high = pd.Series(df['high'])
            low = pd.Series(df['low'])
            close = pd.Series(df['close'])
            
            # 计算RSV
            low_list = low.rolling(window=fastk_period, min_periods=1).min()
            high_list = high.rolling(window=fastk_period, min_periods=1).max()
            rsv = pd.Series(np.zeros_like(close), index=close.index)
            
            # 处理除数为0的情况
            mask = high_list - low_list != 0
            rsv[mask] = (close[mask] - low_list[mask]) / (high_list[mask] - low_list[mask]) * 100
            
            # 计算K值
            k = pd.Series(np.zeros_like(close), index=close.index)
            k[0] = 50
            for i in range(1, len(rsv)):
                k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
                
            # 计算D值
            d = pd.Series(np.zeros_like(close), index=close.index)
            d[0] = 50
            for i in range(1, len(k)):
                d[i] = (2/3) * d[i-1] + (1/3) * k[i]
                
            # 计算J值
            j = 3 * k - 2 * d
            
            # 保存指标值
            result_df['k'] = k
            result_df['d'] = d
            result_df['j'] = j
            
            # 生成交易信号
            result_df['signal'] = 0
            
            # 金叉买入（K上穿D且J值小于20）
            buy_condition = (k > d) & (k.shift(1) <= d.shift(1)) & (j < 20)
            result_df.loc[buy_condition, 'signal'] = 1
            
            # 死叉卖出（K下穿D且J值大于80）
            sell_condition = (k < d) & (k.shift(1) >= d.shift(1)) & (j > 80)
            result_df.loc[sell_condition, 'signal'] = -1
            
            return result_df
            
        except Exception as e:
            logger.error(f"KDJ计算错误: {str(e)}")
            df['signal'] = 0
            return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, timeperiod: int = 20, 
                                 nbdevup: int = 2, nbdevdn: int = 2) -> pd.DataFrame:
        """计算布林带指标"""
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算布林带
        upperband, middleband, lowerband = talib.BBANDS(
            close_prices,
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )
        
        # 添加到DataFrame
        df = df.copy()
        df[f'bb_upper_{timeperiod}_{nbdevup}'] = upperband
        df[f'bb_middle_{timeperiod}_{nbdevup}'] = middleband
        df[f'bb_lower_{timeperiod}_{nbdevdn}'] = lowerband

        # 生成交易信号
        df['signal'] = 0
        
        # 价格跌破下轨为买入信号
        buy_condition = (df['close'] < lowerband) & (df['close'].shift(1) >= lowerband.shift(1))
        # 价格突破上轨为卖出信号
        sell_condition = (df['close'] > upperband) & (df['close'].shift(1) <= upperband.shift(1))
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def _calculate_moving_average(self, df: pd.DataFrame, timeperiod: int = 20) -> pd.DataFrame:
        """计算移动平均线"""
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算移动平均线
        ma = talib.SMA(close_prices, timeperiod=timeperiod)
        
        # 添加到DataFrame
        df = df.copy()
        df[f'ma_{timeperiod}'] = ma
        
        # 生成交易信号
        df['signal'] = 0
        
        # 价格上穿均线买入
        buy_condition = (df['close'] > ma) & (df['close'].shift(1) <= ma.shift(1))
        # 价格下穿均线卖出
        sell_condition = (df['close'] < ma) & (df['close'].shift(1) >= ma.shift(1))
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def _calculate_bias(self, df: pd.DataFrame, timeperiod: int = 20) -> pd.DataFrame:
        """计算乖离率"""
        close_prices = self._ensure_float64(df['close'].values)
        
        # 计算移动平均线
        ma = talib.SMA(close_prices, timeperiod=timeperiod)
        
        # 计算乖离率
        bias = (close_prices - ma) / ma * 100
        
        # 添加到DataFrame
        df = df.copy()
        df[f'bias_{timeperiod}'] = bias
        
        # 生成交易信号
        df['signal'] = 0
        df.loc[bias < -6, 'signal'] = 1  # 负乖离过大，买入信号
        df.loc[bias > 6, 'signal'] = -1  # 正乖离过大，卖出信号
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算平均真实波幅(ATR)"""
        high = self._ensure_float64(df['high'].values)
        low = self._ensure_float64(df['low'].values)
        close = self._ensure_float64(df['close'].values)
        
        # 计算ATR
        atr = talib.ATR(high, low, close, timeperiod=timeperiod)
        
        df = df.copy()
        df[f'atr_{timeperiod}'] = atr
        df['signal'] = 0  # ATR通常不直接产生交易信号
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算能量潮(OBV)"""
        close = self._ensure_float64(df['close'].values)
        volume = self._ensure_float64(df['volume'].values)
        
        # 计算OBV
        obv = talib.OBV(close, volume)
        
        df = df.copy()
        df['obv'] = obv
        
        # OBV突破信号 - 转换为pandas Series
        obv_series = pd.Series(obv, index=df.index)
        df['signal'] = 0
        df.loc[(obv_series > obv_series.shift(1)) & (obv_series.shift(1) <= obv_series.shift(2)), 'signal'] = 1
        df.loc[(obv_series < obv_series.shift(1)) & (obv_series.shift(1) >= obv_series.shift(2)), 'signal'] = -1
        
        return df
    
    def _calculate_williams(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算威廉指标"""
        high = self._ensure_float64(df['high'].values)
        low = self._ensure_float64(df['low'].values)
        close = self._ensure_float64(df['close'].values)
        
        # 计算威廉指标
        williams = talib.WILLR(high, low, close, timeperiod=timeperiod)
        
        df = df.copy()
        df[f'williams_{timeperiod}'] = williams
        
        # 威廉指标信号 - 转换为pandas Series
        williams_series = pd.Series(williams, index=df.index)
        df['signal'] = 0
        df.loc[(williams_series < -80) & (williams_series.shift(1) >= -80), 'signal'] = 1  # 超卖买入
        df.loc[(williams_series > -20) & (williams_series.shift(1) <= -20), 'signal'] = -1  # 超买卖出
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame, timeperiod: int = 20) -> pd.DataFrame:
        """计算商品通道指数(CCI)"""
        high = self._ensure_float64(df['high'].values)
        low = self._ensure_float64(df['low'].values)
        close = self._ensure_float64(df['close'].values)
        
        # 计算CCI
        cci = talib.CCI(high, low, close, timeperiod=timeperiod)
        
        df = df.copy()
        df[f'cci_{timeperiod}'] = cci
        
        # CCI信号 - 转换为pandas Series
        cci_series = pd.Series(cci, index=df.index)
        df['signal'] = 0
        df.loc[(cci_series < -100) & (cci_series.shift(1) >= -100), 'signal'] = 1  # 超卖买入
        df.loc[(cci_series > 100) & (cci_series.shift(1) <= 100), 'signal'] = -1  # 超买卖出
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算平均方向指数(ADX)"""
        high = self._ensure_float64(df['high'].values)
        low = self._ensure_float64(df['low'].values)
        close = self._ensure_float64(df['close'].values)
        
        # 计算ADX
        adx = talib.ADX(high, low, close, timeperiod=timeperiod)
        
        df = df.copy()
        df[f'adx_{timeperiod}'] = adx
        df['signal'] = 0  # ADX通常用于趋势强度判断，不直接产生交易信号
        
        return df
    
    def _calculate_momentum(self, df: pd.DataFrame, timeperiod: int = 10) -> pd.DataFrame:
        """计算动量指标"""
        close = self._ensure_float64(df['close'].values)
        
        # 计算动量
        momentum = talib.MOM(close, timeperiod=timeperiod)
        
        df = df.copy()
        df[f'momentum_{timeperiod}'] = momentum
        
        # 动量信号 - 转换为pandas Series
        momentum_series = pd.Series(momentum, index=df.index)
        df['signal'] = 0
        df.loc[(momentum_series > 0) & (momentum_series.shift(1) <= 0), 'signal'] = 1  # 动量转正买入
        df.loc[(momentum_series < 0) & (momentum_series.shift(1) >= 0), 'signal'] = -1  # 动量转负卖出
        
        return df
    
    def _calculate_sar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.DataFrame:
        """计算抛物线SAR"""
        high = self._ensure_float64(df['high'].values)
        low = self._ensure_float64(df['low'].values)
        
        # 计算SAR
        sar = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
        
        df = df.copy()
        df[f'sar_{acceleration}_{maximum}'] = sar
        
        # SAR信号
        df['signal'] = 0
        df.loc[(df['close'] > sar) & (df['close'].shift(1) <= sar.shift(1)), 'signal'] = 1  # 价格上穿SAR买入
        df.loc[(df['close'] < sar) & (df['close'].shift(1) >= sar.shift(1)), 'signal'] = -1  # 价格下穿SAR卖出
        
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
    
    print(f"支持的指标: {indicator_calc.get_available_indicators()}")
    
    # 测试几个主要指标
    for indicator in ["MACD", "RSI", "布林带", "ATR"]:
        try:
            result_df = indicator_calc.calculate_indicator(test_df, indicator)
            signal_count = len(result_df[result_df['signal'] != 0])
            print(f"{indicator}: {signal_count}个交易信号")
        except Exception as e:
            print(f"{indicator}测试失败: {str(e)}")