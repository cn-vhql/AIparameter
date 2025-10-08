#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据获取模块
使用akshare接口获取A股股票历史行情数据
支持前复权处理和多时间周期
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self):
        self.period_mapping = {
            "日线": "daily",
            "周线": "weekly", 
            "月线": "monthly"
        }
    
    def get_stock_data(self, stock_code: str, start_date: str, end_date: str, 
                      period: str = "日线") -> Optional[pd.DataFrame]:
        """
        获取股票历史数据
        
        Args:
            stock_code: 股票代码，如 "000001.SZ"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            period: 时间周期，"日线"/"周线"/"月线"
            
        Returns:
            pd.DataFrame: 包含股票数据的DataFrame，包含以下列：
                date, open, high, low, close, volume, turnover
        """
        try:
            # 验证股票代码格式
            if not self._validate_stock_code(stock_code):
                logger.error(f"无效的股票代码格式: {stock_code}")
                return None
            
            # 转换日期格式
            start_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime) else start_date
            end_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date
            
            # 获取股票基础信息
            stock_info = self._get_stock_info(stock_code)
            if not stock_info:
                return None
            
            # 根据时间周期选择不同的数据接口
            if period == "日线":
                df = self._get_daily_data(stock_code, start_str, end_str)
            elif period == "周线":
                df = self._get_weekly_data(stock_code, start_str, end_str)
            elif period == "月线":
                df = self._get_monthly_data(stock_code, start_str, end_str)
            else:
                logger.error(f"不支持的时间周期: {period}")
                return None
            
            if df is not None and not df.empty:
                # 数据清洗和处理
                df = self._clean_data(df)
                logger.info(f"成功获取 {stock_code} 的{len(df)}条{period}数据")
                return df
            else:
                logger.error(f"获取 {stock_code} 数据失败")
                return None
                
        except Exception as e:
            logger.error(f"获取股票数据时发生错误: {str(e)}")
            return None
    
    def _validate_stock_code(self, stock_code: str) -> bool:
        """验证股票代码格式"""
        if not stock_code:
            return False
        
        # 基本格式验证：6位数字 + .SZ 或 .SH
        if len(stock_code) != 9:
            return False
        
        code_part = stock_code[:6]
        exchange_part = stock_code[6:]
        
        if not code_part.isdigit():
            return False
        
        if exchange_part not in [".SZ", ".SH"]:
            return False
            
        return True
    
    def _get_stock_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取股票基本信息"""
        try:
            # 使用stock_info_a_code_name接口获取股票基本信息
            stock_info_df = ak.stock_info_a_code_name()
            stock_info = stock_info_df[stock_info_df["code"] == stock_code[:6]]

            if not stock_info.empty:
                return {
                    "code": stock_code[:6],
                    "name": stock_info["name"].iloc[0],
                    "exchange": "SZ" if ".SZ" in stock_code else "SH"
                }
            else:
                logger.error(f"未找到股票代码: {stock_code}")
                return None

        except Exception as e:
            logger.error(f"获取股票信息时发生错误: {str(e)}")
            return None
    
    def _get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取日线数据（前复权）"""
        try:
            # 提取股票代码（去掉交易所后缀）
            code = stock_code[:6]

            # 使用stock_zh_a_hist接口获取数据
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start_date, end_date=end_date, adjust='qfq')

            if df is not None and not df.empty:
                # 设置日期为索引
                df = df.set_index('日期')
                # 按日期排序
                df = df.sort_index()
                return df
            return None

        except Exception as e:
            logger.error(f"获取日线数据时发生错误: {str(e)}")
            return None
    
    def _get_weekly_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取周线数据"""
        try:
            # 先获取日线数据，然后重采样为周线
            daily_df = self._get_daily_data(stock_code, start_date, end_date)
            if daily_df is not None:
                weekly_df = daily_df.resample('W').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                return weekly_df
            return None
            
        except Exception as e:
            logger.error(f"获取周线数据时发生错误: {str(e)}")
            return None
    
    def _get_monthly_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取月线数据"""
        try:
            # 先获取日线数据，然后重采样为月线
            daily_df = self._get_daily_data(stock_code, start_date, end_date)
            if daily_df is not None:
                monthly_df = daily_df.resample('M').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                return monthly_df
            return None
            
        except Exception as e:
            logger.error(f"获取月线数据时发生错误: {str(e)}")
            return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗和处理"""
        # 确保索引是日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 重命名列 - 根据实际数据列名进行映射
        column_mapping = {
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'turnover'
        }

        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"缺少必要列: {col}")
        
        # 处理缺失值
        df = df.dropna(subset=required_columns)
        
        # 确保数据按日期排序
        df = df.sort_index()
        
        return df
    
    def get_multiple_stocks_data(self, stock_codes: list, start_date: str, 
                               end_date: str, period: str = "日线") -> Dict[str, pd.DataFrame]:
        """批量获取多只股票数据"""
        results = {}
        for code in stock_codes:
            df = self.get_stock_data(code, start_date, end_date, period)
            if df is not None:
                results[code] = df
        return results

# 测试函数
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    
    # 测试获取数据
    test_code = "000001.SZ"
    start_date = "20230101"
    end_date = "20231231"
    
    df = fetcher.get_stock_data(test_code, start_date, end_date, "日线")
    
    if df is not None:
        print(f"成功获取 {test_code} 的 {len(df)} 条数据")
        print(df.head())
    else:
        print("获取数据失败")