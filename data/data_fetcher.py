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
                      period: str = "日线", retry_count: int = 3) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据
        
        Args:
            stock_code: 股票代码，如 "000001.SZ"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            period: 时间周期，"日线"/"周线"/"月线"
            retry_count: 重试次数
            
        Returns:
            pd.DataFrame: 包含股票数据的DataFrame，包含以下列：
                date, open, high, low, close, volume, turnover
        """
        # 验证股票代码格式
        if not self._validate_stock_code(stock_code):
            logger.error(f"无效的股票代码格式: {stock_code}")
            return None
            
        # 获取周期映射
        period_en = self.period_mapping.get(period)
        if not period_en:
            logger.error(f"不支持的时间周期: {period}")
            return None
            
        # 转换日期格式
        try:
            start_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime) else start_date
            end_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date
        except Exception as e:
            logger.error(f"日期格式转换错误: {str(e)}")
            return None
        
        # 尝试获取数据
        for attempt in range(retry_count):
            try:
                # 获取股票基础信息
                stock_info = self._get_stock_info(stock_code)
                if not stock_info:
                    logger.warning(f"第 {attempt + 1} 次尝试: 无法获取股票信息")
                    continue
                
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
                    if not df.empty:
                        logger.info(f"成功获取 {stock_code} 的{len(df)}条{period}数据")
                        return df
                    else:
                        logger.warning(f"第 {attempt + 1} 次尝试: 数据清洗后为空")
                        continue
                else:
                    logger.warning(f"第 {attempt + 1} 次尝试: 获取到的数据为空")
                    continue
                    
            except Exception as e:
                logger.warning(f"第 {attempt + 1} 次获取数据失败: {str(e)}")
                if attempt < retry_count - 1:
                    import time
                    time.sleep(2)  # 等待2秒后重试
                    
        logger.error(f"获取 {stock_code} 数据失败，已重试 {retry_count} 次")
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
    
    def _get_period_data(self, stock_code: str, start_date: str, end_date: str, period: str) -> Optional[pd.DataFrame]:
        """根据akshare接口直接获取指定周期的数据"""
        try:
            # 提取股票代码（去掉交易所后缀）
            code = stock_code[:6]
            
            # 映射周期参数
            period_mapping = {
                "日线": "daily",
                "周线": "weekly", 
                "月线": "monthly"
            }
            
            akshare_period = period_mapping.get(period, "daily")
            
            logger.info(f"正在获取 {stock_code} 的{period}数据，日期范围: {start_date} - {end_date}")

            # 使用stock_zh_a_hist接口直接获取数据
            df = ak.stock_zh_a_hist(
                symbol=code, 
                period=akshare_period, 
                start_date=start_date, 
                end_date=end_date, 
                adjust='qfq'
            )

            if df is not None and not df.empty:
                logger.info(f"成功获取原始{period}数据，列名: {list(df.columns)}，数据形状: {df.shape}")
                
                # 设置日期为索引
                if '日期' in df.columns:
                    df = df.set_index('日期')
                elif 'date' in df.columns:
                    df = df.set_index('date')
                else:
                    # 尝试第一列作为日期
                    df = df.set_index(df.columns[0])
                
                # 确保索引是日期类型
                df.index = pd.to_datetime(df.index)
                
                # 按日期排序
                df = df.sort_index()
                
                logger.info(f"处理后的{period}数据形状: {df.shape}")
                return df
            else:
                logger.warning(f"获取到的{period}数据为空")
                return None

        except Exception as e:
            logger.error(f"获取{period}数据时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取日线数据（前复权）"""
        return self._get_period_data(stock_code, start_date, end_date, "日线")
    
    def _get_weekly_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取周线数据（前复权）"""
        return self._get_period_data(stock_code, start_date, end_date, "周线")
    
    def _get_monthly_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取月线数据（前复权）"""
        return self._get_period_data(stock_code, start_date, end_date, "月线")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗和处理
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        try:
            # 复制数据，避免修改原始数据
            df = df.copy()
            
            # 确保索引是日期类型
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            logger.info(f"开始数据清洗，原始列名: {list(df.columns)}")
            
            # 根据akshare接口说明重命名列
            column_mapping = {
                '开盘': 'open',
                '最高': 'high', 
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'turnover',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate',
                # 英文列名映射
                'open': 'open',
                'high': 'high',
                'low': 'low', 
                'close': 'close',
                'volume': 'volume',
                'amount': 'turnover',
                'amplitude': 'amplitude',
                'pct_chg': 'pct_change',
                'change': 'change_amount',
                'turnover': 'turnover_rate'
            }

            # 重命名列
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            logger.info(f"列重命名后: {list(df.columns)}")
            
            # 确保必要的列存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"数据缺少必要列: {', '.join(missing_columns)}")
                return pd.DataFrame()  # 返回空DataFrame
            
            # 处理成交量单位（akshare返回的是手，转换为股）
            if 'volume' in df.columns:
                df['volume'] = df['volume'] * 100  # 1手 = 100股
            
            # 处理异常值
            # 价格不能为负或零
            for col in ['open', 'high', 'low', 'close']:
                mask = df[col] <= 0
                if mask.any():
                    logger.warning(f"发现非正数价格在 {col} 列，将被填充")
                    # 使用前一个有效价格
                    df[col] = df[col].replace(0, np.nan)
                    df[col] = df[col].fillna(method='ffill')
            
            # 确保 high >= low
            invalid_mask = df['high'] < df['low']
            if invalid_mask.any():
                logger.warning(f"发现 {invalid_mask.sum()} 条最高价小于最低价的记录，将进行修正")
                # 交换高低价
                temp = df.loc[invalid_mask, 'high'].copy()
                df.loc[invalid_mask, 'high'] = df.loc[invalid_mask, 'low']
                df.loc[invalid_mask, 'low'] = temp
            
            # 处理缺失值
            na_count = df[required_columns].isna().sum()
            if na_count.any():
                logger.warning(f"发现缺失值，将进行填充:\n{na_count[na_count > 0]}")
                # 使用前向填充价格数据
                price_cols = ['open', 'high', 'low', 'close']
                df[price_cols] = df[price_cols].fillna(method='ffill')
                df[price_cols] = df[price_cols].bfill()
                df['volume'] = df['volume'].fillna(0)
            
            # 删除仍包含缺失值的行
            initial_rows = len(df)
            df = df.dropna(subset=required_columns)
            final_rows = len(df)
            
            if final_rows < initial_rows:
                logger.info(f"删除了 {initial_rows - final_rows} 条包含缺失值的记录")
            
            # 确保数据按日期排序
            df = df.sort_index()
            
            # 添加或更新辅助列
            if 'pct_change' not in df.columns:
                df['pct_change'] = df['close'].pct_change()
            
            if 'amplitude' not in df.columns:
                # 使用收盘价计算振幅
                df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
            
            logger.info(f"数据清洗完成，最终数据形状: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"数据清洗过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
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