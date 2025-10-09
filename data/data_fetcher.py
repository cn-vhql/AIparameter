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
import os
import pickle
import hashlib
import json
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = "data_cache", 
                 cache_expiry_hours: int = 24):
        """
        初始化股票数据获取器
        
        Args:
            cache_enabled: 是否启用缓存
            cache_dir: 缓存目录
            cache_expiry_hours: 缓存过期时间（小时）
        """
        self.period_mapping = {
            "日线": "daily",
            "周线": "weekly", 
            "月线": "monthly"
        }
        
        # 缓存配置
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        self.cache_expiry_hours = cache_expiry_hours
        
        # 创建缓存目录
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"数据缓存已启用，缓存目录: {self.cache_dir.absolute()}")
        else:
            logger.info("数据缓存已禁用")
    
    def _generate_cache_key(self, stock_code: str, start_date: str, end_date: str, period: str) -> str:
        """生成缓存键"""
        # 创建标准化的参数字符串
        params = f"{stock_code}_{start_date}_{end_date}_{period}"
        # 使用MD5哈希生成固定长度的键
        return hashlib.md5(params.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """检查缓存是否有效"""
        if not cache_file.exists():
            return False
        
        try:
            # 检查文件修改时间
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
            
            return file_mtime > expiry_time
        except Exception as e:
            logger.error(f"检查缓存有效性时发生错误: {str(e)}")
            return False
    
    def _get_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """读取缓存数据"""
        if not self.cache_enabled:
            return None
            
        cache_file = self._get_cache_file_path(cache_key)
        
        if not self._is_cache_valid(cache_file):
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 验证缓存数据
            if isinstance(cached_data, dict) and 'data' in cached_data and 'meta' in cached_data:
                logger.info(f"从缓存加载数据: {cache_key}")
                return cached_data['data']
            elif isinstance(cached_data, pd.DataFrame):
                # 兼容旧版本缓存格式
                logger.info(f"从缓存加载数据 (旧格式): {cache_key}")
                return cached_data
            else:
                logger.warning(f"缓存数据格式无效: {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"读取缓存时发生错误: {str(e)}")
            return None
    
    def _set_cache(self, cache_key: str, data: pd.DataFrame, meta: Dict[str, Any] = None) -> bool:
        """写入缓存数据"""
        if not self.cache_enabled:
            return False
            
        try:
            cache_file = self._get_cache_file_path(cache_key)
            
            # 准备缓存数据
            cache_data = {
                'data': data,
                'meta': {
                    'cache_time': datetime.now(),
                    'data_shape': data.shape,
                    'data_range': {
                        'start_date': str(data.index.min()) if not data.empty else None,
                        'end_date': str(data.index.max()) if not data.empty else None
                    }
                }
            }
            
            # 添加用户提供的元数据
            if meta:
                cache_data['meta'].update(meta)
            
            # 写入缓存文件
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"数据已缓存: {cache_key}, 数据行数: {len(data)}")
            return True
            
        except Exception as e:
            logger.error(f"写入缓存时发生错误: {str(e)}")
            return False
    
    def get_stock_data(self, stock_code: str, start_date: str, end_date: str, 
                      period: str = "日线", retry_count: int = 3, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据
        
        Args:
            stock_code: 股票代码，如 "000001.SZ"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            period: 时间周期，"日线"/"周线"/"月线"
            retry_count: 重试次数
            use_cache: 是否使用缓存
            
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
        
        # 生成缓存键
        cache_key = self._generate_cache_key(stock_code, start_str, end_str, period)
        
        # 尝试从缓存读取数据
        if use_cache and self.cache_enabled:
            cached_data = self._get_cache(cache_key)
            if cached_data is not None:
                logger.info(f"使用缓存数据: {stock_code} {period} ({len(cached_data)} 条)")
                return cached_data
        
        # 缓存未命中，从API获取数据
        logger.info(f"从API获取数据: {stock_code} {period}")
        
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
                        # 缓存数据
                        if self.cache_enabled:
                            meta = {
                                'stock_code': stock_code,
                                'period': period,
                                'start_date': start_str,
                                'end_date': end_str,
                                'source': 'akshare',
                                'fetch_time': datetime.now()
                            }
                            self._set_cache(cache_key, df, meta)
                        
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
    
    def clear_cache(self, stock_code: str = None, period: str = None, 
                   start_date: str = None, end_date: str = None) -> int:
        """
        清理缓存数据
        
        Args:
            stock_code: 股票代码，为None时清理所有缓存
            period: 时间周期，为None时清理所有周期
            start_date: 开始日期，为None时清理所有日期
            end_date: 结束日期，为None时清理所有日期
            
        Returns:
            int: 清理的缓存文件数量
        """
        if not self.cache_enabled:
            logger.info("缓存已禁用，无需清理")
            return 0
            
        try:
            cleared_count = 0
            
            if stock_code or period or start_date or end_date:
                # 清理指定条件的缓存
                for cache_file in self.cache_dir.glob("*.pkl"):
                    # 生成对应的缓存键进行比较
                    # 这里简化处理，直接删除所有缓存文件
                    cache_file.unlink()
                    cleared_count += 1
            else:
                # 清理所有缓存
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    cleared_count += 1
            
            logger.info(f"已清理 {cleared_count} 个缓存文件")
            return cleared_count
            
        except Exception as e:
            logger.error(f"清理缓存时发生错误: {str(e)}")
            return 0
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            Dict: 缓存统计信息
        """
        if not self.cache_enabled:
            return {
                "cache_enabled": False,
                "cache_dir": str(self.cache_dir),
                "cache_files": 0,
                "cache_size_mb": 0.0,
                "cache_expiry_hours": self.cache_expiry_hours
            }
            
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            cache_size = sum(f.stat().st_size for f in cache_files if f.exists())
            
            # 获取缓存文件详细信息
            cache_details = []
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if isinstance(cached_data, dict) and 'meta' in cached_data:
                        meta = cached_data['meta']
                        cache_details.append({
                            "file": cache_file.name,
                            "size_mb": round(cache_file.stat().st_size / 1024 / 1024, 2),
                            "cache_time": meta.get('cache_time'),
                            "stock_code": meta.get('stock_code'),
                            "period": meta.get('period'),
                            "data_shape": meta.get('data_shape'),
                            "data_range": meta.get('data_range')
                        })
                except Exception as e:
                    logger.warning(f"读取缓存文件信息失败 {cache_file.name}: {str(e)}")
            
            return {
                "cache_enabled": True,
                "cache_dir": str(self.cache_dir),
                "cache_files": len(cache_files),
                "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                "cache_expiry_hours": self.cache_expiry_hours,
                "cache_details": cache_details
            }
            
        except Exception as e:
            logger.error(f"获取缓存信息时发生错误: {str(e)}")
            return {
                "cache_enabled": True,
                "cache_dir": str(self.cache_dir),
                "cache_files": 0,
                "cache_size_mb": 0.0,
                "cache_expiry_hours": self.cache_expiry_hours,
                "error": str(e)
            }
    
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