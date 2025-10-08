"""
数据获取模块
负责从akshare接口获取股票历史行情数据
"""
from .data_fetcher import StockDataFetcher

__all__ = ['StockDataFetcher']