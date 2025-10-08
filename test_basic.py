#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基本功能测试脚本
验证各个模块的基本功能是否正常工作
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_fetcher():
    """测试数据获取模块"""
    print("🧪 测试数据获取模块...")
    try:
        from data.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        
        # 测试股票代码验证
        assert fetcher._validate_stock_code("000001.SZ") == True
        assert fetcher._validate_stock_code("600036.SH") == True
        assert fetcher._validate_stock_code("invalid") == False
        
        print("✅ 股票代码验证测试通过")
        
    except Exception as e:
        print(f"❌ 数据获取模块测试失败: {e}")
        return False
    return True

def test_technical_indicators():
    """测试技术指标模块"""
    print("🧪 测试技术指标模块...")
    try:
        from indicators.technical_indicators import TechnicalIndicators
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        test_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000000] * 10
        }, index=dates)
        
        indicator_calc = TechnicalIndicators()
        
        # 测试MACD计算
        macd_df = indicator_calc.calculate_indicator(test_df, "MACD")
        assert 'macd_12_26_9' in macd_df.columns
        assert 'macd_signal' in macd_df.columns
        
        # 测试RSI计算
        rsi_df = indicator_calc.calculate_indicator(test_df, "RSI")
        assert 'rsi_14' in rsi_df.columns
        
        print("✅ 技术指标模块测试通过")
        
    except Exception as e:
        print(f"❌ 技术指标模块测试失败: {e}")
        return False
    return True

def test_backtest_engine():
    """测试回测引擎模块"""
    print("🧪 测试回测引擎模块...")
    try:
        from backtest.backtest_engine import BacktestEngine
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-01-20', freq='D')
        n_days = len(dates)
        
        # 生成价格和信号数据
        prices = np.random.normal(0.001, 0.02, n_days).cumsum() + 100
        signals = np.random.choice([-1, 0, 1], n_days, p=[0.2, 0.6, 0.2])
        
        test_df = pd.DataFrame({
            'close': prices,
            'signal': signals
        }, index=dates)
        
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run_backtest(test_df, 'signal', 'close')
        
        assert 'performance_metrics' in results
        assert 'trades' in results
        assert 'equity_curve' in results
        
        print("✅ 回测引擎模块测试通过")
        
    except Exception as e:
        print(f"❌ 回测引擎模块测试失败: {e}")
        return False
    return True

def test_optimizer():
    """测试优化器模块"""
    print("🧪 测试优化器模块...")
    try:
        from optimization.optimizer import ParameterOptimizer
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        n_days = len(dates)
        
        prices = np.random.normal(0.0005, 0.015, n_days).cumsum() + 100
        prices = np.maximum(prices, 1)
        
        test_df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, n_days)
        }, index=dates)
        
        optimizer = ParameterOptimizer(n_jobs=1)  # 使用单线程测试
        
        # 测试获取默认参数范围
        param_ranges = optimizer._get_default_param_ranges("MACD")
        assert 'fastperiod' in param_ranges
        assert 'slowperiod' in param_ranges
        assert 'signalperiod' in param_ranges
        
        print("✅ 优化器模块测试通过")
        
    except Exception as e:
        print(f"❌ 优化器模块测试失败: {e}")
        return False
    return True

def main():
    """运行所有测试"""
    print("🚀 开始运行股票指标参数优化系统测试...")
    print("=" * 50)
    
    tests = [
        test_data_fetcher,
        test_technical_indicators,
        test_backtest_engine,
        test_optimizer
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试执行异常: {e}")
            results.append(False)
    
    print("=" * 50)
    print("📊 测试结果汇总:")
    print(f"总测试数: {len(results)}")
    print(f"通过数: {sum(results)}")
    print(f"失败数: {len(results) - sum(results)}")
    
    if all(results):
        print("🎉 所有测试通过！系统基本功能正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)