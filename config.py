#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全局配置文件
"""

# 交易参数
TRADE_CONFIG = {
    "commission_rate": 0.0003,  # 手续费率（万三）
    "slippage_rate": 0.0001,   # 滑点率（万一）
    "initial_capital": 100000,  # 初始资金
}

# 技术指标参数范围
INDICATOR_PARAMS = {
    "MACD": {
        "fast_period": {"min": 5, "max": 20, "default": 12},
        "slow_period": {"min": 15, "max": 60, "default": 26},
        "signal_period": {"min": 3, "max": 20, "default": 9}
    },
    "RSI": {
        "timeperiod": {"min": 5, "max": 30, "default": 14}
    },
    "KDJ": {
        "fastk_period": {"min": 5, "max": 30, "default": 9},
        "slowk_period": {"min": 2, "max": 15, "default": 3},
        "slowd_period": {"min": 2, "max": 15, "default": 3}
    },
    "布林带": {
        "timeperiod": {"min": 5, "max": 40, "default": 20},
        "nbdevup": {"min": 1, "max": 4, "default": 2},
        "nbdevdn": {"min": 1, "max": 4, "default": 2}
    },
    "均线": {
        "timeperiod": {"min": 5, "max": 120, "default": 20}
    },
    "乖离率": {
        "timeperiod": {"min": 5, "max": 120, "default": 20}
    },
    "ATR": {
        "timeperiod": {"min": 5, "max": 30, "default": 14}
    },
    "OBV": {
        # OBV无参数
    },
    "威廉指标": {
        "timeperiod": {"min": 5, "max": 30, "default": 14}
    },
    "CCI": {
        "timeperiod": {"min": 10, "max": 30, "default": 20}
    },
    "ADX": {
        "timeperiod": {"min": 5, "max": 30, "default": 14}
    },
    "动量指标": {
        "timeperiod": {"min": 5, "max": 20, "default": 10}
    },
    "抛物线SAR": {
        "acceleration": {"min": 0.01, "max": 0.05, "default": 0.02},
        "maximum": {"min": 0.1, "max": 0.4, "default": 0.2}
    }
}

# 优化算法参数
OPTIMIZER_CONFIG = {
    "grid_search": {
        "n_jobs": -1,  # 并行任务数，-1表示使用所有CPU核心
        "chunk_size": 1000  # 每批处理的参数组合数
    },
    "genetic": {
        "population_size": 50,
        "generations": 30,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8
    }
}

# 回测引擎参数
BACKTEST_CONFIG = {
    "price_limits": True,  # 是否考虑涨跌停限制
    "max_position_pct": 1.0,  # 最大持仓比例
    "retry_count": 3,  # 数据获取重试次数
    "retry_delay": 5,  # 重试等待时间（秒）
}

# 风险管理参数
RISK_CONFIG = {
    "max_position_size": 1.0,        # 最大仓位比例
    "max_loss_per_trade": 0.02,      # 单笔交易最大亏损比例
    "max_drawdown": 0.15,            # 最大回撤限制
    "stop_loss_pct": 0.05,           # 止损百分比
    "take_profit_pct": 0.10,         # 止盈百分比
    "max_daily_loss": 0.03,          # 每日最大亏损限制
    "risk_levels": {                 # 风险等级定义
        "low": {"max_drawdown": 0.05, "position_size": 1.0},
        "medium": {"max_drawdown": 0.10, "position_size": 0.8},
        "high": {"max_drawdown": 0.15, "position_size": 0.6},
        "extreme": {"max_drawdown": 0.20, "position_size": 0.4}
    },
    "volatility_adjustment": {        # 波动率调整
        "high_vol_threshold": 0.03,   # 高波动率阈值
        "low_vol_threshold": 0.01,    # 低波动率阈值
        "stop_loss_multiplier": 1.5,  # 高波动时止损倍数
        "take_profit_multiplier": 1.5 # 高波动时止盈倍数
    }
}

# 多品种配置
MULTI_SYMBOL_CONFIG = {
    "max_symbols": 5,               # 最大同时持有品种数
    "correlation_threshold": 0.7,    # 相关性阈值
    "sector_limits": {               # 行业限制
        "technology": 0.4,           # 科技行业最大仓位
        "finance": 0.3,              # 金融行业最大仓位
        "consumer": 0.3,             # 消费行业最大仓位
        "others": 0.2                # 其他行业最大仓位
    }
}
