"""
参数评估包装模块
为并行计算提供独立的参数评估函数
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd

from indicators.technical_indicators import TechnicalIndicators
from backtest.backtest_engine import BacktestEngine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_parameters_wrapper(data: Tuple[pd.DataFrame, str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    用于并行计算的参数评估包装函数
    
    Args:
        data: 包含所有必要信息的元组 (df, indicator_name, params)
        
    Returns:
        Dict: 评估结果
    """
    df, indicator_name, params = data
    try:
        # 创建新的计算器实例
        indicator_calc = TechnicalIndicators()
        backtest_engine = BacktestEngine()
        
        # 计算技术指标
        df_with_indicator = indicator_calc.calculate_indicator(df.copy(), indicator_name, params)
        
        # 执行回测
        results = backtest_engine.run_backtest(df_with_indicator)
        
        # 添加参数信息到结果中
        if isinstance(results, dict):
            results.update(params)
        else:
            results = {
                **params,
                '年化收益率': 0.0,
                '最大回撤': 0.0,
                '夏普比率': 0.0
            }
        
        return results
        
    except Exception as e:
        logger.error(f"参数评估失败: {str(e)}, params: {params}")
        return {
            **params,
            '年化收益率': float('-inf'),
            '最大回撤': float('-inf'),
            '夏普比率': float('-inf')
        }
