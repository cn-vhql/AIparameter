#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版参数优化模块
避免DEAP库的复杂依赖问题
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import random

# 导入自定义模块
from indicators.technical_indicators import TechnicalIndicators
from backtest.backtest_engine import BacktestEngine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleParameterOptimizer:
    """简化版参数优化器"""

    def __init__(self, n_jobs: int = -1):
        """
        初始化参数优化器

        Args:
            n_jobs: 并行工作数，-1表示使用所有CPU核心
        """
        self.n_jobs = n_jobs
        self.indicator_calc = TechnicalIndicators()
        self.backtest_engine = BacktestEngine()

    def optimize_parameters(self, df: pd.DataFrame, indicator_name: str,
                          algorithm: str = "grid_search", **kwargs) -> Tuple[Dict, pd.DataFrame]:
        """
        优化技术指标参数
        """
        # 支持多种算法名称
        if algorithm in ["grid_search", "网格搜索"]:
            return self._grid_search_optimization(df, indicator_name, **kwargs)
        elif algorithm in ["genetic", "genetic_algorithm", "遗传算法"]:
            return self._simple_genetic_optimization(df, indicator_name, **kwargs)
        else:
            logger.error(f"不支持的优化算法: {algorithm}")
            return {}, pd.DataFrame()

    def _grid_search_optimization(self, df: pd.DataFrame, indicator_name: str,
                                param_ranges: Dict[str, List] = None,
                                objective: str = "年化收益率") -> Tuple[Dict, pd.DataFrame]:
        """网格搜索优化"""
        # 获取默认参数范围
        if param_ranges is None:
            param_ranges = self._get_default_param_ranges(indicator_name)

        # 生成所有参数组合
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        param_combinations = list(product(*param_values))

        logger.info(f"开始网格搜索，共 {len(param_combinations)} 种参数组合")

        # 并行计算所有参数组合的回测结果
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_parameters)(df, indicator_name, dict(zip(param_names, params)))
            for params in tqdm(param_combinations, desc="网格搜索进度")
        )

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 找到最优参数
        if not results_df.empty:
            best_idx = results_df[objective].idxmax()
            best_params = results_df.loc[best_idx, 'parameters']
            return best_params, results_df
        else:
            return {}, pd.DataFrame()

    def _simple_genetic_optimization(self, df: pd.DataFrame, indicator_name: str,
                                   population_size: int = 20,
                                   generations: int = 5,
                                   objective: str = "年化收益率") -> Tuple[Dict, pd.DataFrame]:
        """简化版遗传算法优化"""
        # 获取参数信息
        param_info = self.indicator_calc.get_indicator_info(indicator_name)
        if not param_info:
            return {}, pd.DataFrame()

        param_names = param_info["params"]
        param_ranges = []
        for param_name in param_names:
            param_range = self._get_param_range(param_name, indicator_name)
            param_ranges.append(param_range)

        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = []
            for min_val, max_val in param_ranges:
                individual.append(random.randint(min_val, max_val))
            population.append(individual)

        logger.info(f"开始遗传算法优化，种群大小: {population_size}, 代数: {generations}")

        all_results = []

        for gen in range(generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                params = dict(zip(param_names, individual))
                result = self._evaluate_parameters(df, indicator_name, params)
                fitness = result.get(objective, 0)
                fitness_scores.append(fitness)

                # 记录结果
                all_results.append({
                    'generation': gen,
                    'parameters': params.copy(),
                    objective: fitness
                })

            # 选择（简化选择逻辑）
            new_population = []
            for _ in range(population_size):
                if random.random() < 0.5:  # 50%概率选择现有个体
                    # 选择适应度较高的个体
                    if fitness_scores:
                        # 找到非负的最大适应度
                        valid_scores = [(i, score) for i, score in enumerate(fitness_scores) if score > 0]
                        if valid_scores:
                            # 按适应度排序，选择较好的
                            valid_scores.sort(key=lambda x: x[1], reverse=True)
                            # 选择前30%中的一个
                            top_count = max(1, len(valid_scores) // 3)
                            selected_idx = valid_scores[random.randint(0, top_count - 1)][0]
                            selected = population[selected_idx].copy()
                        else:
                            # 所有适应度都<=0，随机选择
                            selected_idx = random.randint(0, population_size - 1)
                            selected = population[selected_idx].copy()
                    else:
                        # 随机生成新个体
                        selected = []
                        for min_val, max_val in param_ranges:
                            selected.append(random.randint(min_val, max_val))
                else:
                    # 随机生成新个体
                    selected = []
                    for min_val, max_val in param_ranges:
                        selected.append(random.randint(min_val, max_val))

                # 变异
                if random.random() < 0.3:  # 30%变异率
                    for i in range(len(selected)):
                        if random.random() < 0.2:  # 20%每个基因变异率
                            min_val, max_val = param_ranges[i]
                            selected[i] = random.randint(min_val, max_val)

                new_population.append(selected)

            population = new_population

        # 找到最优解
        best_fitness = -float('inf')
        best_individual = None
        for individual in population:
            params = dict(zip(param_names, individual))
            result = self._evaluate_parameters(df, indicator_name, params)
            fitness = result.get(objective, 0)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = params

        return best_individual or {}, pd.DataFrame(all_results)

    def _evaluate_parameters(self, df: pd.DataFrame, indicator_name: str,
                           params: Dict[str, int]) -> Dict[str, Any]:
        """评估特定参数组合"""
        try:
            # 计算技术指标
            indicator_df = self.indicator_calc.calculate_indicator(df, indicator_name, params)

            # 确定信号列名
            signal_column = f"{indicator_name.lower()}_signal"
            if signal_column not in indicator_df.columns:
                # 尝试其他可能的信号列名
                possible_signal_columns = [col for col in indicator_df.columns if 'signal' in col]
                if possible_signal_columns:
                    signal_column = possible_signal_columns[0]
                else:
                    return {'parameters': params, '年化收益率': -100, '夏普比率': -100}

            # 运行回测
            results = self.backtest_engine.run_backtest(indicator_df, signal_column)

            # 提取绩效指标
            metrics = results.get('performance_metrics', {})
            # 确保所有值都是可序列化的基本类型
            clean_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, str)):
                    clean_metrics[key] = value
                elif isinstance(value, dict):
                    # 跳过嵌套字典
                    continue
                else:
                    # 跳过其他复杂对象
                    continue
            clean_metrics['parameters'] = params

            return clean_metrics

        except Exception as e:
            logger.error(f"评估参数 {params} 时发生错误: {str(e)}")
            return {'parameters': params, '年化收益率': -100, '夏普比率': -100}

    def _get_default_param_ranges(self, indicator_name: str) -> Dict[str, List]:
        """获取默认参数范围"""
        ranges = {
            "MACD": {
                "fastperiod": list(range(5, 21, 2)),      # 5-20，步长2
                "slowperiod": list(range(15, 41, 5)),    # 15-40，步长5
                "signalperiod": list(range(3, 16, 2))     # 3-15，步长2
            },
            "RSI": {
                "timeperiod": list(range(6, 31, 2))      # 6-30，步长2
            },
            "KDJ": {
                "fastk_period": list(range(5, 16, 2)),   # 5-15，步长2
                "slowk_period": list(range(2, 6)),         # 2-5
                "slowd_period": list(range(2, 6))          # 2-5
            },
            "布林带": {
                "timeperiod": list(range(10, 31, 5)),     # 10-30，步长5
                "nbdevup": [1, 2, 3],                    # 1-3
                "nbdevdn": [1, 2, 3]                    # 1-3
            },
            "均线": {
                "timeperiod": list(range(5, 61, 5))      # 5-60，步长5
            },
            "乖离率": {
                "timeperiod": list(range(10, 61, 5))      # 10-60，步长5
            }
        }

        return ranges.get(indicator_name, {})

    def _get_param_range(self, param_name: str, indicator_name: str) -> Tuple[int, int]:
        """获取参数范围"""
        default_ranges = self._get_default_param_ranges(indicator_name)
        if param_name in default_ranges:
            values = default_ranges[param_name]
            return min(values), max(values)
        else:
            # 默认范围
            return 1, 100

# 兼容性别名
ParameterOptimizer = SimpleParameterOptimizer