#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
参数优化模块
实现网格搜索和遗传算法优化技术指标参数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import random
from deap import base, creator, tools, algorithms

# 导入自定义模块
from indicators.technical_indicators import TechnicalIndicators
from backtest.backtest_engine import BacktestEngine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """参数优化器"""
    
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
        
        Args:
            df: 股票数据DataFrame
            indicator_name: 指标名称
            algorithm: 优化算法 ("grid_search" 或 "genetic")
            **kwargs: 算法特定参数
            
        Returns:
            Tuple: (最优参数, 所有参数组合结果DataFrame)
        """
        # 支持多种算法名称
        if algorithm in ["grid_search", "网格搜索"]:
            return self._grid_search_optimization(df, indicator_name, **kwargs)
        elif algorithm in ["genetic", "genetic_algorithm", "遗传算法"]:
            # 使用简化版遗传算法
            from .simple_optimizer import SimpleParameterOptimizer
            simple_optimizer = SimpleParameterOptimizer(self.n_jobs)
            return simple_optimizer._simple_genetic_optimization(df, indicator_name, **kwargs)
        else:
            logger.error(f"不支持的优化算法: {algorithm}")
            return {}, pd.DataFrame()
    
    def _grid_search_optimization(self, df: pd.DataFrame, indicator_name: str,
                                param_ranges: Dict[str, List] = None,
                                objective: str = "年化收益率") -> Tuple[Dict, pd.DataFrame]:
        """
        网格搜索优化
        
        Args:
            df: 股票数据
            indicator_name: 指标名称
            param_ranges: 参数范围字典
            objective: 优化目标指标
            
        Returns:
            Tuple: (最优参数, 所有结果)
        """
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
    
    def _genetic_algorithm_optimization(self, df: pd.DataFrame, indicator_name: str,
                                      population_size: int = 50,
                                      generations: int = 10,
                                      cx_prob: float = 0.5,
                                      mut_prob: float = 0.2,
                                      objective: str = "年化收益率") -> Tuple[Dict, pd.DataFrame]:
        """
        遗传算法优化
        
        Args:
            df: 股票数据
            indicator_name: 指标名称
            population_size: 种群大小
            generations: 进化代数
            cx_prob: 交叉概率
            mut_prob: 变异概率
            objective: 优化目标
            
        Returns:
            Tuple: (最优参数, 所有结果)
        """
        # 获取参数信息
        param_info = self.indicator_calc.get_indicator_info(indicator_name)
        if not param_info:
            return {}, pd.DataFrame()
        
        param_names = param_info["params"]
        default_values = param_info["defaults"]
        
        # 创建遗传算法工具（清理可能存在的重复创建）
        try:
            del creator.FitnessMax
        except AttributeError:
            pass
        try:
            del creator.Individual
        except AttributeError:
            pass

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # 定义基因生成函数
        for i, param_name in enumerate(param_names):
            param_range = self._get_param_range(param_name, indicator_name)
            toolbox.register(f"attr_{i}", random.randint, param_range[0], param_range[1])
        
        # 创建个体和种群
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        [getattr(toolbox, f"attr_{i}") for i in range(len(param_names))], 1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # 注册遗传操作
        toolbox.register("evaluate", self._evaluate_individual, df, indicator_name, param_names, objective)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[self._get_param_range(name, indicator_name)[0] for name in param_names],
                        up=[self._get_param_range(name, indicator_name)[1] for name in param_names], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # 初始化种群
        population = toolbox.population(n=population_size)
        
        # 运行遗传算法
        logger.info(f"开始遗传算法优化，种群大小: {population_size}, 代数: {generations}")
        
        all_results = []
        for gen in range(generations):
            # 评估种群
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # 记录当前代结果
            for ind in population:
                params = dict(zip(param_names, ind))
                all_results.append({
                    'generation': gen,
                    'parameters': params,
                    objective: ind.fitness.values[0]
                })
            
            # 选择下一代
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # 交叉和变异
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 更新种群
            population[:] = offspring
        
        # 找到最优个体
        best_individual = tools.selBest(population, 1)[0]
        best_params = dict(zip(param_names, best_individual))
        
        return best_params, pd.DataFrame(all_results)
    
    def _evaluate_individual(self, individual: list, df: pd.DataFrame, 
                           indicator_name: str, param_names: List[str],
                           objective: str = "年化收益率") -> Tuple[float]:
        """
        评估个体适应度
        """
        params = dict(zip(param_names, individual))
        result = self._evaluate_parameters(df, indicator_name, params)
        return (result.get(objective, 0),)
    
    def _evaluate_parameters(self, df: pd.DataFrame, indicator_name: str,
                           params: Dict[str, int]) -> Dict[str, Any]:
        """
        评估特定参数组合
        """
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

# 测试函数
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # 生成价格数据
    prices = np.random.normal(0.0005, 0.015, n_days).cumsum() + 100
    prices = np.maximum(prices, 1)
    
    test_df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_days)
    }, index=dates)
    
    # 测试优化器
    optimizer = ParameterOptimizer(n_jobs=2)
    
    print("测试网格搜索优化MACD参数...")
    best_params, results_df = optimizer.optimize_parameters(
        test_df, "MACD", "grid_search"
    )
    
    print(f"最优参数: {best_params}")
    print(f"最佳年化收益率: {results_df['年化收益率'].max():.2f}%")
    
    if not results_df.empty:
        print("\n前5个最佳参数组合:")
        top_results = results_df.nlargest(5, '年化收益率')
        for i, (idx, row) in enumerate(top_results.iterrows()):
            print(f"{i+1}. {row['parameters']} -> {row['年化收益率']:.2f}%")