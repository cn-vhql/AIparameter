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
        # 保存最优权益曲线（兼容前端调用）
        self._best_equity_curve = None
    
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
            # 使用内置遗传算法
            return self._genetic_algorithm_optimization(df, indicator_name, **kwargs)
        else:
            logger.error(f"不支持的优化算法: {algorithm}")
            return {}, pd.DataFrame()
    
    def _grid_search_optimization(self, df: pd.DataFrame, indicator_name: str,
                                param_ranges: Dict[str, List] = None,
                                objective: str = "年化收益率",
                                progress_callback: callable = None,
                                **kwargs) -> Tuple[Dict, pd.DataFrame]:
        """
        网格搜索优化
        
        Args:
            df: 股票数据
            indicator_name: 指标名称
            param_ranges: 参数范围字典
            objective: 优化目标指标
            progress_callback: 进度回调函数，接受当前进度和总数两个参数
            **kwargs: 其他参数
            
        Returns:
            Tuple: (最优参数, 所有结果)
        """
        try:
            # 获取默认参数范围
            if param_ranges is None:
                param_ranges = self._get_default_param_ranges(indicator_name)
            
            # 生成所有参数组合
            param_names = list(param_ranges.keys())
            param_values = [list(range(start, end + 1)) for start, end in param_ranges.values()]
            logger.info(f"参数范围: {dict(zip(param_names, param_values))}")
            param_combinations = list(product(*param_values))
            total_combinations = len(param_combinations)
            
            logger.info(f"开始网格搜索，共 {total_combinations} 种参数组合")
            
            def evaluate_single(params):
                """评估单个参数组合"""
                try:
                    param_dict = dict(zip(param_names, params))
                    df_copy = df.copy()

                    # 计算技术指标
                    df_with_indicator = self.indicator_calc.calculate_indicator(df_copy, indicator_name, param_dict)

                    # 检查信号生成
                    if 'signal' not in df_with_indicator.columns:
                        logger.warning(f"参数 {param_dict} 未生成信号列")
                        return None

                    signal_count = (df_with_indicator['signal'] != 0).sum()
                    if signal_count == 0:
                        logger.debug(f"参数 {param_dict} 没有生成交易信号")
                        # 返回空结果但不是None，以便统计
                        empty_result = {
                            '年化收益率': 0.0,
                            '最大回撤': 0.0,
                            '夏普比率': 0.0,
                            '交易次数': 0,
                            **param_dict
                        }
                        return empty_result

                    # 执行回测
                    results = self.backtest_engine.run_backtest(df_with_indicator)

                    if isinstance(results, dict):
                        results.update(param_dict)
                        # 确保所有必要的指标都存在
                        for key in ['年化收益率', '最大回撤', '夏普比率', '交易次数']:
                            if key not in results:
                                results[key] = 0.0
                        return results
                    else:
                        logger.warning(f"参数 {param_dict} 回测结果格式异常")
                        return None

                except Exception as e:
                    logger.warning(f"参数评估失败 {params}: {str(e)}")
                    return None
            
            # 并行计算所有参数组合
            results = []
            processed = 0
            batch_size = min(100, total_combinations)
            
            for i in range(0, total_combinations, batch_size):
                batch = param_combinations[i:i + batch_size]
                
                try:
                    batch_results = Parallel(n_jobs=self.n_jobs)(
                        delayed(evaluate_single)(params)
                        for params in batch
                    )
                    
                    # 过滤有效结果
                    valid_results = [r for r in batch_results if r is not None]
                    results.extend(valid_results)
                    
                except Exception as batch_error:
                    logger.error(f"批次处理错误: {str(batch_error)}")
                    continue
                finally:
                    processed += len(batch)
                    if progress_callback:
                        progress_callback(processed, total_combinations)
            
            # 处理结果
            if results:
                results_df = pd.DataFrame(results)
                
                if not results_df.empty and objective in results_df.columns:
                    # 按目标指标排序
                    results_df = results_df.sort_values(objective, ascending=False)
                    
                    # 获取最优参数
                    best_row = results_df.iloc[0]
                    best_params = {param: best_row[param] for param in param_names}
                    
                    logger.info(f"找到最优参数组合: {best_params}")
                    # 运行回测以获取最优权益曲线的完整结果并保存
                    try:
                        # 使用最优参数重新计算指标并请求完整回测结果
                        df_with_indicator = self.indicator_calc.calculate_indicator(df.copy(), indicator_name, best_params)
                        full_results = self.backtest_engine.run_backtest(df_with_indicator, return_full=True)
                        equity_curve = None
                        if isinstance(full_results, dict) and 'equity_curve' in full_results:
                            equity_df = pd.DataFrame(full_results['equity_curve'])
                            equity_df.set_index('date', inplace=True)
                            # 添加基准（使用首个close不变的基准收益）
                            if 'price' in equity_df.columns:
                                equity_df['benchmark'] = equity_df['price'] / equity_df['price'].iloc[0] * self.backtest_engine.initial_capital
                            else:
                                equity_df['benchmark'] = np.nan
                            equity_curve = equity_df
                        self._best_equity_curve = equity_curve
                    except Exception:
                        self._best_equity_curve = None

                    return best_params, results_df
            
            logger.error("未找到有效的优化结果")
            return {}, pd.DataFrame()
            
        except Exception as e:
            logger.error(f"优化过程发生错误: {str(e)}")
            return {}, pd.DataFrame()
        eval_data = []
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            eval_data.append((df.copy(), indicator_name, param_dict))
        
        # 分批处理
        batch_size = min(100, total_combinations)  # 减小批次大小，提高稳定性
        results = []
        processed = 0
        
        try:
            for i in range(0, total_combinations, batch_size):
                # 获取当前批次的数据
                batch = eval_data[i:i + batch_size]
                
                try:
                    # 并行计算当前批次
                    batch_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                        delayed(evaluate_parameters_wrapper)(data)
                        for data in batch
                    )
                    
                    # 过滤掉无效结果
                    valid_results = [r for r in batch_results if r is not None and isinstance(r, dict)]
                    results.extend(valid_results)
                    
                except Exception as batch_error:
                    logger.error(f"批次处理错误（{i}/{total_combinations}）: {str(batch_error)}")
                    continue
                
                finally:
                    # 更新进度
                    processed += len(batch)
                    if progress_callback:
                        progress_callback(processed, total_combinations)
            
            if not results:
                logger.error("并行计算没有产生任何有效结果")
                return {}, pd.DataFrame()
        except Exception as e:
            logger.error(f"并行计算过程中发生错误: {str(e)}")
            if progress_callback:
                progress_callback(total_combinations, total_combinations)  # 完成进度条
            return {}, pd.DataFrame()
        
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
                                      param_ranges: Dict[str, List] = None,
                                      population_size: int = 20,
                                      generations: int = 5,
                                      objective: str = "年化收益率") -> Tuple[Dict, pd.DataFrame]:
        """
        简化版遗传算法优化（避免DEAP依赖）
        
        Args:
            df: 股票数据
            indicator_name: 指标名称
            param_ranges: 参数范围
            population_size: 种群大小
            generations: 进化代数
            objective: 优化目标
            
        Returns:
            Tuple: (最优参数, 所有结果)
        """
        # 获取参数信息
        param_info = self.indicator_calc.get_indicator_info(indicator_name)
        if not param_info:
            return {}, pd.DataFrame()

        param_names = param_info["params"]

        # 如果没有提供参数范围，使用默认范围
        if param_ranges is None:
            param_ranges = {}
            for param_name in param_names:
                min_val, max_val = self._get_param_range(param_name, indicator_name)
                param_ranges[param_name] = list(range(min_val, max_val + 1))

        # 转换为遗传算法需要的格式
        genetic_param_ranges = []
        for param_name in param_names:
            if param_name in param_ranges:
                genetic_param_ranges.append((min(param_ranges[param_name]), max(param_ranges[param_name])))
            else:
                min_val, max_val = self._get_param_range(param_name, indicator_name)
                genetic_param_ranges.append((min_val, max_val))

        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = []
            for min_val, max_val in genetic_param_ranges:
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
                        for min_val, max_val in genetic_param_ranges:
                            selected.append(random.randint(min_val, max_val))
                else:
                    # 随机生成新个体
                    selected = []
                    for min_val, max_val in genetic_param_ranges:
                        selected.append(random.randint(min_val, max_val))

                # 变异
                if random.random() < 0.3:  # 30%变异率
                    for i in range(len(selected)):
                        if random.random() < 0.2:  # 20%每个基因变异率
                            min_val, max_val = genetic_param_ranges[i]
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

    def get_best_equity_curve(self) -> Optional[pd.DataFrame]:
        """返回上次优化保存的最优权益曲线（DataFrame）或 None"""
        return self._best_equity_curve

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