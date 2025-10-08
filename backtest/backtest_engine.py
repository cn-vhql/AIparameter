#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回测引擎模块
实现完整的交易逻辑、仓位管理和绩效计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import empyrical as ep

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.0003,  # 默认万三手续费
                 slippage_rate: float = 0.0001):   # 默认万分之一滑点
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.reset()
    
    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.positions = 0
        self.trades = []
        self.equity_curve = []
        self.current_date = None
        self.current_price = None
    
    def run_backtest(self, df: pd.DataFrame, signal_column: str = 'signal', 
                    price_column: str = 'close') -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            df: 包含信号和价格数据的DataFrame
            signal_column: 信号列名
            price_column: 价格列名
            
        Returns:
            Dict: 包含回测结果的字典
        """
        self.reset()
        
        if signal_column not in df.columns:
            logger.error(f"信号列 {signal_column} 不存在")
            return {}
        
        if price_column not in df.columns:
            logger.error(f"价格列 {price_column} 不存在")
            return {}
        
        # 按日期排序
        df = df.sort_index()
        
        results = {
            'trades': [],
            'equity_curve': [],
            'performance_metrics': {}
        }
        
        # 执行回测
        for date, row in df.iterrows():
            self.current_date = date
            self.current_price = row[price_column]
            
            # 处理交易信号
            signal = row.get(signal_column, 0)
            if signal != 0:
                self._execute_trade(signal)
            
            # 记录权益曲线
            current_equity = self.cash + self.positions * self.current_price
            results['equity_curve'].append({
                'date': date,
                'equity': current_equity,
                'cash': self.cash,
                'positions': self.positions,
                'price': self.current_price
            })
        
        # 计算绩效指标
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df.set_index('date', inplace=True)
        returns = equity_df['equity'].pct_change().fillna(0)
        
        results['performance_metrics'] = self._calculate_performance_metrics(returns, equity_df)
        results['trades'] = self.trades
        
        return results
    
    def _execute_trade(self, signal: int):
        """执行交易"""
        if signal == 1 and self.positions == 0:  # 买入信号且空仓
            self._buy()
        elif signal == -1 and self.positions > 0:  # 卖出信号且持有仓位
            self._sell()
    
    def _buy(self):
        """执行买入操作"""
        if self.cash <= 0:
            return
        
        # 计算可买入数量（全仓买入）
        commission = self.cash * self.commission_rate
        slippage = self.cash * self.slippage_rate
        available_cash = self.cash - commission - slippage
        
        buy_quantity = available_cash / self.current_price
        buy_amount = buy_quantity * self.current_price
        
        # 更新仓位和现金
        self.positions = buy_quantity
        self.cash = 0
        
        # 记录交易
        self.trades.append({
            'date': self.current_date,
            'action': 'BUY',
            'price': self.current_price,
            'quantity': buy_quantity,
            'amount': buy_amount,
            'commission': commission,
            'slippage': slippage
        })
    
    def _sell(self):
        """执行卖出操作"""
        if self.positions <= 0:
            return
        
        # 计算卖出金额
        sell_amount = self.positions * self.current_price
        commission = sell_amount * self.commission_rate
        slippage = sell_amount * self.slippage_rate
        net_proceeds = sell_amount - commission - slippage
        
        # 更新仓位和现金
        self.cash += net_proceeds
        self.positions = 0
        
        # 记录交易
        self.trades.append({
            'date': self.current_date,
            'action': 'SELL',
            'price': self.current_price,
            'quantity': self.positions,
            'amount': sell_amount,
            'commission': commission,
            'slippage': slippage,
            'net_proceeds': net_proceeds
        })
    
    def _calculate_performance_metrics(self, returns: pd.Series,
                                     equity_df: pd.DataFrame) -> Dict[str, float]:
        """计算绩效指标"""
        if len(returns) == 0:
            return {}

        # 基础指标
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1) * 100

        # 手动计算年化收益率，避免empyrical与NumPy 2.0兼容性问题
        days = len(returns)
        if days > 0:
            daily_return = returns.mean()
            annual_return = (1 + daily_return) ** 252 - 1  # 假设一年252个交易日
            annual_return = annual_return * 100
        else:
            annual_return = 0

        # 风险指标 - 使用手动计算避免empyrical问题
        volatility = returns.std() * (252 ** 0.5) * 100  # 年化波动率

        # 手动计算夏普比率
        if volatility > 0:
            sharpe_ratio = annual_return / volatility
        else:
            sharpe_ratio = 0

        # 手动计算最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # 手动计算索提诺比率（简化版本）
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_volatility = downside_returns.std() * (252 ** 0.5)
            if downside_volatility > 0:
                sortino_ratio = annual_return / (downside_volatility * 100)
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
        
        # 交易相关指标
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('net_proceeds', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 持仓时间分析
        hold_times = self._analyze_hold_times()

        metrics = {
            '年化收益率': round(annual_return, 2),  # 修改键名以匹配优化器期望
            '总收益率(%)': round(total_return, 2),
            '年化收益率(%)': round(annual_return, 2),
            '年化波动率(%)': round(volatility, 2),
            '夏普比率': round(sharpe_ratio, 2),
            '索提诺比率': round(sortino_ratio, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '总交易次数': total_trades,
            '胜率(%)': round(win_rate, 2),
            '平均持仓天数': round(hold_times['avg_hold_days'], 2),
            '最大持仓天数': hold_times['max_hold_days'],
            '最小持仓天数': hold_times['min_hold_days'],
            '初始资金': self.initial_capital,
            '最终权益': round(equity_df['equity'].iloc[-1], 2),
            '净利润': round(equity_df['equity'].iloc[-1] - self.initial_capital, 2)
        }
        
        return metrics
    
    def _analyze_hold_times(self) -> Dict[str, Any]:
        """分析持仓时间"""
        if not self.trades:
            return {'avg_hold_days': 0, 'max_hold_days': 0, 'min_hold_days': 0}
        
        hold_times = []
        buy_dates = []
        
        for trade in self.trades:
            if trade['action'] == 'BUY':
                buy_dates.append(trade['date'])
            elif trade['action'] == 'SELL' and buy_dates:
                buy_date = buy_dates.pop(0)
                hold_days = (trade['date'] - buy_date).days
                hold_times.append(hold_days)
        
        if hold_times:
            return {
                'avg_hold_days': np.mean(hold_times),
                'max_hold_days': np.max(hold_times),
                'min_hold_days': np.min(hold_times)
            }
        else:
            return {'avg_hold_days': 0, 'max_hold_days': 0, 'min_hold_days': 0}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成回测报告"""
        metrics = results.get('performance_metrics', {})
        
        report = f"""
        📊 回测绩效报告
        =====================
        
        资金相关:
        ---------
        初始资金: {metrics.get('初始资金', 0):,.2f} 元
        最终权益: {metrics.get('最终权益', 0):,.2f} 元
        净利润: {metrics.get('净利润', 0):,.2f} 元
        总收益率: {metrics.get('总收益率(%)', 0):.2f}%
        年化收益率: {metrics.get('年化收益率(%)', 0):.2f}%
        
        风险指标:
        ---------
        年化波动率: {metrics.get('年化波动率(%)', 0):.2f}%
        最大回撤: {metrics.get('最大回撤(%)', 0):.2f}%
        夏普比率: {metrics.get('夏普比率', 0):.2f}
        索提诺比率: {metrics.get('索提诺比率', 0):.2f}
        
        交易统计:
        ---------
        总交易次数: {metrics.get('总交易次数', 0)}
        胜率: {metrics.get('胜率(%)', 0):.2f}%
        平均持仓天数: {metrics.get('平均持仓天数', 0):.1f} 天
        最大持仓天数: {metrics.get('最大持仓天数', 0)} 天
        最小持仓天数: {metrics.get('最小持仓天数', 0)} 天
        
        交易明细:
        ---------
        """
        
        for i, trade in enumerate(results.get('trades', [])[:10]):  # 显示前10笔交易
            report += f"{i+1}. {trade['date'].strftime('%Y-%m-%d')} {trade['action']} {trade['quantity']:.2f}股 @ {trade['price']:.2f}\n"
        
        if len(results.get('trades', [])) > 10:
            report += f"... 还有 {len(results.get('trades', [])) - 10} 笔交易未显示\n"
        
        return report

# 测试函数
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    n_days = len(dates)
    
    # 生成价格数据
    prices = np.random.normal(0.001, 0.02, n_days).cumsum() + 100
    prices = np.maximum(prices, 1)  # 确保价格为正
    
    # 生成随机信号
    signals = np.random.choice([-1, 0, 1], n_days, p=[0.1, 0.8, 0.1])
    
    test_df = pd.DataFrame({
        'close': prices,
        'signal': signals
    }, index=dates)
    
    # 运行回测
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(test_df, 'signal', 'close')
    
    # 输出结果
    print("回测完成!")
    print(f"总交易次数: {len(results['trades'])}")
    print(f"最终权益: {results['performance_metrics'].get('最终权益', 0):,.2f}")
    print(f"总收益率: {results['performance_metrics'].get('总收益率(%)', 0):.2f}%")
    
    # 生成报告
    report = engine.generate_report(results)
    print(report)