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

# 导入风险管理系统
from risk_management.risk_manager import RiskManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.0003,  # 默认万三手续费
                 slippage_rate: float = 0.0001,   # 默认万分之一滑点
                 enable_risk_management: bool = True):  # 是否启用风险控制
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            enable_risk_management: 是否启用风险控制
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.enable_risk_management = enable_risk_management
        
        # 初始化风险管理器
        if self.enable_risk_management:
            self.risk_manager = RiskManager()
            logger.info("风险管理已启用")
        else:
            self.risk_manager = None
            logger.info("风险管理已禁用，使用简单交易逻辑")
        
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
                    price_column: str = 'close', return_full: bool = False) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            df: 包含信号和价格数据的DataFrame
            signal_column: 信号列名
            price_column: 价格列名
            
        Returns:
            Dict: 包含回测结果的字典
        """
        try:
            self.reset()
            
            # 验证输入数据并确保使用正确的信号列
            if signal_column != 'signal' or signal_column not in df.columns:
                logger.info("使用统一的'signal'列名")
                if 'signal' not in df.columns:
                    df['signal'] = 0
                signal_column = 'signal'
            
            if price_column not in df.columns:
                logger.error(f"价格列 {price_column} 不存在")
                return {
                    '年化收益率': 0.0,
                    '最大回撤': 0.0,
                    '夏普比率': 0.0,
                    '交易次数': 0
                }
        
            # 按日期排序并处理缺失值
            df = df.sort_index()
            df[signal_column] = df[signal_column].fillna(0)
            
            # 初始化结果字典
            results = {
                'trades': [],
                'equity_curve': [],
                'performance_metrics': {}
            }
            
            # 记录初始状态
            results['equity_curve'].append({
                'date': df.index[0],
                'equity': self.initial_capital,
                'cash': self.initial_capital,
                'positions': 0,
                'price': df[price_column].iloc[0]
            })
            
            # 执行回测
            for date, row in df.iterrows():
                try:
                    self.current_date = date
                    self.current_price = row[price_column]
                    
                    # 处理交易信号
                    signal = row.get(signal_column, 0)
                    if pd.notna(signal) and signal != 0:
                        logger.debug(f"检查交易信号: 日期={date}, 信号={signal}, 价格={self.current_price}, 当前仓位={self.positions}")

                        # 信号大于0为买入信号，小于0为卖出信号
                        if signal > 0 and self.positions == 0:  # 买入信号且空仓
                            logger.info(f"执行买入操作: 日期={date}, 价格={self.current_price}")
                            self._buy()
                        elif signal < 0 and self.positions > 0:  # 卖出信号且持有仓位
                            logger.info(f"执行卖出操作: 日期={date}, 价格={self.current_price}")
                            self._sell()
                        else:
                            logger.debug(f"信号但不符合交易条件: 信号={signal}, 仓位={self.positions}")

                        logger.debug(f"交易后状态: 现金={self.cash:.2f}, 仓位={self.positions}")
                    
                    # 记录权益曲线
                    current_equity = self.cash + self.positions * self.current_price
                    results['equity_curve'].append({
                        'date': date,
                        'equity': current_equity,
                        'cash': self.cash,
                        'positions': self.positions,
                        'price': self.current_price
                    })
                    
                except Exception as e:
                    logger.error(f"回测执行错误 at {date}: {str(e)}")
                    continue
        
            # 计算绩效指标
            if len(results['equity_curve']) > 0:
                equity_df = pd.DataFrame(results['equity_curve'])
                equity_df.set_index('date', inplace=True)

                # 计算收益率
                returns = equity_df['equity'].pct_change().fillna(0)

                # 计算绩效指标
                metrics = self._calculate_performance_metrics(returns, equity_df)
                results['performance_metrics'] = metrics
                results['trades'] = self.trades

                # 如果请求完整结果，则返回包含曲线和指标的字典
                if return_full:
                    return results

                # 否则返回简化的指标摘要，保持向后兼容
                return {
                    '年化收益率': metrics.get('年化收益率', 0.0),
                    '最大回撤': metrics.get('最大回撤(%)', 0.0),
                    '夏普比率': metrics.get('夏普比率', 0.0),
                    '交易次数': len(self.trades)
                }
            
            return {
                '年化收益率': 0.0,
                '最大回撤': 0.0,
                '夏普比率': 0.0,
                '交易次数': 0
            }
            
        except Exception as e:
            logger.error(f"回测过程发生错误: {str(e)}")
            return {
                '年化收益率': 0.0,
                '最大回撤': 0.0,
                '夏普比率': 0.0,
                '交易次数': 0
            }
    
    def _execute_trade(self, signal: int):
        """执行交易"""
        if signal == 1 and self.positions == 0:  # 买入信号且空仓
            self._buy()
        elif signal == -1 and self.positions > 0:  # 卖出信号且持有仓位
            self._sell()
    
    def _buy(self):
        """执行买入操作（集成风险控制）"""
        if self.cash <= 0:
            logger.warning("现金不足，无法买入")
            return

        # 计算交易成本
        commission = self.cash * self.commission_rate
        slippage = self.cash * self.slippage_rate
        available_cash = self.cash - commission - slippage

        if available_cash <= 0:
            logger.warning("扣除手续费后现金不足，无法买入")
            return

        # 风险控制：计算仓位大小
        if self.enable_risk_management and self.risk_manager:
            try:
                # 基于风险的仓位计算
                position_size = self.risk_manager.calculate_position_size(
                    available_cash, self.current_price, self.current_price * 0.95  # 假设5%止损
                )
                buy_quantity = position_size
                buy_amount = buy_quantity * self.current_price
                
                # 风险检查
                risk_level = self.risk_manager.assess_risk_level(
                    available_cash, buy_quantity, self.current_price
                )
                
                if risk_level in ['极高风险', '高风险']:
                    logger.warning(f"风险等级过高({risk_level})，减少仓位")
                    buy_quantity *= 0.5  # 减半仓位
                    buy_amount = buy_quantity * self.current_price
                
                logger.info(f"风险控制买入: 仓位={buy_quantity:.2f}, 风险等级={risk_level}")
                
            except Exception as e:
                logger.warning(f"风险控制计算失败，使用简单买入: {str(e)}")
                # 降级为简单全仓买入
                buy_quantity = available_cash / self.current_price
                buy_amount = buy_quantity * self.current_price
        else:
            # 简单全仓买入
            buy_quantity = available_cash / self.current_price
            buy_amount = buy_quantity * self.current_price

        # 执行交易
        old_cash = self.cash
        self.positions = buy_quantity
        self.cash = self.cash - buy_amount - commission - slippage

        # 记录交易
        trade_record = {
            'date': self.current_date,
            'action': 'BUY',
            'price': self.current_price,
            'quantity': buy_quantity,
            'amount': buy_amount,
            'commission': commission,
            'slippage': slippage,
            'cash_before': old_cash,
            'cash_after': self.cash,
            'positions_after': self.positions
        }
        
        # 记录风险控制信息
        if self.enable_risk_management and self.risk_manager:
            trade_record['risk_level'] = self.risk_manager.assess_risk_level(
                old_cash, buy_quantity, self.current_price
            )
            trade_record['position_size_pct'] = (buy_amount / old_cash * 100) if old_cash > 0 else 0
            
            # 检查风险违规
            if self.risk_manager.check_violation(
                self.cash + self.positions * self.current_price, 
                self.initial_capital
            ):
                trade_record['risk_violation'] = True
                logger.warning("检测到风险违规")
        
        self.trades.append(trade_record)
        logger.info(f"买入成功: 数量={buy_quantity:.2f}, 价格={self.current_price:.2f}, 金额={buy_amount:.2f}")
    
    def _sell(self):
        """执行卖出操作（集成风险控制）"""
        if self.positions <= 0:
            logger.warning("无持仓，无法卖出")
            return

        # 风险控制：检查是否触发止损止盈
        sell_reason = "信号卖出"  # 默认原因
        
        if self.enable_risk_management and self.risk_manager:
            try:
                # 查找对应的买入记录
                buy_price = None
                for trade in reversed(self.trades):
                    if trade['action'] == 'BUY':
                        buy_price = trade['price']
                        break
                
                if buy_price:
                    # 计算当前收益率
                    current_return = (self.current_price - buy_price) / buy_price * 100
                    
                    # 检查止损
                    stop_loss = self.risk_manager.calculate_stop_loss(buy_price)
                    if self.current_price <= stop_loss:
                        sell_reason = f"止损卖出({current_return:.2f}%)"
                        logger.info(f"触发止损: 买入价={buy_price:.2f}, 当前价={self.current_price:.2f}, 止损价={stop_loss:.2f}")
                    
                    # 检查止盈
                    take_profit = self.risk_manager.calculate_take_profit(buy_price)
                    if self.current_price >= take_profit:
                        sell_reason = f"止盈卖出({current_return:.2f}%)"
                        logger.info(f"触发止盈: 买入价={buy_price:.2f}, 当前价={self.current_price:.2f}, 止盈价={take_profit:.2f}")
                    
                    # 风险回报比检查
                    risk_reward = self.risk_manager.calculate_risk_reward_ratio(buy_price, stop_loss)
                    if risk_reward:
                        expected_return = risk_reward * 100  # 预期回报率
                        if current_return >= expected_return * 0.8:  # 达到80%预期回报
                            sell_reason = f"风险回报比卖出({current_return:.2f}%)"
                            logger.info(f"风险回报比卖出: 预期={expected_return:.2f}%, 实际={current_return:.2f}%")
                
            except Exception as e:
                logger.warning(f"风险控制检查失败: {str(e)}")

        # 计算卖出金额
        old_positions = self.positions
        sell_amount = self.positions * self.current_price
        commission = sell_amount * self.commission_rate
        slippage = sell_amount * self.slippage_rate
        net_proceeds = sell_amount - commission - slippage

        # 更新仓位和现金
        old_cash = self.cash
        self.cash += net_proceeds
        self.positions = 0

        # 记录交易
        trade_record = {
            'date': self.current_date,
            'action': 'SELL',
            'price': self.current_price,
            'quantity': old_positions,
            'amount': sell_amount,
            'commission': commission,
            'slippage': slippage,
            'net_proceeds': net_proceeds,
            'cash_before': old_cash,
            'cash_after': self.cash,
            'positions_after': self.positions,
            'sell_reason': sell_reason
        }
        
        # 记录风险控制信息
        if self.enable_risk_management and self.risk_manager:
            if buy_price:
                trade_record['buy_price'] = buy_price
                trade_record['return_pct'] = (self.current_price - buy_price) / buy_price * 100
                trade_record['stop_loss'] = self.risk_manager.calculate_stop_loss(buy_price)
                trade_record['take_profit'] = self.risk_manager.calculate_take_profit(buy_price)
            
            # 检查风险违规
            current_equity = self.cash + self.positions * self.current_price
            if self.risk_manager.check_violation(current_equity, self.initial_capital):
                trade_record['risk_violation'] = True
                logger.warning("检测到风险违规")

        self.trades.append(trade_record)
        logger.info(f"卖出成功: 数量={old_positions:.2f}, 价格={self.current_price:.2f}, 净收入={net_proceeds:.2f}, 原因={sell_reason}")
    
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