#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版回测引擎
集成风险管理、多品种、多时间框架支持
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import empyrical as ep
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from risk_management.risk_manager import RiskManager, RiskLevel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Position:
    """持仓类"""
    
    def __init__(self, symbol: str, entry_price: float, quantity: float, 
                 entry_date: datetime, stop_loss: float = None, take_profit: float = None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_date = entry_date
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.holding_days = 0
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.exit_reason = None  # 'stop_loss', 'take_profit', 'signal', 'risk_limit'
    
    def update(self, current_price: float, current_date: datetime):
        """更新持仓状态"""
        if self.exit_price is not None:
            return
        
        self.holding_days = (current_date - self.entry_date).days
        current_pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        # 更新最大盈利和最大亏损
        self.max_profit = max(self.max_profit, current_pnl_pct)
        self.max_loss = min(self.max_loss, current_pnl_pct)
    
    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """检查是否应该平仓"""
        if self.stop_loss and current_price <= self.stop_loss:
            return True, 'stop_loss'
        if self.take_profit and current_price >= self.take_profit:
            return True, 'take_profit'
        return False, None
    
    def close(self, exit_price: float, exit_date: datetime, exit_reason: str = None):
        """平仓"""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_reason = exit_reason
        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price

class EnhancedBacktestEngine:
    """增强版回测引擎"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.0001,
                 risk_manager: RiskManager = None):
        """
        初始化增强版回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            risk_manager: 风险管理器
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_manager = risk_manager or RiskManager()
        
        # 回测状态
        self.reset()
    
    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> Position
        self.closed_positions = []
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.current_date = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # 重置风险管理器
        self.risk_manager.reset_daily_metrics()
    
    def run_multi_symbol_backtest(self, 
                                 data_dict: Dict[str, pd.DataFrame],
                                 signals_dict: Dict[str, pd.DataFrame],
                                 progress_callback: callable = None) -> Dict[str, Any]:
        """
        运行多品种回测
        
        Args:
            data_dict: 品种数据字典 {symbol: DataFrame}
            signals_dict: 信号数据字典 {symbol: DataFrame}
            progress_callback: 进度回调函数
            
        Returns:
            Dict: 回测结果
        """
        try:
            self.reset()
            
            # 获取所有交易日期
            all_dates = set()
            for df in data_dict.values():
                all_dates.update(df.index)
            all_dates = sorted(list(all_dates))
            
            logger.info(f"开始多品种回测，共{len(data_dict)}个品种，{len(all_dates)}个交易日")
            
            for i, date in enumerate(all_dates):
                self.current_date = date
                
                # 更新所有持仓
                self._update_positions(date, data_dict)
                
                # 处理每个品种的信号
                for symbol, data in data_dict.items():
                    if date in data.index:
                        self._process_symbol_signals(
                            symbol, date, data, signals_dict.get(symbol)
                        )
                
                # 记录权益曲线
                self._record_equity_curve(date, data_dict)
                
                # 更新风险指标
                current_equity = self._calculate_total_equity(data_dict)
                self.risk_manager.update_risk_metrics(current_equity)
                
                # 进度回调
                if progress_callback:
                    progress_callback(i + 1, len(all_dates))
            
            # 计算最终结果
            results = self._calculate_final_results(data_dict)
            
            logger.info(f"多品种回测完成，总交易次数: {len(self.closed_positions)}")
            
            return results
            
        except Exception as e:
            logger.error(f"多品种回测发生错误: {str(e)}")
            return self._get_empty_results()
    
    def _process_symbol_signals(self, symbol: str, date: datetime, 
                               data: pd.DataFrame, signals: pd.DataFrame = None):
        """处理单个品种的信号"""
        try:
            if symbol in self.positions:
                # 已有持仓，检查是否需要平仓
                position = self.positions[symbol]
                current_price = data.loc[date, 'close']
                
                # 更新持仓
                position.update(current_price, date)
                
                # 检查止损止盈
                should_close, reason = position.should_close(current_price)
                if should_close:
                    self._close_position(symbol, current_price, date, reason)
                elif signals is not None and date in signals.index:
                    # 检查信号平仓
                    signal = signals.loc[date, 'signal']
                    if signal < 0:  # 卖出信号
                        self._close_position(symbol, current_price, date, 'signal')
            
            elif signals is not None and date in signals.index:
                # 无持仓，检查买入信号
                signal = signals.loc[date, 'signal']
                if signal > 0:  # 买入信号
                    current_price = data.loc[date, 'close']
                    self._open_position(symbol, current_price, date, data)
                    
        except Exception as e:
            logger.error(f"处理{symbol}信号时发生错误: {str(e)}")
    
    def _open_position(self, symbol: str, price: float, date: datetime, data: pd.DataFrame):
        """开仓"""
        try:
            # 计算ATR用于止损止盈
            atr = self._calculate_atr(data, date)
            
            # 计算止损止盈价格
            stop_loss = self.risk_manager.calculate_stop_loss(price, atr, "atr")
            take_profit = self.risk_manager.calculate_take_profit(price, atr, "atr")
            
            # 风险检查
            can_enter, reason = self.risk_manager.check_entry_risk(
                price, price, stop_loss, self._calculate_total_equity({symbol: data})
            )
            
            if not can_enter:
                logger.warning(f"{symbol}开仓被拒绝: {reason}")
                return
            
            # 计算仓位大小
            position_size = self.risk_manager.calculate_position_size(
                self.cash, price, stop_loss
            )
            
            if position_size * price > self.cash:
                logger.warning(f"{symbol}资金不足，无法开仓")
                return
            
            # 计算手续费和滑点
            commission = position_size * price * self.commission_rate
            slippage = position_size * price * self.slippage_rate
            total_cost = position_size * price + commission + slippage
            
            if total_cost > self.cash:
                logger.warning(f"{symbol}考虑手续费后资金不足")
                return
            
            # 创建持仓
            position = Position(symbol, price, position_size, date, stop_loss, take_profit)
            self.positions[symbol] = position
            
            # 更新现金
            self.cash -= total_cost
            
            # 记录交易
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'quantity': position_size,
                'amount': position_size * price,
                'commission': commission,
                'slippage': slippage,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
            logger.info(f"{symbol}开仓成功: 价格={price:.2f}, 数量={position_size:.2f}")
            
        except Exception as e:
            logger.error(f"{symbol}开仓失败: {str(e)}")
    
    def _close_position(self, symbol: str, price: float, date: datetime, reason: str):
        """平仓"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # 计算手续费和滑点
            sell_amount = position.quantity * price
            commission = sell_amount * self.commission_rate
            slippage = sell_amount * self.slippage_rate
            net_proceeds = sell_amount - commission - slippage
            
            # 平仓
            position.close(price, date, reason)
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # 更新现金
            self.cash += net_proceeds
            
            # 更新交易统计
            self.total_trades += 1
            if position.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # 记录交易
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'quantity': position.quantity,
                'amount': sell_amount,
                'commission': commission,
                'slippage': slippage,
                'net_proceeds': net_proceeds,
                'pnl': position.pnl,
                'pnl_pct': position.pnl_pct,
                'exit_reason': reason,
                'holding_days': position.holding_days
            })
            
            logger.info(f"{symbol}平仓: 价格={price:.2f}, 盈亏={position.pnl:.2f}, 原因={reason}")
            
        except Exception as e:
            logger.error(f"{symbol}平仓失败: {str(e)}")
    
    def _update_positions(self, date: datetime, data_dict: Dict[str, pd.DataFrame]):
        """更新所有持仓"""
        for symbol, position in list(self.positions.items()):
            if symbol in data_dict and date in data_dict[symbol].index:
                current_price = data_dict[symbol].loc[date, 'close']
                position.update(current_price, date)
                
                # 检查止损止盈
                should_close, reason = position.should_close(current_price)
                if should_close:
                    self._close_position(symbol, current_price, date, reason)
    
    def _calculate_atr(self, data: pd.DataFrame, date: datetime, period: int = 14) -> float:
        """计算ATR"""
        try:
            # 获取最近的数据
            end_idx = data.index.get_loc(date)
            start_idx = max(0, end_idx - period + 1)
            recent_data = data.iloc[start_idx:end_idx + 1]
            
            if len(recent_data) < 2:
                return 0.0
            
            high = recent_data['high'].values
            low = recent_data['low'].values
            close = recent_data['close'].values
            
            # 计算真实波幅
            tr1 = high[1:] - low[1:]
            tr2 = abs(high[1:] - close[:-1])
            tr3 = abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            return np.mean(tr) if len(tr) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算ATR失败: {str(e)}")
            return 0.0
    
    def _calculate_total_equity(self, data_dict: Dict[str, pd.DataFrame]) -> float:
        """计算总权益"""
        total_equity = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in data_dict and self.current_date in data_dict[symbol].index:
                current_price = data_dict[symbol].loc[self.current_date, 'close']
                position_value = position.quantity * current_price
                total_equity += position_value
        
        return total_equity
    
    def _record_equity_curve(self, date: datetime, data_dict: Dict[str, pd.DataFrame]):
        """记录权益曲线"""
        total_equity = self._calculate_total_equity(data_dict)
        
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'cash': self.cash,
            'positions_value': total_equity - self.cash,
            'positions_count': len(self.positions)
        })
    
    def _calculate_final_results(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """计算最终结果"""
        try:
            if not self.equity_curve:
                return self._get_empty_results()
            
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('date', inplace=True)
            
            # 计算收益率
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
            returns = equity_df['returns']
            
            # 基础指标
            total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1) * 100
            annual_return = (1 + returns.mean()) ** 252 - 1
            annual_return = annual_return * 100
            
            # 风险指标
            volatility = returns.std() * (252 ** 0.5) * 100
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # 交易统计
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            # 持仓分析
            if self.closed_positions:
                holding_days = [pos.holding_days for pos in self.closed_positions]
                avg_holding_days = np.mean(holding_days)
                max_profit = max([pos.max_profit for pos in self.closed_positions])
                max_loss = min([pos.max_loss for pos in self.closed_positions])
            else:
                avg_holding_days = 0
                max_profit = 0
                max_loss = 0
            
            # 风险报告
            risk_report = self.risk_manager.get_risk_report()
            
            results = {
                'performance_metrics': {
                    '总收益率(%)': round(total_return, 2),
                    '年化收益率(%)': round(annual_return, 2),
                    '年化波动率(%)': round(volatility, 2),
                    '夏普比率': round(sharpe_ratio, 2),
                    '最大回撤(%)': round(max_drawdown, 2),
                    '胜率(%)': round(win_rate, 2),
                    '总交易次数': self.total_trades,
                    '盈利交易次数': self.winning_trades,
                    '亏损交易次数': self.losing_trades,
                    '平均持仓天数': round(avg_holding_days, 1),
                    '最大盈利(%)': round(max_profit * 100, 2),
                    '最大亏损(%)': round(max_loss * 100, 2),
                    '最终权益': round(equity_df['equity'].iloc[-1], 2),
                    '净利润': round(equity_df['equity'].iloc[-1] - self.initial_capital, 2)
                },
                'risk_metrics': risk_report,
                'equity_curve': equity_df.reset_index().to_dict('records'),
                'trades': self.trades,
                'positions': [pos.__dict__ for pos in self.closed_positions],
                'current_positions': [pos.__dict__ for pos in self.positions.values()]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"计算最终结果时发生错误: {str(e)}")
            return self._get_empty_results()
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """获取空结果"""
        return {
            'performance_metrics': {
                '总收益率(%)': 0.0,
                '年化收益率(%)': 0.0,
                '年化波动率(%)': 0.0,
                '夏普比率': 0.0,
                '最大回撤(%)': 0.0,
                '胜率(%)': 0.0,
                '总交易次数': 0,
                '盈利交易次数': 0,
                '亏损交易次数': 0,
                '平均持仓天数': 0.0,
                '最大盈利(%)': 0.0,
                '最大亏损(%)': 0.0,
                '最终权益': self.initial_capital,
                '净利润': 0.0
            },
            'risk_metrics': self.risk_manager.get_risk_report(),
            'equity_curve': [],
            'trades': [],
            'positions': [],
            'current_positions': []
        }
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """生成详细报告"""
        metrics = results['performance_metrics']
        risk_metrics = results['risk_metrics']
        
        report = f"""
📊 增强版回测详细报告
========================

🎯 绩效指标
-----------
初始资金: {self.initial_capital:,.2f} 元
最终权益: {metrics['最终权益']:,.2f} 元
净利润: {metrics['净利润']:,.2f} 元
总收益率: {metrics['总收益率(%)']:.2f}%
年化收益率: {metrics['年化收益率(%)']:.2f}%

⚠️ 风险指标
-----------
年化波动率: {metrics['年化波动率(%)']:.2f}%
最大回撤: {metrics['最大回撤(%)']:.2f}%
夏普比率: {metrics['夏普比率']:.2f}
当前风险等级: {risk_metrics['risk_level']}

📈 交易统计
-----------
总交易次数: {metrics['总交易次数']}
盈利交易: {metrics['盈利交易次数']}
亏损交易: {metrics['亏损交易次数']}
胜率: {metrics['胜率(%)']:.2f}%
平均持仓天数: {metrics['平均持仓天数']:.1f} 天
最大盈利: {metrics['最大盈利(%)']:.2f}%
最大亏损: {metrics['最大亏损(%)']:.2f}%

🛡️ 风险控制
-----------
当前回撤: {risk_metrics['current_drawdown']}
峰值权益: {risk_metrics['peak_equity']}
今日盈亏: {risk_metrics['daily_pnl']}
风险违规次数: {risk_metrics['violations_count']}

📋 最近交易记录
---------------
"""
        
        # 添加最近交易记录
        recent_trades = results['trades'][-10:] if results['trades'] else []
        for i, trade in enumerate(recent_trades):
            if trade['action'] == 'SELL':
                report += f"{i+1}. {trade['date'].strftime('%Y-%m-%d')} {trade['symbol']} " \
                        f"平仓 盈亏:{trade.get('pnl', 0):.2f} 原因:{trade.get('exit_reason', 'N/A')}\n"
        
        return report

# 测试函数
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    
    # 生成两个品种的数据
    data_dict = {}
    signals_dict = {}
    
    for symbol in ['STOCK_A', 'STOCK_B']:
        prices = np.random.normal(0.001, 0.02, len(dates)).cumsum() + 100
        signals = np.random.choice([-1, 0, 1], len(dates), p=[0.1, 0.8, 0.1])
        
        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        signal_df = pd.DataFrame({'signal': signals}, index=dates)
        
        data_dict[symbol] = df
        signals_dict[symbol] = signal_df
    
    # 创建风险管理器和回测引擎
    risk_manager = RiskManager(max_loss_per_trade=0.02, max_drawdown=0.15)
    engine = EnhancedBacktestEngine(initial_capital=100000, risk_manager=risk_manager)
    
    # 运行回测
    def progress_callback(current, total):
        print(f"进度: {current}/{total} ({current/total:.1%})")
    
    results = engine.run_multi_symbol_backtest(data_dict, signals_dict, progress_callback)
    
    # 输出结果
    print("\n回测完成!")
    print(f"总收益率: {results['performance_metrics']['总收益率(%)']:.2f}%")
    print(f"夏普比率: {results['performance_metrics']['夏普比率']:.2f}")
    print(f"最大回撤: {results['performance_metrics']['最大回撤(%)']:.2f}%")
    
    # 生成详细报告
    report = engine.generate_detailed_report(results)
    print(report)