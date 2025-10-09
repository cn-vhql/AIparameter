#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
风险管理模块
实现仓位管理、止损止盈、风险控制等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    EXTREME = "极高风险"

class RiskManager:
    """风险管理器"""
    
    def __init__(self, 
                 max_position_size: float = 1.0,
                 max_loss_per_trade: float = 0.02,
                 max_drawdown: float = 0.15,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 max_daily_loss: float = 0.03):
        """
        初始化风险管理器
        
        Args:
            max_position_size: 最大仓位比例
            max_loss_per_trade: 单笔交易最大亏损比例
            max_drawdown: 最大回撤限制
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            max_daily_loss: 每日最大亏损限制
        """
        self.max_position_size = max_position_size
        self.max_loss_per_trade = max_loss_per_trade
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
        
        # 风险记录
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.risk_violations = []
        
    def calculate_position_size(self, 
                              available_capital: float,
                              entry_price: float,
                              stop_loss_price: float,
                              risk_per_share: float = None) -> float:
        """
        计算合适的仓位大小
        
        Args:
            available_capital: 可用资金
            entry_price: 入场价格
            stop_loss_price: 止损价格
            risk_per_share: 每股风险金额
            
        Returns:
            float: 建议的持仓数量
        """
        try:
            # 计算每股风险
            if risk_per_share is None:
                risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share <= 0:
                logger.warning("风险金额为零或负数，使用默认仓位大小")
                return available_capital * self.max_position_size / entry_price
            
            # 基于风险金额计算最大仓位
            max_risk_capital = available_capital * self.max_loss_per_trade
            max_shares_by_risk = max_risk_capital / risk_per_share
            
            # 基于最大仓位比例计算
            max_shares_by_position = available_capital * self.max_position_size / entry_price
            
            # 取较小值
            recommended_shares = min(max_shares_by_risk, max_shares_by_position)
            
            logger.info(f"风险管理建议: 持仓数量={recommended_shares:.2f}, "
                       f"风险金额={risk_per_share:.2f}, "
                       f"预期最大亏损={max_risk_capital:.2f}")
            
            return recommended_shares
            
        except Exception as e:
            logger.error(f"计算仓位大小时发生错误: {str(e)}")
            return available_capital * self.max_position_size / entry_price
    
    def check_entry_risk(self, 
                        current_price: float,
                        entry_price: float,
                        stop_loss_price: float,
                        current_equity: float) -> Tuple[bool, str]:
        """
        检查入场风险
        
        Args:
            current_price: 当前价格
            entry_price: 计划入场价格
            stop_loss_price: 止损价格
            current_equity: 当前权益
            
        Returns:
            Tuple[bool, str]: (是否允许入场, 拒绝原因)
        """
        try:
            # 检查止损合理性
            if stop_loss_price >= entry_price:
                return False, "止损价格必须低于入场价格"
            
            # 计算潜在亏损比例
            potential_loss_pct = abs(entry_price - stop_loss_price) / entry_price
            
            if potential_loss_pct > self.max_loss_per_trade:
                return False, f"潜在亏损比例{potential_loss_pct:.2%}超过限制{self.max_loss_per_trade:.2%}"
            
            # 检查当前回撤
            if self.current_drawdown > self.max_drawdown:
                return False, f"当前回撤{self.current_drawdown:.2%}超过最大限制{self.max_drawdown:.2%}"
            
            # 检查每日亏损
            if self.daily_pnl < -current_equity * self.max_daily_loss:
                return False, f"今日亏损{abs(self.daily_pnl):.2f}已达到每日限制"
            
            return True, "风险检查通过"
            
        except Exception as e:
            logger.error(f"入场风险检查时发生错误: {str(e)}")
            return False, f"风险检查失败: {str(e)}"
    
    def calculate_stop_loss(self, 
                           entry_price: float,
                           atr: float = None,
                           method: str = "fixed") -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            atr: 平均真实波幅
            method: 止损方法 ("fixed", "atr", "percentage")
            
        Returns:
            float: 止损价格
        """
        try:
            if method == "fixed":
                return entry_price * (1 - self.stop_loss_pct)
            elif method == "atr" and atr is not None:
                return entry_price - (2 * atr)  # 2倍ATR止损
            elif method == "percentage":
                return entry_price * (1 - self.stop_loss_pct)
            else:
                return entry_price * (1 - self.stop_loss_pct)
                
        except Exception as e:
            logger.error(f"计算止损价格时发生错误: {str(e)}")
            return entry_price * (1 - self.stop_loss_pct)
    
    def calculate_take_profit(self, 
                            entry_price: float,
                            atr: float = None,
                            method: str = "fixed") -> float:
        """
        计算止盈价格
        
        Args:
            entry_price: 入场价格
            atr: 平均真实波幅
            method: 止盈方法 ("fixed", "atr", "percentage", "risk_reward")
            
        Returns:
            float: 止盈价格
        """
        try:
            if method == "fixed":
                return entry_price * (1 + self.take_profit_pct)
            elif method == "atr" and atr is not None:
                return entry_price + (3 * atr)  # 3倍ATR止盈
            elif method == "percentage":
                return entry_price * (1 + self.take_profit_pct)
            elif method == "risk_reward":
                # 风险回报比1:2
                risk_amount = entry_price * self.stop_loss_pct
                return entry_price + (2 * risk_amount)
            else:
                return entry_price * (1 + self.take_profit_pct)
                
        except Exception as e:
            logger.error(f"计算止盈价格时发生错误: {str(e)}")
            return entry_price * (1 + self.take_profit_pct)
    
    def update_risk_metrics(self, 
                          current_equity: float,
                          daily_pnl: float = None):
        """
        更新风险指标
        
        Args:
            current_equity: 当前权益
            daily_pnl: 每日盈亏
        """
        try:
            # 更新峰值权益
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            # 计算当前回撤
            if self.peak_equity > 0:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            # 更新每日盈亏
            if daily_pnl is not None:
                self.daily_pnl = daily_pnl
            
            # 检查风险违规
            self._check_risk_violations(current_equity)
            
        except Exception as e:
            logger.error(f"更新风险指标时发生错误: {str(e)}")
    
    def _check_risk_violations(self, current_equity: float):
        """检查风险违规"""
        violations = []
        
        # 检查回撤违规
        if self.current_drawdown > self.max_drawdown:
            violations.append({
                'type': '回撤超限',
                'current': self.current_drawdown,
                'limit': self.max_drawdown,
                'timestamp': pd.Timestamp.now()
            })
        
        # 检查每日亏损违规
        if abs(self.daily_pnl) > current_equity * self.max_daily_loss:
            violations.append({
                'type': '每日亏损超限',
                'current': abs(self.daily_pnl),
                'limit': current_equity * self.max_daily_loss,
                'timestamp': pd.Timestamp.now()
            })
        
        # 记录违规
        if violations:
            self.risk_violations.extend(violations)
            logger.warning(f"检测到风险违规: {len(violations)}项")
    
    def get_risk_level(self, current_equity: float) -> RiskLevel:
        """
        获取当前风险等级
        
        Args:
            current_equity: 当前权益
            
        Returns:
            RiskLevel: 风险等级
        """
        drawdown_ratio = self.current_drawdown
        
        if drawdown_ratio < 0.05:
            return RiskLevel.LOW
        elif drawdown_ratio < 0.10:
            return RiskLevel.MEDIUM
        elif drawdown_ratio < 0.15:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        获取风险报告
        
        Returns:
            Dict: 风险报告
        """
        return {
            'current_drawdown': f"{self.current_drawdown:.2%}",
            'peak_equity': f"{self.peak_equity:,.2f}",
            'daily_pnl': f"{self.daily_pnl:,.2f}",
            'risk_level': self.get_risk_level(self.peak_equity).value,
            'violations_count': len(self.risk_violations),
            'max_position_size': f"{self.max_position_size:.1%}",
            'max_loss_per_trade': f"{self.max_loss_per_trade:.1%}",
            'max_drawdown_limit': f"{self.max_drawdown:.1%}",
            'recent_violations': self.risk_violations[-5:] if self.risk_violations else []
        }
    
    def reset_daily_metrics(self):
        """重置每日指标"""
        self.daily_pnl = 0.0
        logger.info("已重置每日风险指标")
    
    def adjust_risk_parameters(self, market_volatility: float):
        """
        根据市场波动率调整风险参数
        
        Args:
            market_volatility: 市场波动率
        """
        try:
            # 根据波动率调整止损和止盈
            if market_volatility > 0.03:  # 高波动
                self.stop_loss_pct = min(0.08, self.stop_loss_pct * 1.5)
                self.take_profit_pct = max(0.15, self.take_profit_pct * 1.5)
                logger.info(f"高波动市场，调整止损={self.stop_loss_pct:.1%}, 止盈={self.take_profit_pct:.1%}")
            elif market_volatility < 0.01:  # 低波动
                self.stop_loss_pct = max(0.03, self.stop_loss_pct * 0.8)
                self.take_profit_pct = min(0.08, self.take_profit_pct * 0.8)
                logger.info(f"低波动市场，调整止损={self.stop_loss_pct:.1%}, 止盈={self.take_profit_pct:.1%}")
                
        except Exception as e:
            logger.error(f"调整风险参数时发生错误: {str(e)}")

# 测试函数
if __name__ == "__main__":
    # 创建风险管理器
    risk_manager = RiskManager()
    
    # 测试仓位计算
    position_size = risk_manager.calculate_position_size(
        available_capital=100000,
        entry_price=100,
        stop_loss_price=95
    )
    print(f"建议持仓数量: {position_size:.2f}")
    
    # 测试入场风险检查
    can_enter, reason = risk_manager.check_entry_risk(
        current_price=100,
        entry_price=100,
        stop_loss_price=95,
        current_equity=100000
    )
    print(f"是否允许入场: {can_enter}, 原因: {reason}")
    
    # 测试止损止盈计算
    stop_loss = risk_manager.calculate_stop_loss(entry_price=100)
    take_profit = risk_manager.calculate_take_profit(entry_price=100)
    print(f"止损价格: {stop_loss:.2f}, 止盈价格: {take_profit:.2f}")
    
    # 测试风险报告
    risk_manager.update_risk_metrics(current_equity=95000, daily_pnl=-2000)
    report = risk_manager.get_risk_report()
    print("风险报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")