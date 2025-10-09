#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工具类模块
包含项目通用的工具函数和类
"""

import os
import sys
import logging
from typing import Optional, Any, Dict
from pathlib import Path

class ProjectUtils:
    """项目工具类"""
    
    @staticmethod
    def setup_project_path() -> None:
        """设置项目路径"""
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.absolute()
        
        # 将项目根目录添加到Python路径
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
    @staticmethod
    def setup_logging(level: int = logging.INFO) -> None:
        """
        设置日志配置
        
        Args:
            level: 日志级别
        """
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict: 配置参数字典
        """
        try:
            from config import (
                TRADE_CONFIG,
                INDICATOR_PARAMS,
                OPTIMIZER_CONFIG,
                BACKTEST_CONFIG
            )
            
            return {
                "trade": TRADE_CONFIG,
                "indicators": INDICATOR_PARAMS,
                "optimizer": OPTIMIZER_CONFIG,
                "backtest": BACKTEST_CONFIG
            }
        except ImportError as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            return {}
