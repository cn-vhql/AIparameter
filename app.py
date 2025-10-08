#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票指标参数优化与回测系统
主应用程序入口
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
try:
    from data.data_fetcher import StockDataFetcher
    from indicators.technical_indicators import TechnicalIndicators
    from backtest.backtest_engine import BacktestEngine
    from optimization.optimizer import ParameterOptimizer
except ImportError:
    # 如果直接运行，尝试相对导入
    from .data.data_fetcher import StockDataFetcher
    from .indicators.technical_indicators import TechnicalIndicators
    from .backtest.backtest_engine import BacktestEngine
    from .optimization.optimizer import ParameterOptimizer

# 页面配置
st.set_page_config(
    page_title="股票指标参数优化系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("📈 股票指标参数优化与回测系统")
st.markdown("""
通过算法搜索最优技术指标参数组合，辅助确定股票交易策略的最佳参数配置
""")

def main():
    # 侧边栏 - 输入参数
    st.sidebar.header("📊 回测参数设置")
    
    # 股票代码输入
    stock_code = st.sidebar.text_input(
        "股票代码",
        value="000001.SZ",
        help="请输入A股股票代码，如：000001.SZ（平安银行）、600036.SH（招商银行）"
    )
    
    # 时间周期选择
    period = st.sidebar.selectbox(
        "时间周期",
        options=["日线", "周线", "月线"],
        index=0
    )
    
    # 回测时间范围
    time_range = st.sidebar.selectbox(
        "回测时间范围",
        options=["近1年", "近3年", "近5年", "自定义"],
        index=1
    )
    
    # 自定义时间范围
    if time_range == "自定义":
        start_date = st.sidebar.date_input("开始日期", datetime.now() - timedelta(days=365*3))
        end_date = st.sidebar.date_input("结束日期", datetime.now())
    else:
        # 自动计算日期范围
        end_date = datetime.now()
        if time_range == "近1年":
            start_date = end_date - timedelta(days=365)
        elif time_range == "近3年":
            start_date = end_date - timedelta(days=365*3)
        else:  # 近5年
            start_date = end_date - timedelta(days=365*5)
    
    # 技术指标选择
    st.sidebar.header("📈 技术指标选择")
    indicator = st.sidebar.selectbox(
        "选择技术指标",
        options=["MACD", "RSI", "KDJ", "布林带", "均线", "乖离率"],
        index=0
    )

    # 技术指标交易说明
    indicator_descriptions = {
        "MACD": """
        **MACD指标交易规则：**
        - **买入信号**：MACD线上穿信号线（金叉）
        - **卖出信号**：MACD线下穿信号线（死叉）
        - **参数说明**：快线周期、慢线周期、信号线周期
        - **适用场景**：趋势明显的市场，避免震荡市
        """,

        "RSI": """
        **RSI指标交易规则：**
        - **买入信号**：RSI < 30（超卖区域）
        - **卖出信号**：RSI > 70（超买区域）
        - **参数说明**：计算周期，默认14天
        - **适用场景**：震荡市场，识别超买超卖状态
        """,

        "KDJ": """
        **KDJ指标交易规则：**
        - **买入信号**：K线上穿D线且J值 < 20
        - **卖出信号**：K线下穿D线且J值 > 80
        - **参数说明**：K值周期、D值平滑周期、J值计算周期
        - **适用场景**：短期交易，捕捉价格转折点
        """,

        "布林带": """
        **布林带指标交易规则：**
        - **买入信号**：价格跌破下轨
        - **卖出信号**：价格突破上轨
        - **参数说明**：计算周期、上轨偏差、下轨偏差
        - **适用场景**：波动率分析，识别价格相对位置
        """,

        "均线": """
        **均线指标交易规则：**
        - **买入信号**：价格上穿均线
        - **卖出信号**：价格下穿均线
        - **参数说明**：移动平均计算周期
        - **适用场景**：趋势跟踪，过滤短期噪音
        """,

        "乖离率": """
        **乖离率指标交易规则：**
        - **买入信号**：乖离率 < -6%（价格过度偏离均线向下）
        - **卖出信号**：乖离率 > 6%（价格过度偏离均线向上）
        - **参数说明**：移动平均周期
        - **适用场景**：识别价格回归机会，逆势操作
        """
    }

    # 显示技术指标说明
    st.sidebar.markdown("---")
    st.sidebar.markdown("📋 **指标使用说明**")
    st.sidebar.info(indicator_descriptions.get(indicator, "选择一个指标查看使用说明"))
    
    # 优化算法选择
    st.sidebar.header("⚙️ 优化设置")
    algorithm = st.sidebar.selectbox(
        "优化算法",
        options=["网格搜索", "遗传算法"],
        index=0
    )
    
    # 参数范围设置（根据选择的指标动态显示）
    st.sidebar.header("🔧 参数范围设置")
    
    if indicator == "MACD":
        fast_period_min = st.sidebar.slider("Fast Period 最小值", 5, 20, 8)
        fast_period_max = st.sidebar.slider("Fast Period 最大值", 10, 40, 15)
        slow_period_min = st.sidebar.slider("Slow Period 最小值", 15, 40, 20)
        slow_period_max = st.sidebar.slider("Slow Period 最大值", 20, 60, 30)
        signal_period_min = st.sidebar.slider("Signal Period 最小值", 3, 10, 5)
        signal_period_max = st.sidebar.slider("Signal Period 最大值", 8, 20, 12)
        
    elif indicator == "RSI":
        rsi_period_min = st.sidebar.slider("RSI Period 最小值", 5, 10, 6)
        rsi_period_max = st.sidebar.slider("RSI Period 最大值", 12, 30, 20)
        
    # 添加其他指标的参数设置...
    
    # 开始优化按钮
    if st.sidebar.button("🚀 开始参数优化", type="primary"):
        with st.spinner("正在获取数据并执行优化..."):
            try:
                # 获取股票数据
                fetcher = StockDataFetcher()
                df = fetcher.get_stock_data(stock_code, start_date, end_date, period)
                
                if df is not None and len(df) > 0:
                    st.success(f"成功获取 {stock_code} 的{len(df)}条数据")
                    
                    # 显示数据预览
                    st.subheader("📊 数据预览")
                    st.dataframe(df.tail(10))
                    
                    # 执行参数优化
                    optimizer = ParameterOptimizer()
                    best_params, results_df = optimizer.optimize_parameters(
                        df, indicator, algorithm
                    )
                    
                    # 显示优化结果
                    st.subheader("🎯 最优参数组合")
                    st.write(best_params)
                    
                    # 显示所有参数组合结果
                    st.subheader("📋 所有参数组合绩效")
                    st.dataframe(results_df.sort_values("年化收益率", ascending=False))
                    
                else:
                    st.error("获取数据失败，请检查股票代码是否正确")
                    
            except Exception as e:
                st.error(f"执行过程中发生错误: {str(e)}")
    
    # 添加使用说明
    st.sidebar.header("ℹ️ 使用说明")
    st.sidebar.info("""
    1. 输入股票代码（A股格式）
    2. 选择时间周期和回测范围
    3. 选择技术指标和优化算法
    4. 设置参数搜索范围
    5. 点击开始优化按钮
    """)

if __name__ == "__main__":
    main()