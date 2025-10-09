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

# 导入项目工具类并初始化
from utils.project_utils import ProjectUtils
ProjectUtils.setup_project_path()
ProjectUtils.setup_logging()

# 导入自定义模块
from data.data_fetcher import StockDataFetcher
from indicators.technical_indicators import TechnicalIndicators
from backtest.backtest_engine import BacktestEngine
from optimization.optimizer import ParameterOptimizer

# 加载配置
config = ProjectUtils.load_config()

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
        options=["MACD", "RSI", "KDJ", "布林带", "均线", "乖离率", 
                "ATR", "OBV", "威廉指标", "CCI", "ADX", "动量指标", "抛物线SAR"],
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
        """,

        "ATR": """
        **ATR指标交易规则：**
        - **买入信号**：无直接买入信号（主要辅助其他指标）
        - **卖出信号**：无直接卖出信号（主要辅助其他指标）
        - **参数说明**：平均真实波幅计算周期，默认14天
        - **适用场景**：波动率分析，为仓位管理和止损提供参考
        """,

        "OBV": """
        **OBV指标交易规则：**
        - **买入信号**：OBV连续上升且突破近期高点（资金持续流入）
        - **卖出信号**：OBV连续下降且跌破近期低点（资金持续流出）
        - **参数说明**：无参数，基于成交量和价格累积计算
        - **适用场景**：成交量确认，识别资金流向和趋势持续性
        """,

        "威廉指标": """
        **威廉指标交易规则：**
        - **买入信号**：威廉指标 < -80（超卖区域，反弹机会）
        - **卖出信号**：威廉指标 > -20（超买区域，回调风险）
        - **参数说明**：计算周期，默认14天
        - **适用场景**：超买超卖判断，识别短期反转机会
        """,

        "CCI": """
        **CCI指标交易规则：**
        - **买入信号**：CCI < -100（超卖区域，价格偏离均值过大）
        - **卖出信号**：CCI > 100（超买区域，价格偏离均值过大）
        - **参数说明**：商品通道指数计算周期，默认20天
        - **适用场景**：周期性商品分析，识别价格偏离常态的程度
        """,

        "ADX": """
        **ADX指标交易规则：**
        - **买入信号**：无直接买入信号（仅判断趋势强度）
        - **卖出信号**：无直接卖出信号（仅判断趋势强度）
        - **参数说明**：平均方向指数计算周期，默认14天
        - **适用场景**：趋势强度判断，配合其他指标确认交易时机
        """,

        "动量指标": """
        **动量指标交易规则：**
        - **买入信号**：动量指标上穿零轴（价格开始加速上涨）
        - **卖出信号**：动量指标下穿零轴（价格开始加速下跌）
        - **参数说明**：动量计算周期，默认10天
        - **适用场景**：趋势动量分析，识别价格加速变化的时机
        """,

        "抛物线SAR": """
        **抛物线SAR指标交易规则：**
        - **买入信号**：价格上穿SAR线（趋势转强，买入确认）
        - **卖出信号**：价格下穿SAR线（趋势转弱，卖出确认）
        - **参数说明**：加速因子0.02，最大值0.2
        - **适用场景**：趋势跟踪止损，动态调整止损止盈点位
        """
    }

    # 显示技术指标说明
    with st.sidebar.expander("📋 指标使用说明", expanded=False):
        st.markdown(indicator_descriptions.get(indicator, "选择一个指标查看使用说明"))
    
    # 风险控制设置
    st.sidebar.markdown("---")
    st.sidebar.header("🛡️ 风险控制设置")
    enable_risk_management = st.sidebar.checkbox(
        "启用风险控制",
        value=True,
        help="开启后将应用智能仓位管理、止损止盈等风险控制策略"
    )
    
    # 风险控制说明提示框
    if enable_risk_management:
        with st.sidebar.expander("📋 风险控制说明", expanded=False):
            st.markdown("""
            **🛡️ 风险控制功能说明：**
            
            **1. 智能仓位管理**
            - 基于风险暴露计算最优仓位大小
            - 高风险交易自动减半仓位保护资本
            - 实时评估交易风险等级
            
            **2. 自动止损止盈**
            - 止损：价格跌破买入价5%时自动卖出
            - 止盈：价格达到预期收益目标时自动卖出
            - 风险回报比：基于风险调整卖出决策
            
            **3. 风险等级评估**
            - 低风险：正常仓位，标准交易策略
            - 中风险：适当调整仓位，谨慎交易
            - 高风险：减少仓位，降低风险暴露
            - 极高风险：大幅减仓，保护本金安全
            
            **4. 交易记录增强**
            - 详细记录风险等级和决策原因
            - 监控风险违规情况
            - 提供完整的风险控制日志
            
            **⚠️ 注意事项：**
            - 风险控制可能减少交易频率但提高安全性
            - 建议新手投资者开启风险控制
            - 可根据个人风险偏好调整设置
            """)
    
    # 优化算法选择
    st.sidebar.header("⚙️ 优化设置")
    algorithm = st.sidebar.selectbox(
        "优化算法",
        options=["网格搜索", "遗传算法"],
        index=0
    )
    
    # 参数范围设置（根据选择的指标动态显示）
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 参数范围设置")
    
    from config import INDICATOR_PARAMS
    
    param_ranges = {}
    indicator_params = INDICATOR_PARAMS.get(indicator, {})
    
    for param_name, param_config in indicator_params.items():
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_value = st.slider(
                f"{param_name} 最小值",
                min_value=param_config["min"],
                max_value=param_config["max"],
                value=param_config["default"],
                key=f"{param_name}_min"
            )
        with col2:
            max_value = st.slider(
                f"{param_name} 最大值",
                min_value=param_config["min"],
                max_value=param_config["max"],
                value=param_config["max"],
                key=f"{param_name}_max"
            )
            
        # 验证参数范围的合理性
        if min_value >= max_value:
            st.sidebar.error(f"{param_name} 的最小值不能大于或等于最大值")
            param_ranges = {}
            break
            
        param_ranges[param_name] = (min_value, max_value)
        
    # MACD特殊处理：验证快慢线周期
    if indicator == "MACD" and param_ranges:
        if param_ranges["fast_period"][1] >= param_ranges["slow_period"][0]:
            st.sidebar.error("快线周期不能大于或等于慢线周期")
            param_ranges = {}
        
    # 添加其他指标的参数设置...
    
    # 开始优化按钮
    if st.sidebar.button("🚀 开始参数优化", type="primary"):
        if not param_ranges:
            st.error("请正确设置参数范围")
            return
            
        # 创建进度条容器
        progress_container = st.empty()
        results_container = st.container()
        
        with st.spinner("正在获取数据..."):
            try:
                # 获取股票数据
                fetcher = StockDataFetcher()
                df = fetcher.get_stock_data(stock_code, start_date, end_date, period)
                
                if df is not None and len(df) > 0:
                    progress_container.success(f"成功获取 {stock_code} 的{len(df)}条数据")
                    
                    with results_container:
                        # 显示数据预览
                        st.subheader("📊 数据预览")
                        fig = go.Figure(data=[
                            go.Candlestick(
                                x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close']
                            )
                        ])
                        fig.update_layout(title=f"{stock_code} 行情走势")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.spinner("正在执行参数优化..."):
                            # 执行参数优化
                            optimizer = ParameterOptimizer()
                            
                            # 创建优化进度容器
                            progress_container = st.empty()
                            progress_text = st.empty()
                            
                            def progress_callback(current, total):
                                progress = int(current / total * 100)
                                progress_container.progress(progress / 100)
                                progress_text.text(f"参数优化进度: {progress}% ({current}/{total} 组合)")
                            
                            best_params, results_df = optimizer.optimize_parameters(
                                df, indicator, algorithm,
                                param_ranges=param_ranges,
                                progress_callback=progress_callback
                            )
                            
                            # 清除进度显示
                            progress_container.empty()
                            progress_text.empty()
                            
                            if best_params and not results_df.empty:
                                # 清除进度显示（使用已创建的容器）
                                progress_container.empty()
                                progress_text.empty()
                                
                                # 显示风险控制状态
                                if enable_risk_management:
                                    st.success("🛡️ 风险控制已启用")
                                else:
                                    st.info("📊 风险控制已禁用")
                                
                                # 显示优化结果
                                st.subheader("🎯 最优参数组合")
                                st.json(best_params)
                                
                                # 创建回测引擎（传递风险控制设置）
                                engine = BacktestEngine(enable_risk_management=enable_risk_management)
                                
                                # 使用最优参数运行完整回测
                                full_results = engine.run_backtest(df, 'signal', 'close', return_full=True)
                                
                                # 显示回测曲线
                                st.subheader("📈 最优参数回测曲线")
                                if full_results and 'equity_curve' in full_results:
                                    equity_curve_df = pd.DataFrame(full_results['equity_curve'])
                                    equity_curve_df.set_index('date', inplace=True)
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=equity_curve_df.index,
                                        y=equity_curve_df['equity'],
                                        name="策略收益"
                                    ))
                                    if 'benchmark' in equity_curve_df.columns:
                                        fig.add_trace(go.Scatter(
                                            x=equity_curve_df.index,
                                            y=equity_curve_df['benchmark'],
                                            name="基准收益"
                                        ))
                                    fig.update_layout(title="策略收益vs基准收益")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # 显示所有参数组合结果
                                st.subheader("📋 参数优化结果")
                                st.dataframe(
                                    results_df.sort_values("年化收益率", ascending=False)
                                    .style.format({
                                        "年化收益率": "{:.2%}",
                                        "最大回撤": "{:.2%}",
                                        "夏普比率": "{:.2f}"
                                    })
                                )
                            else:
                                st.error("优化过程未能找到有效的参数组合")
                else:
                    st.error("获取数据失败，请检查股票代码是否正确")
                    
            except Exception as e:
                import traceback
                st.error(f"执行过程中发生错误: {str(e)}")
                st.code(traceback.format_exc(), language="python")
    
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