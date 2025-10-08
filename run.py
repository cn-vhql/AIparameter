#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票指标参数优化系统 - 启动脚本
"""

import streamlit as st
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("🚀 启动股票指标参数优化系统...")
    print("📊 请确保已安装所有依赖: pip install -r requirements.txt")
    print("🌐 启动Streamlit应用...")
    
    # 启动Streamlit应用
    os.system("streamlit run app.py")

if __name__ == "__main__":
    main()