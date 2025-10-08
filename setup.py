#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
项目安装设置脚本
自动安装依赖并配置环境
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    if sys.version_info < (3, 11):
        print(f"❌ 需要Python 3.11或更高版本，当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version}")
    return True

def install_requirements():
    """安装项目依赖"""
    print("📦 安装项目依赖...")
    
    # 检查uv是否可用
    try:
        result = subprocess.run([sys.executable, "-m", "uv", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 使用uv安装依赖...")
            subprocess.run([sys.executable, "-m", "uv", "sync"], check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  uv不可用，使用pip安装依赖...")
    
    # 使用pip安装
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def check_ta_lib():
    """检查TA-Lib安装"""
    print("📊 检查TA-Lib...")
    try:
        import talib
        print("✅ TA-Lib已正确安装")
        return True
    except ImportError as e:
        print(f"❌ TA-Lib安装有问题: {e}")
        print("💡 提示: 在Windows上可能需要从 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib 下载预编译版本")
        return False

def create_directories():
    """创建必要的目录"""
    print("📁 创建项目目录结构...")
    directories = [
        "data",
        "indicators", 
        "backtest",
        "optimization",
        "results",
        "logs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_name}")
    
    return True

def setup_environment():
    """设置环境变量"""
    print("🌐 设置环境变量...")
    
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 设置环境变量（如果需要）
    os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')
    
    print(f"✅ 项目根目录: {project_root}")
    return True

def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    
    modules_to_test = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("akshare", "AkShare"),
        ("talib", "TA-Lib"),
        ("plotly", "Plotly")
    ]
    
    all_imports_ok = True
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name} 导入成功")
        except ImportError as e:
            print(f"❌ {display_name} 导入失败: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def main():
    """主安装函数"""
    print("=" * 50)
    print("🚀 股票指标参数优化系统 - 安装程序")
    print("=" * 50)
    
    # 执行安装步骤
    steps = [
        ("检查Python版本", check_python_version),
        ("创建目录结构", create_directories),
        ("安装依赖", install_requirements),
        ("检查TA-Lib", check_ta_lib),
        ("设置环境", setup_environment),
        ("测试导入", test_imports)
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n🔧 {step_name}...")
        try:
            result = step_func()
            results.append(result)
            if not result:
                print(f"❌ {step_name} 失败")
                break
        except Exception as e:
            print(f"❌ {step_name} 异常: {e}")
            results.append(False)
            break
    
    print("\n" + "=" * 50)
    print("📊 安装结果汇总:")
    
    if all(results):
        print("🎉 安装成功完成！")
        print("\n🚀 接下来可以:")
        print("1. 运行测试: python test_basic.py")
        print("2. 启动应用: python run.py")
        print("3. 或直接运行: streamlit run app.py")
        return True
    else:
        print("⚠️  安装过程中出现问题")
        print(f"总步骤: {len(results)}, 成功: {sum(results)}, 失败: {len(results) - sum(results)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)