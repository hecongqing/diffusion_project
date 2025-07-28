#!/usr/bin/env python3
"""
Diffusion模型实战教程 - 演示脚本
==============================

这个脚本演示了本项目的核心功能：
1. 快速体验扩散模型生成
2. 训练流程演示
3. 可视化功能展示

运行方式:
    python run_demo.py

作者: Diffusion教程团队
日期: 2024年
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def main():
    """
    主演示函数
    """
    print("🎉 欢迎使用Diffusion模型实战教程！")
    print("=" * 60)
    
    # 检查环境
    print("🔍 检查运行环境...")
    check_environment()
    
    # 显示项目结构
    print("\n📁 项目结构:")
    show_project_structure()
    
    # 运行快速演示
    print("\n🚀 运行快速演示...")
    try:
        from examples.quick_start import main as quick_start_main
        quick_start_main()
    except ImportError as e:
        print(f"❌ 导入快速演示失败: {e}")
        print("💡 请确保已安装所有依赖: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ 运行演示时出错: {e}")
    
    # 显示教程信息
    print("\n📚 教程信息:")
    show_tutorial_info()
    
    print("\n✨ 演示完成！祝您学习愉快！")

def check_environment():
    """
    检查运行环境
    """
    print(f"   Python版本: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   CUDA可用: ✅ {torch.cuda.get_device_name()}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"   CUDA可用: ❌ (将使用CPU)")
    except ImportError:
        print("   PyTorch: ❌ 未安装")
    
    try:
        import matplotlib
        print(f"   Matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("   Matplotlib: ❌ 未安装")
    
    try:
        import numpy
        print(f"   NumPy版本: {numpy.__version__}")
    except ImportError:
        print("   NumPy: ❌ 未安装")

def show_project_structure():
    """
    显示项目结构
    """
    structure = """
    diffusion-tutorial/
    ├── 📖 README.md                    # 项目介绍
    ├── 📋 requirements.txt             # 依赖列表
    ├── 🚀 run_demo.py                  # 演示脚本
    ├── 📁 examples/                    # 示例代码
    │   └── 🎯 quick_start.py           # 快速入门示例
    ├── 📁 models/                      # 模型实现
    │   ├── 📁 ddpm/                    # DDPM模型
    │   ├── 📁 unet/                    # UNet网络
    │   └── 📁 schedulers/              # 噪声调度器
    ├── 📁 training/                    # 训练脚本
    │   └── 🏋️ train_ddpm.py            # DDPM训练
    ├── 📁 utils/                       # 工具函数
    │   ├── 🔧 training_utils.py        # 训练工具
    │   └── 📊 visualization.py         # 可视化工具
    ├── 📁 tutorials/                   # 教程笔记本
    │   └── 📓 01_diffusion_basics.ipynb # 基础教程
    └── 📁 theory/                      # 理论文档
        └── 📝 01_basic_concepts.md     # 基础概念
    """
    print(structure)

def show_tutorial_info():
    """
    显示教程信息
    """
    print("📚 可用教程:")
    print("   1. 📓 Jupyter教程: tutorials/01_diffusion_basics.ipynb")
    print("   2. 📝 理论文档: theory/01_basic_concepts.md")
    print("   3. 🎯 快速入门: examples/quick_start.py")
    print("   4. 🏋️ 完整训练: training/train_ddpm.py")
    
    print("\n🚀 快速开始:")
    print("   # 1. 安装依赖")
    print("   pip install -r requirements.txt")
    print()
    print("   # 2. 运行快速示例")
    print("   python examples/quick_start.py")
    print()
    print("   # 3. 开始Jupyter教程")
    print("   jupyter notebook tutorials/01_diffusion_basics.ipynb")
    print()
    print("   # 4. 训练自己的模型")
    print("   python training/train_ddpm.py --dataset mnist --epochs 10")
    
    print("\n📖 学习路径:")
    print("   第1步: 阅读基础概念 (theory/01_basic_concepts.md)")
    print("   第2步: 运行快速示例 (examples/quick_start.py)")
    print("   第3步: 完成Jupyter教程 (tutorials/01_diffusion_basics.ipynb)")
    print("   第4步: 尝试完整训练 (training/train_ddpm.py)")
    print("   第5步: 探索高级功能...")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='Diffusion模型实战教程演示脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_demo.py                    # 运行完整演示
  python run_demo.py --quick           # 只显示项目信息
  python run_demo.py --check-env       # 只检查环境
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='快速模式，只显示项目信息'
    )
    
    parser.add_argument(
        '--check-env',
        action='store_true', 
        help='只检查运行环境'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.check_env:
        print("🔍 检查运行环境...")
        check_environment()
    elif args.quick:
        print("📁 项目结构:")
        show_project_structure()
        print("📚 教程信息:")
        show_tutorial_info()
    else:
        main()