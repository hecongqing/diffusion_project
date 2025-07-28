#!/usr/bin/env python3
"""
训练实用工具函数
==============

这个模块提供训练过程中的各种实用函数：
1. 日志配置和管理
2. 模型检查点保存和加载
3. 训练状态管理
4. 性能监控工具

作者: Diffusion教程团队
日期: 2024年
"""

import os
import sys
import torch
import torch.nn as nn
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Union
import numpy as np


def setup_logging(log_dir: str, log_level: str = 'INFO') -> None:
    """
    设置日志配置
    
    参数:
        log_dir: 日志保存目录
        log_level: 日志级别
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 获取日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 配置日志
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 控制台输出
            logging.StreamHandler(sys.stdout),
            # 文件输出
            logging.FileHandler(
                os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                encoding='utf-8'
            )
        ]
    )
    
    # 设置第三方库日志级别
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logging.info(f"日志系统已初始化，日志保存至: {log_dir}")


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    extra_info: Dict[str, Any] = None
) -> None:
    """
    保存训练检查点
    
    参数:
        filepath: 保存路径
        model: 模型对象
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        train_loss: 训练损失
        val_loss: 验证损失
        extra_info: 额外信息字典
    """
    # 准备检查点数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    # 添加学习率调度器状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 添加额外信息
    if extra_info:
        checkpoint.update(extra_info)
    
    # 保存检查点
    try:
        torch.save(checkpoint, filepath)
        logging.info(f"✅ 检查点已保存: {filepath}")
    except Exception as e:
        logging.error(f"❌ 保存检查点失败: {e}")
        raise


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载训练检查点
    
    参数:
        filepath: 检查点文件路径
        model: 模型对象
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 目标设备（可选）
        
    返回:
        checkpoint: 检查点信息字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")
    
    try:
        # 加载检查点
        if device is not None:
            checkpoint = torch.load(filepath, map_location=device)
        else:
            checkpoint = torch.load(filepath)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"✅ 模型状态已加载")
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f"✅ 优化器状态已加载")
        
        # 加载学习率调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f"✅ 调度器状态已加载")
        
        logging.info(f"✅ 检查点加载完成: {filepath}")
        logging.info(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
        logging.info(f"   训练损失: {checkpoint.get('train_loss', 'Unknown')}")
        logging.info(f"   验证损失: {checkpoint.get('val_loss', 'Unknown')}")
        
        return checkpoint
        
    except Exception as e:
        logging.error(f"❌ 加载检查点失败: {e}")
        raise


def save_model_config(model: nn.Module, save_path: str) -> None:
    """
    保存模型配置信息
    
    参数:
        model: 模型对象
        save_path: 保存路径
    """
    config = {
        'model_class': model.__class__.__name__,
        'model_config': getattr(model, 'config', {}),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1e6,
        'timestamp': datetime.now().isoformat(),
    }
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ 模型配置已保存: {save_path}")
    except Exception as e:
        logging.error(f"❌ 保存模型配置失败: {e}")


class TrainingMonitor:
    """
    训练监控器
    
    用于跟踪和记录训练过程中的各种指标。
    """
    
    def __init__(self, save_dir: str):
        """
        初始化训练监控器
        
        参数:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': [],
            'timestamps': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def update(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        lr: float
    ) -> bool:
        """
        更新训练指标
        
        参数:
            epoch: 当前epoch
            train_loss: 训练损失
            val_loss: 验证损失
            lr: 当前学习率
            
        返回:
            is_best: 是否是最佳模型
        """
        # 记录指标
        self.metrics['epochs'].append(epoch)
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)
        self.metrics['learning_rates'].append(lr)
        self.metrics['timestamps'].append(datetime.now().isoformat())
        
        # 检查是否是最佳模型
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            logging.info(f"🎯 发现更好的模型! Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
        
        return is_best
    
    def save_metrics(self) -> None:
        """
        保存训练指标到文件
        """
        metrics_path = os.path.join(self.save_dir, 'training_metrics.json')
        
        # 添加统计信息
        summary = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
            'final_val_loss': self.metrics['val_losses'][-1] if self.metrics['val_losses'] else None,
            'total_epochs': len(self.metrics['epochs']),
        }
        
        data = {
            'metrics': self.metrics,
            'summary': summary
        }
        
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"✅ 训练指标已保存: {metrics_path}")
        except Exception as e:
            logging.error(f"❌ 保存训练指标失败: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取训练总结
        
        返回:
            summary: 训练总结字典
        """
        if not self.metrics['epochs']:
            return {'message': '暂无训练数据'}
        
        return {
            '总训练轮数': len(self.metrics['epochs']),
            '最佳验证损失': f"{self.best_val_loss:.4f}",
            '最佳模型epoch': self.best_epoch + 1,
            '最终训练损失': f"{self.metrics['train_losses'][-1]:.4f}",
            '最终验证损失': f"{self.metrics['val_losses'][-1]:.4f}",
            '最终学习率': f"{self.metrics['learning_rates'][-1]:.6f}",
        }


class EarlyStopping:
    """
    早停机制
    
    在验证损失不再改善时提前停止训练。
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        初始化早停机制
        
        参数:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            mode: 监控模式 ('min' 或 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        参数:
            score: 当前监控指标值
            
        返回:
            early_stop: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f"🛑 早停触发! 已连续 {self.patience} 轮无改善")
        
        return self.early_stop


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数数量
    
    参数:
        model: 模型对象
        
    返回:
        param_count: 参数统计字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / 1e6  # 假设float32
    }


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息
    
    返回:
        device_info: 设备信息字典
    """
    info = {
        'platform': sys.platform,
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'current_gpu': torch.cuda.current_device(),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
        })
    else:
        info['cuda_available'] = False
    
    return info


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    参数:
        seconds: 秒数
        
    返回:
        formatted_time: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    创建实验目录
    
    参数:
        base_dir: 基础目录
        experiment_name: 实验名称
        
    返回:
        experiment_dir: 实验目录路径
    """
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # 创建目录结构
    subdirs = ['checkpoints', 'logs', 'samples', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    logging.info(f"📁 实验目录已创建: {experiment_dir}")
    return experiment_dir


# 测试函数
def test_training_utils():
    """
    测试训练工具函数
    """
    print("🧪 测试训练工具函数...")
    
    # 测试设备信息
    device_info = get_device_info()
    print(f"✅ 设备信息: {device_info}")
    
    # 测试时间格式化
    print(f"✅ 时间格式化: {format_time(3661)}")
    
    # 测试训练监控器
    monitor = TrainingMonitor('./test_monitor')
    monitor.update(0, 1.5, 1.2, 0.001)
    monitor.update(1, 1.3, 1.1, 0.0009)
    summary = monitor.get_summary()
    print(f"✅ 训练监控器: {summary}")
    
    print("✅ 所有测试通过!")


if __name__ == "__main__":
    test_training_utils()