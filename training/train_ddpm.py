#!/usr/bin/env python3
"""
DDPM模型训练脚本
==============

这个脚本提供了完整的DDPM训练流程，包括：
1. 数据加载和预处理
2. 模型初始化和配置
3. 训练循环和验证
4. 模型保存和加载
5. 训练进度可视化

作者: Diffusion教程团队
日期: 2024年
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import json
import wandb
from typing import Dict, List, Optional, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ddpm.ddpm_model import DDPM, NoiseScheduler
from models.unet.unet_architecture import UNet
from utils.training_utils import setup_logging, save_checkpoint, load_checkpoint
from utils.visualization import plot_training_curves, create_sample_grid

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='DDPM模型训练')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'celeba', 'mnist', 'custom'],
                       help='训练数据集')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='数据集路径')
    parser.add_argument('--image_size', type=int, default=32,
                       help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    
    # 模型相关参数  
    parser.add_argument('--model_channels', type=int, nargs='+', 
                       default=[128, 256, 512, 512],
                       help='UNet各层通道数')
    parser.add_argument('--attention_levels', type=int, nargs='+',
                       default=[0, 0, 1, 1],
                       help='各层是否使用注意力 (0/1)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='每个块的残差层数')
    parser.add_argument('--time_emb_dim', type=int, default=512,
                       help='时间嵌入维度')
    
    # 扩散过程参数
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='扩散步数')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                       help='噪声调度起始值')
    parser.add_argument('--beta_end', type=float, default=0.02,
                       help='噪声调度结束值')
    parser.add_argument('--schedule', type=str, default='linear',
                       choices=['linear', 'cosine'],
                       help='噪声调度策略')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪阈值')
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='模型保存间隔(轮数)')
    parser.add_argument('--sample_interval', type=int, default=5,
                       help='生成样本间隔(轮数)')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志记录间隔(步数)')
    
    # 设备和其他参数
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cuda/cpu)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='是否使用混合精度训练')
    parser.add_argument('--use_wandb', action='store_true',
                       help='是否使用Weights & Biases记录')
    parser.add_argument('--wandb_project', type=str, default='ddpm-training',
                       help='WandB项目名称')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """
    设置计算设备
    
    参数:
        device_arg: 设备参数
        
    返回:
        device: PyTorch设备对象
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 使用GPU: {torch.cuda.get_device_name()}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🚀 使用Apple Silicon GPU")
        else:
            device = torch.device('cpu')
            print("⚠️  使用CPU训练（速度较慢）")
    else:
        device = torch.device(device_arg)
        print(f"🚀 使用指定设备: {device}")
    
    return device


def create_datasets(args) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据集
    
    参数:
        args: 命令行参数
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 数据预处理
    if args.dataset == 'cifar10':
        # CIFAR-10数据集预处理
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(p=0.5),  # 数据增强
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
        ])
        
        # 加载数据集
        train_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform
        )
        
        val_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        
    elif args.dataset == 'mnist':
        # MNIST数据集预处理
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
        ])
        
        train_dataset = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform
        )
        
        val_dataset = datasets.MNIST(
            root=args.data_path,
            train=False,
            download=True,
            transform=transform
        )
        
    elif args.dataset == 'celeba':
        # CelebA数据集预处理
        transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 注意：CelebA需要手动下载
        train_dataset = datasets.CelebA(
            root=args.data_path,
            split='train',
            download=False,  # 手动下载
            transform=transform
        )
        
        val_dataset = datasets.CelebA(
            root=args.data_path,
            split='valid',
            download=False,
            transform=transforms.Compose([
                transforms.CenterCrop(140),
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"📊 数据集信息:")
    print(f"   训练集: {len(train_dataset)} 张图像")
    print(f"   验证集: {len(val_dataset)} 张图像")
    print(f"   批次大小: {args.batch_size}")
    print(f"   训练批次数: {len(train_loader)}")
    
    return train_loader, val_loader


def create_model(args, device: torch.device) -> DDPM:
    """
    创建DDPM模型
    
    参数:
        args: 命令行参数
        device: 计算设备
        
    返回:
        model: DDPM模型
    """
    # 确定输入通道数
    if args.dataset == 'mnist':
        in_channels = 1
    else:
        in_channels = 3
    
    # 转换注意力级别参数
    attention_levels = [bool(x) for x in args.attention_levels]
    
    # 创建UNet网络
    unet = UNet(
        in_channels=in_channels,
        out_channels=in_channels,
        channels=args.model_channels,
        num_layers=args.num_layers,
        attention_levels=attention_levels,
        time_emb_dim=args.time_emb_dim
    )
    
    # 创建噪声调度器
    noise_scheduler = NoiseScheduler(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule
    )
    
    # 创建DDPM模型
    model = DDPM(unet, noise_scheduler)
    model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  模型信息:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1e6:.1f} MB")
    
    return model


def train_epoch(
    model: DDPM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    训练一个epoch
    
    参数:
        model: DDPM模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 计算设备
        epoch: 当前epoch
        args: 命令行参数
        scaler: 混合精度缩放器
        
    返回:
        metrics: 训练指标字典
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    # 创建进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        if args.mixed_precision and scaler is not None:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                loss = model.compute_loss(images)
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通训练
            loss = model.compute_loss(images)
            loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss:.4f}'
        })
        
        # 记录日志
        if (batch_idx + 1) % args.log_interval == 0:
            step = epoch * num_batches + batch_idx + 1
            
            if args.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/step': step,
                    'train/epoch': epoch + 1
                })
    
    return {
        'loss': total_loss / num_batches,
        'num_batches': num_batches
    }


def validate_model(
    model: DDPM,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    args
) -> Dict[str, float]:
    """
    验证模型性能
    
    参数:
        model: DDPM模型
        val_loader: 验证数据加载器
        device: 计算设备
        epoch: 当前epoch
        args: 命令行参数
        
    返回:
        metrics: 验证指标字典
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = model.compute_loss(images)
            else:
                loss = model.compute_loss(images)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    print(f"📊 验证结果 - Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return {
        'loss': avg_loss,
        'num_batches': num_batches
    }


def generate_samples(
    model: DDPM,
    device: torch.device,
    num_samples: int = 16,
    image_size: int = 32,
    channels: int = 3
) -> torch.Tensor:
    """
    生成样本图像
    
    参数:
        model: DDPM模型
        device: 计算设备
        num_samples: 生成样本数量
        image_size: 图像尺寸
        channels: 图像通道数
        
    返回:
        samples: 生成的样本
    """
    model.eval()
    
    with torch.no_grad():
        # 从随机噪声开始
        samples = torch.randn(num_samples, channels, image_size, image_size, device=device)
        
        # 逐步去噪
        samples = model.sample(samples, show_progress=True)
        
        # 限制到[-1, 1]范围
        samples = torch.clamp(samples, -1, 1)
    
    return samples


def main():
    """
    主训练函数
    """
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'samples'), exist_ok=True)
    
    # 设置日志
    setup_logging(args.log_dir)
    logging.info(f"开始DDPM训练 - {datetime.now()}")
    logging.info(f"参数: {vars(args)}")
    
    # 初始化WandB
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"ddpm_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # 设置设备
    device = setup_device(args.device)
    
    # 创建数据集
    train_loader, val_loader = create_datasets(args)
    
    # 创建模型
    model = create_model(args, device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # 恢复训练（如果指定）
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ 从epoch {start_epoch}恢复训练")
    
    # 训练历史
    train_losses = []
    val_losses = []
    
    # 确定图像通道数
    channels = 1 if args.dataset == 'mnist' else 3
    
    print("🚀 开始训练...")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # 训练一个epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, args, scaler
        )
        train_losses.append(train_metrics['loss'])
        
        # 验证模型
        val_metrics = validate_model(model, val_loader, device, epoch, args)
        val_losses.append(val_metrics['loss'])
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录到WandB
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_metrics['loss'],
                'val/epoch_loss': val_metrics['loss'],
                'lr': current_lr
            })
        
        # 打印epoch总结
        print(f"\n📈 Epoch {epoch+1} 总结:")
        print(f"   训练损失: {train_metrics['loss']:.4f}")
        print(f"   验证损失: {val_metrics['loss']:.4f}")
        print(f"   学习率: {current_lr:.6f}")
        
        # 生成样本
        if (epoch + 1) % args.sample_interval == 0:
            print("🎨 生成样本图像...")
            samples = generate_samples(
                model, device, num_samples=16, 
                image_size=args.image_size, channels=channels
            )
            
            # 保存样本
            sample_path = os.path.join(
                args.save_dir, 'samples', f'epoch_{epoch+1:03d}.png'
            )
            save_image(
                samples, sample_path,
                nrow=4, normalize=True, value_range=(-1, 1)
            )
            
            if args.use_wandb:
                wandb.log({
                    f'samples/epoch_{epoch+1}': wandb.Image(sample_path)
                })
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f'checkpoint_epoch_{epoch+1:03d}.pth'
            )
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, 
                epoch, train_metrics['loss'], val_metrics['loss']
            )
            print(f"💾 已保存检查点: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(
        final_model_path, model, optimizer, scheduler,
        args.epochs - 1, train_losses[-1], val_losses[-1]
    )
    print(f"🎯 训练完成！最终模型保存至: {final_model_path}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, args.save_dir)
    
    # 关闭WandB
    if args.use_wandb:
        wandb.finish()
    
    logging.info(f"训练完成 - {datetime.now()}")


if __name__ == '__main__':
    main()