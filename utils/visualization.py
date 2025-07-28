#!/usr/bin/env python3
"""
可视化工具函数
============

这个模块提供各种可视化功能：
1. 训练曲线绘制
2. 生成样本网格展示
3. 扩散过程可视化
4. 注意力热图可视化

作者: Diffusion教程团队
日期: 2024年
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image
from typing import List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "训练曲线",
    show_best: bool = True
) -> None:
    """
    绘制训练和验证损失曲线
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径（目录）
        title: 图表标题
        show_best: 是否标记最佳点
    """
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制训练和验证曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    
    if show_best and val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        plt.scatter(best_epoch, best_loss, color='red', s=100, zorder=5)
        plt.annotate(f'最佳: Epoch {best_epoch}\nLoss: {best_loss:.4f}',
                    xy=(best_epoch, best_loss),
                    xytext=(best_epoch + len(epochs) * 0.1, best_loss + max(val_losses) * 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'{title} - 损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制对数损失曲线
    plt.subplot(2, 2, 2)
    plt.semilogy(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    plt.semilogy(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('损失 (对数尺度)')
    plt.title('对数损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制损失差异
    if len(train_losses) == len(val_losses):
        plt.subplot(2, 2, 3)
        gap = np.array(val_losses) - np.array(train_losses)
        plt.plot(epochs, gap, 'g-', label='验证-训练损失差', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('损失差异')
        plt.title('过拟合监控')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 绘制损失分布
    plt.subplot(2, 2, 4)
    plt.hist(train_losses, bins=20, alpha=0.7, label='训练损失', color='blue')
    plt.hist(val_losses, bins=20, alpha=0.7, label='验证损失', color='red')
    plt.xlabel('损失值')
    plt.ylabel('频次')
    plt.title('损失分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_file = os.path.join(save_path, 'training_curves.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 训练曲线已保存: {save_file}")


def create_sample_grid(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 4,
    title: str = "生成样本",
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1, 1),
    add_labels: bool = False
) -> None:
    """
    创建样本网格图
    
    参数:
        samples: 样本张量 [batch_size, channels, height, width]
        save_path: 保存路径
        nrow: 每行样本数
        title: 图表标题
        normalize: 是否归一化
        value_range: 数值范围
        add_labels: 是否添加标签
    """
    # 创建网格
    grid = make_grid(
        samples, 
        nrow=nrow, 
        normalize=normalize, 
        value_range=value_range,
        pad_value=1
    )
    
    # 转换为numpy数组
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)
        cmap = 'gray'
    else:
        cmap = None
    
    # 绘制图像
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np, cmap=cmap)
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    
    # 添加样本编号
    if add_labels:
        num_samples = samples.shape[0]
        sample_height = samples.shape[2]
        sample_width = samples.shape[3]
        
        for i in range(num_samples):
            row = i // nrow
            col = i % nrow
            x = col * (sample_width + 2) + sample_width // 2
            y = row * (sample_height + 2) + sample_height // 2
            ax.text(x, y, str(i+1), ha='center', va='center',
                   fontsize=12, color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️  样本网格已保存: {save_path}")


def visualize_diffusion_process(
    original_image: torch.Tensor,
    noisy_images: List[torch.Tensor],
    timesteps: List[int],
    save_path: str,
    title: str = "扩散过程可视化"
) -> None:
    """
    可视化扩散过程
    
    参数:
        original_image: 原始图像 [channels, height, width]
        noisy_images: 不同时间步的噪声图像列表
        timesteps: 对应的时间步列表
        save_path: 保存路径
        title: 图表标题
    """
    num_steps = len(noisy_images)
    fig, axes = plt.subplots(2, (num_steps + 1) // 2 + 1, figsize=(20, 8))
    axes = axes.flatten()
    
    # 显示原始图像
    img = original_image.permute(1, 2, 0).cpu().numpy()
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
        cmap = 'gray'
    else:
        img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
        cmap = None
    
    axes[0].imshow(img, cmap=cmap)
    axes[0].set_title('原始图像\n(t=0)', fontsize=12)
    axes[0].axis('off')
    
    # 显示噪声图像
    for i, (noisy_img, t) in enumerate(zip(noisy_images, timesteps)):
        img = noisy_img.permute(1, 2, 0).cpu().numpy()
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        else:
            img = (img + 1) / 2
        
        axes[i + 1].imshow(img, cmap=cmap)
        axes[i + 1].set_title(f'时间步 {t}', fontsize=12)
        axes[i + 1].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_steps + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🔄 扩散过程可视化已保存: {save_path}")


def visualize_denoising_process(
    initial_noise: torch.Tensor,
    denoised_images: List[torch.Tensor],
    save_path: str,
    title: str = "去噪过程可视化",
    show_every: int = 50
) -> None:
    """
    可视化去噪过程
    
    参数:
        initial_noise: 初始噪声 [channels, height, width]
        denoised_images: 去噪过程中的图像列表
        save_path: 保存路径
        title: 图表标题
        show_every: 显示间隔
    """
    # 选择要显示的时间步
    total_steps = len(denoised_images)
    selected_indices = list(range(0, total_steps, show_every))
    if total_steps - 1 not in selected_indices:
        selected_indices.append(total_steps - 1)
    
    num_show = len(selected_indices)
    cols = min(6, num_show)
    rows = (num_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, step_idx in enumerate(selected_indices):
        row = idx // cols
        col = idx % cols
        
        if step_idx == 0:
            # 显示初始噪声
            img = initial_noise
            step_title = f"初始噪声\n(步骤 {total_steps})"
        else:
            img = denoised_images[step_idx]
            remaining_steps = total_steps - step_idx
            step_title = f"步骤 {remaining_steps}"
        
        # 处理图像
        img_np = img.permute(1, 2, 0).cpu().numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
            cmap = 'gray'
        else:
            img_np = torch.clamp((img + 1) / 2, 0, 1).permute(1, 2, 0).cpu().numpy()
            cmap = None
        
        axes[row, col].imshow(img_np, cmap=cmap)
        axes[row, col].set_title(step_title, fontsize=10)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_show, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 去噪过程可视化已保存: {save_path}")


def plot_noise_schedule(
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    save_path: str,
    title: str = "噪声调度可视化"
) -> None:
    """
    可视化噪声调度参数
    
    参数:
        betas: β值 [T]
        alphas: α值 [T]  
        alpha_bars: α̅值 [T]
        save_path: 保存路径
        title: 图表标题
    """
    timesteps = range(len(betas))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # β值曲线
    axes[0, 0].plot(timesteps, betas.cpu().numpy(), 'b-', linewidth=2)
    axes[0, 0].set_title('噪声调度 β(t)')
    axes[0, 0].set_xlabel('时间步 t')
    axes[0, 0].set_ylabel('β(t)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # α值曲线
    axes[0, 1].plot(timesteps, alphas.cpu().numpy(), 'r-', linewidth=2)
    axes[0, 1].set_title('α(t) = 1 - β(t)')
    axes[0, 1].set_xlabel('时间步 t')
    axes[0, 1].set_ylabel('α(t)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # α̅值曲线
    axes[1, 0].plot(timesteps, alpha_bars.cpu().numpy(), 'g-', linewidth=2)
    axes[1, 0].set_title('累积噪声 α̅(t)')
    axes[1, 0].set_xlabel('时间步 t')
    axes[1, 0].set_ylabel('α̅(t)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 信噪比
    snr = alpha_bars / (1 - alpha_bars)
    axes[1, 1].semilogy(timesteps, snr.cpu().numpy(), 'm-', linewidth=2)
    axes[1, 1].set_title('信噪比 SNR(t)')
    axes[1, 1].set_xlabel('时间步 t')
    axes[1, 1].set_ylabel('SNR(t) [对数尺度]')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 噪声调度可视化已保存: {save_path}")


def plot_attention_maps(
    attention_weights: torch.Tensor,
    input_image: torch.Tensor,
    save_path: str,
    num_heads: int = 8,
    title: str = "注意力热图"
) -> None:
    """
    可视化注意力权重热图
    
    参数:
        attention_weights: 注意力权重 [batch, heads, seq_len, seq_len]
        input_image: 输入图像 [batch, channels, height, width]
        save_path: 保存路径
        num_heads: 显示的注意力头数
        title: 图表标题
    """
    batch_size, num_heads_total, seq_len, _ = attention_weights.shape
    num_heads = min(num_heads, num_heads_total)
    
    # 取第一个样本
    attn = attention_weights[0, :num_heads].cpu().numpy()  # [num_heads, seq_len, seq_len]
    img = input_image[0].cpu().numpy()  # [channels, height, width]
    
    # 处理输入图像
    if img.shape[0] == 1:
        img = img.squeeze(0)
        img_cmap = 'gray'
    else:
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
        img_cmap = None
    
    # 计算网格尺寸
    height, width = img.shape[:2]
    grid_size = int(np.sqrt(seq_len))
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, num_heads + 1, height_ratios=[1, 1, 1])
    
    # 显示原始图像
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img, cmap=img_cmap)
    ax_img.set_title('原始图像', fontsize=12)
    ax_img.axis('off')
    
    # 显示每个注意力头的平均注意力
    for head in range(num_heads):
        ax = fig.add_subplot(gs[0, head + 1])
        
        # 计算平均注意力权重
        avg_attn = np.mean(attn[head], axis=0)  # [seq_len]
        attn_map = avg_attn.reshape(grid_size, grid_size)
        
        im = ax.imshow(attn_map, cmap='hot', interpolation='bilinear')
        ax.set_title(f'注意力头 {head + 1}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 显示注意力权重矩阵
    for head in range(min(4, num_heads)):
        ax = fig.add_subplot(gs[1 + head // 2, (head % 2) * 2 + 1:(head % 2) * 2 + 3])
        
        im = ax.imshow(attn[head], cmap='Blues', aspect='auto')
        ax.set_title(f'注意力矩阵 - 头 {head + 1}', fontsize=10)
        ax.set_xlabel('键位置')
        ax.set_ylabel('查询位置')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🔍 注意力热图已保存: {save_path}")


def create_comparison_grid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    save_path: str,
    title: str = "真实 vs 生成图像对比"
) -> None:
    """
    创建真实图像与生成图像的对比网格
    
    参数:
        real_images: 真实图像 [batch_size, channels, height, width]
        generated_images: 生成图像 [batch_size, channels, height, width]
        save_path: 保存路径
        title: 图表标题
    """
    batch_size = min(real_images.shape[0], generated_images.shape[0])
    
    # 创建交替排列的图像
    comparison_images = []
    for i in range(batch_size):
        comparison_images.append(real_images[i])
        comparison_images.append(generated_images[i])
    
    comparison_tensor = torch.stack(comparison_images)
    
    # 创建网格
    grid = make_grid(
        comparison_tensor,
        nrow=2,  # 每行2张图：真实，生成
        normalize=True,
        value_range=(-1, 1),
        pad_value=1
    )
    
    # 绘制
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)
        cmap = 'gray'
    else:
        cmap = None
    
    fig, ax = plt.subplots(figsize=(8, batch_size * 2))
    ax.imshow(grid_np, cmap=cmap)
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    
    # 添加标签
    for i in range(batch_size):
        y_pos = i * (real_images.shape[2] + 2) + real_images.shape[2] // 2
        
        # 真实图像标签
        ax.text(real_images.shape[3] // 2, y_pos, '真实', 
               ha='center', va='center', fontsize=12, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
        
        # 生成图像标签  
        ax.text(real_images.shape[3] + 2 + generated_images.shape[3] // 2, y_pos, '生成',
               ha='center', va='center', fontsize=12, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🔄 对比网格已保存: {save_path}")


def create_interpolation_grid(
    start_image: torch.Tensor,
    end_image: torch.Tensor,
    interpolated_images: List[torch.Tensor],
    save_path: str,
    title: str = "图像插值"
) -> None:
    """
    创建图像插值可视化
    
    参数:
        start_image: 起始图像 [channels, height, width]
        end_image: 结束图像 [channels, height, width]
        interpolated_images: 插值图像列表
        save_path: 保存路径
        title: 图表标题
    """
    # 组合所有图像
    all_images = [start_image] + interpolated_images + [end_image]
    all_tensor = torch.stack(all_images)
    
    # 创建水平网格
    grid = make_grid(
        all_tensor,
        nrow=len(all_images),
        normalize=True,
        value_range=(-1, 1),
        pad_value=1
    )
    
    # 绘制
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)
        cmap = 'gray'
    else:
        cmap = None
    
    fig, ax = plt.subplots(figsize=(len(all_images) * 2, 4))
    ax.imshow(grid_np, cmap=cmap)
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    
    # 添加插值比例标签
    for i, ratio in enumerate(np.linspace(0, 1, len(all_images))):
        x_pos = i * (start_image.shape[2] + 2) + start_image.shape[2] // 2
        y_pos = start_image.shape[1] + 10
        ax.text(x_pos, y_pos, f'{ratio:.2f}',
               ha='center', va='top', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🌈 插值网格已保存: {save_path}")


# 测试函数
def test_visualization():
    """
    测试可视化函数
    """
    print("🧪 测试可视化函数...")
    
    # 创建测试数据
    train_losses = [2.5, 2.1, 1.8, 1.5, 1.3, 1.1, 0.9, 0.8, 0.7, 0.6]
    val_losses = [2.6, 2.2, 1.9, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.75]
    
    # 测试训练曲线
    os.makedirs('./test_viz', exist_ok=True)
    plot_training_curves(train_losses, val_losses, './test_viz')
    
    # 测试样本网格
    samples = torch.randn(16, 3, 32, 32)
    create_sample_grid(samples, './test_viz/samples.png')
    
    print("✅ 可视化测试完成!")


if __name__ == "__main__":
    test_visualization()