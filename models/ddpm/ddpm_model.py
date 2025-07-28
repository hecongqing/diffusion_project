#!/usr/bin/env python3
"""
DDPM (Denoising Diffusion Probabilistic Models) 实现
=================================================

这个文件实现了完整的DDPM模型，包括：
1. 噪声调度器 (Noise Scheduler)
2. UNet去噪网络 (Denoising Network)  
3. 训练和推理逻辑 (Training & Inference)

参考论文:
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- https://arxiv.org/abs/2006.11239

作者: Diffusion教程团队
日期: 2024年
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math

class NoiseScheduler:
    """
    DDPM噪声调度器
    
    负责管理扩散过程中的噪声添加和参数计算。
    实现了线性噪声调度策略。
    """
    
    def __init__(
        self, 
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear"
    ):
        """
        初始化噪声调度器
        
        参数:
            num_timesteps: 扩散总步数 T
            beta_start: 噪声调度起始值
            beta_end: 噪声调度结束值
            schedule: 调度策略 ("linear" 或 "cosine")
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # 1. 计算噪声调度 β_t
        if schedule == "linear":
            # 线性调度：β_t 从 beta_start 线性增长到 beta_end
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            # 余弦调度：更平滑的噪声添加策略
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"未知的调度策略: {schedule}")
        
        # 2. 计算相关参数
        self.alphas = 1.0 - self.betas  # α_t = 1 - β_t
        
        # 累积乘积：ᾱ_t = ∏(α_i) for i=1 to t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 前一步的累积乘积：ᾱ_{t-1}
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 3. 预计算采样所需的系数
        # 用于前向过程：q(x_t|x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # √ᾱ_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # √(1-ᾱ_t)
        
        # 用于反向过程：p(x_{t-1}|x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/√α_t
        
        # 后验方差：β̃_t = β_t * (1-ᾱ_{t-1})/(1-ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # 后验均值系数
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """
        余弦噪声调度策略
        
        相比线性调度，余弦调度在训练早期添加更少噪声，
        在后期添加更多噪声，通常能获得更好的生成质量。
        """
        s = 0.008  # 小的偏移量
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        
        # 余弦函数
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # 计算 β_t
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(
        self, 
        original_samples: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        前向扩散过程：向原始样本添加噪声
        
        实现公式：x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        
        参数:
            original_samples: 原始样本 x_0 [batch_size, channels, height, width]
            noise: 高斯噪声 ε [batch_size, channels, height, width]  
            timesteps: 时间步 t [batch_size]
            
        返回:
            noisy_samples: 添加噪声后的样本 x_t
        """
        # 获取对应时间步的系数
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # 调整形状以便广播 [batch_size] -> [batch_size, 1, 1, 1]
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        # 应用重参数化技巧：x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        noisy_samples = (
            sqrt_alpha_cumprod * original_samples + 
            sqrt_one_minus_alpha_cumprod * noise
        )
        
        return noisy_samples
    
    def step(
        self, 
        model_output: torch.Tensor, 
        timestep: int, 
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        反向去噪步骤：从 x_t 预测 x_{t-1}
        
        实现DDPM的采样公式：
        x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z
        
        参数:
            model_output: 模型预测的噪声 ε_θ(x_t,t)
            timestep: 当前时间步 t
            sample: 当前样本 x_t
            generator: 随机数生成器
            
        返回:
            prev_sample: 前一步样本 x_{t-1}
        """
        # 获取当前时间步的参数
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        beta_t = self.betas[timestep]
        
        # 1. 计算预测的原始样本 x_0
        # x_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_cumprod_t) * model_output
        ) / torch.sqrt(alpha_cumprod_t)
        
        # 2. 计算前一步样本的均值
        # μ = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
        pred_prev_mean = (
            1.0 / torch.sqrt(alpha_t) * 
            (sample - beta_t / torch.sqrt(1 - alpha_cumprod_t) * model_output)
        )
        
        # 3. 添加噪声（除了最后一步）
        if timestep > 0:
            variance = self.posterior_variance[timestep]
            # 生成标准高斯噪声
            noise = torch.randn_like(sample, generator=generator)
            # 添加方差缩放的噪声
            prev_sample = pred_prev_mean + torch.sqrt(variance) * noise
        else:
            # 最后一步不添加噪声
            prev_sample = pred_prev_mean
        
        return prev_sample


class TimeEmbedding(nn.Module):
    """
    时间步嵌入模块
    
    将时间步 t 编码为高维向量，用于指导去噪网络。
    使用正弦位置编码的变体。
    """
    
    def __init__(self, dim: int):
        """
        参数:
            dim: 嵌入维度
        """
        super().__init__()
        self.dim = dim
        
        # 投影层
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),  # Swish激活函数
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            timesteps: 时间步 [batch_size]
            
        返回:
            嵌入向量 [batch_size, dim]
        """
        # 1. 正弦位置编码
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        
        # 2. 正弦和余弦
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        # 3. 通过投影网络
        embeddings = self.proj(embeddings)
        
        return embeddings


class ResidualBlock(nn.Module):
    """
    残差块
    
    UNet的基本构建模块，包含：
    - 组归一化
    - 卷积层  
    - 时间嵌入注入
    - 残差连接
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 第一个卷积块
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 时间嵌入投影
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # 第二个卷积块
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, in_channels, height, width]
            time_emb: 时间嵌入 [batch_size, time_emb_dim]
            
        返回:
            输出特征 [batch_size, out_channels, height, width]
        """
        # 保存残差
        residual = self.residual_conv(x)
        
        # 第一个卷积块
        h = self.norm1(x)
        h = F.silu(h)  # Swish激活
        h = self.conv1(h)
        
        # 注入时间信息
        time_emb_proj = self.time_proj(time_emb)  # [batch_size, out_channels]
        h = h + time_emb_proj[:, :, None, None]  # 广播到空间维度
        
        # 第二个卷积块
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + residual


class AttentionBlock(nn.Module):
    """
    自注意力块
    
    在UNet的特定层使用，帮助模型关注图像的不同区域。
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, channels, height, width]
            
        返回:
            输出特征 [batch_size, channels, height, width]
        """
        residual = x
        batch_size, channels, height, width = x.shape
        
        # 归一化
        h = self.norm(x)
        
        # 计算 Q, K, V
        qkv = self.qkv(h)  # [batch_size, channels*3, height, width]
        q, k, v = qkv.chunk(3, dim=1)  # 每个都是 [batch_size, channels, height, width]
        
        # 重塑为序列形式
        q = q.view(batch_size, channels, height * width).transpose(1, 2)  # [batch_size, hw, channels]
        k = k.view(batch_size, channels, height * width).transpose(1, 2)  # [batch_size, hw, channels]  
        v = v.view(batch_size, channels, height * width).transpose(1, 2)  # [batch_size, hw, channels]
        
        # 计算注意力
        scale = 1.0 / math.sqrt(channels)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # [batch_size, hw, hw]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        h = torch.bmm(attn, v)  # [batch_size, hw, channels]
        
        # 重塑回空间形式
        h = h.transpose(1, 2).view(batch_size, channels, height, width)
        
        # 投影
        h = self.proj(h)
        
        return h + residual


class UNet(nn.Module):
    """
    UNet去噪网络
    
    扩散模型的核心网络，负责预测噪声。
    采用编码器-解码器结构，具有跳跃连接。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1
    ):
        """
        初始化UNet
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            model_channels: 基础通道数
            num_res_blocks: 每个分辨率级别的残差块数量
            attention_resolutions: 使用注意力的分辨率级别
            channel_mult: 通道数倍增因子
            dropout: Dropout概率
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 编码器
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        input_ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            # 残差块
            for _ in range(num_res_blocks):
                block = ResidualBlock(input_ch, out_ch, time_embed_dim, dropout)
                self.down_blocks.append(block)
                input_ch = out_ch
                
                # 添加注意力（如果需要）
                if 32 // (2 ** level) in attention_resolutions:
                    attn = AttentionBlock(out_ch)
                    self.down_blocks.append(attn)
            
            # 下采样（除了最后一层）
            if level < len(channel_mult) - 1:
                downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
                self.down_samples.append(downsample)
            else:
                self.down_samples.append(nn.Identity())
        
        # 中间块
        mid_ch = model_channels * channel_mult[-1]
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_embed_dim, dropout)
        
        # 解码器
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            
            # 残差块
            for i in range(num_res_blocks + 1):
                # 第一个块需要处理跳跃连接
                in_ch = input_ch + (out_ch if i == 0 else 0)
                block = ResidualBlock(in_ch, out_ch, time_embed_dim, dropout)
                self.up_blocks.append(block)
                input_ch = out_ch
                
                # 添加注意力（如果需要）
                resolution = 32 // (2 ** (len(channel_mult) - 1 - level))
                if resolution in attention_resolutions:
                    attn = AttentionBlock(out_ch)
                    self.up_blocks.append(attn)
            
            # 上采样（除了最后一层）
            if level < len(channel_mult) - 1:
                upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
                self.up_samples.append(upsample)
            else:
                self.up_samples.append(nn.Identity())
        
        # 输出投影
        self.output_norm = nn.GroupNorm(8, model_channels)
        self.output_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 噪声图像 [batch_size, channels, height, width]
            timesteps: 时间步 [batch_size]
            
        返回:
            预测的噪声 [batch_size, channels, height, width]
        """
        # 时间嵌入
        time_emb = self.time_embed(timesteps)
        
        # 输入投影
        h = self.input_conv(x)
        
        # 编码器路径（保存跳跃连接）
        skip_connections = [h]
        
        block_idx = 0
        for level in range(len(self.channel_mult)):
            # 残差块
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, time_emb)
                block_idx += 1
                
                # 注意力块（如果存在）
                if isinstance(self.down_blocks[block_idx], AttentionBlock):
                    h = self.down_blocks[block_idx](h)
                    block_idx += 1
                
                skip_connections.append(h)
            
            # 下采样
            h = self.down_samples[level](h)
            if level < len(self.channel_mult) - 1:
                skip_connections.append(h)
        
        # 中间块
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # 解码器路径（使用跳跃连接）
        block_idx = 0
        for level in range(len(self.channel_mult)):
            # 残差块
            for i in range(self.num_res_blocks + 1):
                # 第一个块连接跳跃连接
                if i == 0:
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                
                h = self.up_blocks[block_idx](h, time_emb)
                block_idx += 1
                
                # 注意力块（如果存在）
                if isinstance(self.up_blocks[block_idx], AttentionBlock):
                    h = self.up_blocks[block_idx](h)
                    block_idx += 1
            
            # 上采样
            if level < len(self.channel_mult) - 1:
                h = self.up_samples[level](h)
        
        # 输出投影
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


class DDPM(nn.Module):
    """
    完整的DDPM模型
    
    集成了噪声调度器和UNet网络，提供训练和推理接口。
    """
    
    def __init__(
        self,
        unet: UNet,
        noise_scheduler: NoiseScheduler
    ):
        """
        参数:
            unet: UNet去噪网络
            noise_scheduler: 噪声调度器
        """
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播（训练时使用）
        
        参数:
            x: 噪声图像
            timesteps: 时间步
            
        返回:
            预测的噪声
        """
        return self.unet(x, timesteps)
    
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        训练步骤
        
        参数:
            batch: 干净图像批次 [batch_size, channels, height, width]
            
        返回:
            损失值
        """
        batch_size = batch.shape[0]
        device = batch.device
        
        # 1. 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps, 
            (batch_size,), device=device
        )
        
        # 2. 生成随机噪声
        noise = torch.randn_like(batch)
        
        # 3. 前向扩散：添加噪声
        noisy_images = self.noise_scheduler.add_noise(batch, noise, timesteps)
        
        # 4. 预测噪声
        noise_pred = self.unet(noisy_images, timesteps)
        
        # 5. 计算MSE损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int = 1,
        image_size: int = 32,
        channels: int = 3,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        采样生成图像
        
        参数:
            batch_size: 批次大小
            image_size: 图像尺寸
            channels: 通道数
            device: 设备
            generator: 随机数生成器
            
        返回:
            生成的图像 [batch_size, channels, image_size, image_size]
        """
        # 从纯噪声开始
        shape = (batch_size, channels, image_size, image_size)
        image = torch.randn(shape, device=device, generator=generator)
        
        # 逐步去噪
        for t in range(self.noise_scheduler.num_timesteps - 1, -1, -1):
            # 预测噪声
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(image, timesteps)
            
            # 去噪一步
            image = self.noise_scheduler.step(noise_pred, t, image, generator)
        
        return image


# 工厂函数
def create_ddpm_model(
    image_size: int = 32,
    channels: int = 3,
    num_timesteps: int = 1000
) -> DDPM:
    """
    创建DDPM模型的工厂函数
    
    参数:
        image_size: 图像尺寸
        channels: 通道数
        num_timesteps: 扩散步数
        
    返回:
        配置好的DDPM模型
    """
    # 创建噪声调度器
    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps)
    
    # 创建UNet网络
    unet = UNet(
        in_channels=channels,
        out_channels=channels,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        channel_mult=(1, 2, 4, 8) if image_size >= 64 else (1, 2, 4),
        dropout=0.1
    )
    
    # 创建完整模型
    model = DDPM(unet, noise_scheduler)
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试DDPM模型...")
    
    # 创建模型
    model = create_ddpm_model(image_size=32, channels=3, num_timesteps=100)
    
    # 创建测试数据
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 32, 32)
    
    # 测试训练步骤
    loss = model.training_step(test_images)
    print(f"✅ 训练损失: {loss.item():.4f}")
    
    # 测试采样
    with torch.no_grad():
        samples = model.sample(batch_size=2, image_size=32, channels=3)
        print(f"✅ 采样形状: {samples.shape}")
    
    print("🎉 DDPM模型测试完成！")