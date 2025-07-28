#!/usr/bin/env python3
"""
UNet网络架构实现 - Diffusion模型的核心组件
=======================================

UNet是扩散模型中最重要的组件，负责预测和去除噪声。
这个实现包含了现代UNet的所有关键特性：
1. 时间步嵌入 (Time Embedding)
2. 残差连接 (Residual Connections)
3. 注意力机制 (Attention Mechanisms)
4. 跳跃连接 (Skip Connections)

作者: Diffusion教程团队
日期: 2024年
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class TimeEmbedding(nn.Module):
    """
    时间步嵌入层
    
    将离散的时间步t转换为连续的特征表示，
    使模型能够感知当前处于扩散过程的哪个阶段。
    """
    
    def __init__(self, dim: int):
        """
        初始化时间嵌入层
        
        参数:
            dim: 嵌入维度
        """
        super().__init__()
        self.dim = dim
        
        # 使用正弦位置编码的思想
        # 不同频率的正弦和余弦函数可以唯一编码时间步
        half_dim = dim // 2
        self.register_buffer(
            'freqs',
            torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
        )
        
        # MLP层将位置编码映射到更高维空间
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),  # Swish激活函数，在扩散模型中表现更好
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            timesteps: 时间步张量 [batch_size]
            
        返回:
            time_emb: 时间嵌入 [batch_size, dim]
        """
        # 计算正弦位置编码
        # timesteps: [batch_size] -> [batch_size, 1]
        timesteps = timesteps.unsqueeze(1).float()
        
        # 广播相乘: [batch_size, 1] * [half_dim] -> [batch_size, half_dim]
        args = timesteps * self.freqs
        
        # 拼接正弦和余弦编码
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        
        # 通过MLP进一步处理
        return self.mlp(embeddings)


class ResidualBlock(nn.Module):
    """
    残差块 - UNet的基本构建单元
    
    包含：
    1. GroupNorm标准化
    2. 卷积层
    3. 时间嵌入注入
    4. 残差连接
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int,
        groups: int = 8
    ):
        """
        初始化残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            time_emb_dim: 时间嵌入维度
            groups: GroupNorm的组数
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 第一个卷积路径
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 时间嵌入投影层
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # 第二个卷积路径
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 残差连接的投影层（当输入输出通道数不同时）
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征图 [batch_size, in_channels, height, width]
            time_emb: 时间嵌入 [batch_size, time_emb_dim]
            
        返回:
            output: 输出特征图 [batch_size, out_channels, height, width]
        """
        # 保存残差连接
        residual = self.residual_proj(x)
        
        # 第一个卷积块
        h = self.norm1(x)
        h = F.silu(h)  # Swish激活
        h = self.conv1(h)
        
        # 注入时间信息
        # time_emb: [batch_size, time_emb_dim] -> [batch_size, out_channels]
        time_proj = self.time_proj(time_emb)
        # 添加空间维度: [batch_size, out_channels] -> [batch_size, out_channels, 1, 1]
        time_proj = time_proj.unsqueeze(-1).unsqueeze(-1)
        h = h + time_proj
        
        # 第二个卷积块
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + residual


class AttentionBlock(nn.Module):
    """
    自注意力块
    
    在特征图的空间维度上应用自注意力机制，
    帮助模型捕获长距离依赖关系。
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        """
        初始化注意力块
        
        参数:
            channels: 通道数
            num_heads: 注意力头数
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels必须能被num_heads整除"
        
        # 标准化层
        self.norm = nn.GroupNorm(8, channels)
        
        # 查询、键、值投影层
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征图 [batch_size, channels, height, width]
            
        返回:
            output: 输出特征图 [batch_size, channels, height, width]
        """
        batch_size, channels, height, width = x.shape
        residual = x
        
        # 标准化
        x = self.norm(x)
        
        # 计算QKV
        qkv = self.to_qkv(x)  # [B, C*3, H, W]
        
        # 重塑为多头注意力格式
        qkv = qkv.view(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力权重
        # [B, num_heads, H*W, head_dim] @ [B, num_heads, head_dim, H*W] 
        # -> [B, num_heads, H*W, H*W]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        # [B, num_heads, H*W, H*W] @ [B, num_heads, H*W, head_dim]
        # -> [B, num_heads, H*W, head_dim]
        out = torch.matmul(attention_weights, v)
        
        # 重塑回原始格式
        out = out.permute(0, 1, 3, 2)  # [B, num_heads, head_dim, H*W]
        out = out.contiguous().view(batch_size, channels, height, width)
        
        # 输出投影
        out = self.to_out(out)
        
        # 残差连接
        return out + residual


class DownBlock(nn.Module):
    """
    下采样块 - UNet编码器的组件
    
    包含多个残差块和可选的注意力块，
    最后进行下采样操作。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        downsample: bool = True
    ):
        super().__init__()
        
        # 残差块序列
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.layers.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim)
            )
            
            # 在最后一层添加注意力（如果需要）
            if use_attention and i == num_layers - 1:
                self.layers.append(AttentionBlock(out_channels))
        
        # 下采样层
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1) \
            if downsample else None
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        
        参数:
            x: 输入特征图
            time_emb: 时间嵌入
            
        返回:
            x: 输出特征图
            skip_connections: 跳跃连接特征
        """
        skip_connections = []
        
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:  # AttentionBlock
                x = layer(x)
            skip_connections.append(x)
        
        if self.downsample:
            x = self.downsample(x)
        
        return x, skip_connections


class UpBlock(nn.Module):
    """
    上采样块 - UNet解码器的组件
    
    接收跳跃连接并进行上采样，
    恢复原始分辨率。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        upsample: bool = True
    ):
        super().__init__()
        
        # 上采样层
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) \
            if upsample else None
        
        # 残差块序列
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 第一层需要处理跳跃连接的拼接
            in_ch = in_channels * 2 if i == 0 else out_channels
            self.layers.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim)
            )
            
            # 在最后一层添加注意力（如果需要）
            if use_attention and i == num_layers - 1:
                self.layers.append(AttentionBlock(out_channels))
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_connections: List[torch.Tensor],
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征图
            skip_connections: 来自编码器的跳跃连接
            time_emb: 时间嵌入
            
        返回:
            x: 输出特征图
        """
        if self.upsample:
            x = self.upsample(x)
        
        # 逆序处理跳跃连接
        skip_connections = skip_connections[::-1]
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ResidualBlock):
                if i == 0:
                    # 第一层：拼接跳跃连接
                    skip = skip_connections[i] if i < len(skip_connections) else None
                    if skip is not None:
                        x = torch.cat([x, skip], dim=1)
                x = layer(x, time_emb)
            else:  # AttentionBlock
                x = layer(x)
        
        return x


class UNet(nn.Module):
    """
    完整的UNet架构
    
    用于扩散模型的噪声预测网络。
    采用编码器-解码器结构，具有跳跃连接。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        num_layers: int = 2,
        attention_levels: List[bool] = [False, False, True, True],
        time_emb_dim: int = 512
    ):
        """
        初始化UNet
        
        参数:
            in_channels: 输入通道数（RGB图像为3）
            out_channels: 输出通道数（通常等于输入通道数）
            channels: 各层的通道数配置
            num_layers: 每个块的残差层数
            attention_levels: 各层是否使用注意力机制
            time_emb_dim: 时间嵌入维度
        """
        super().__init__()
        
        # 时间嵌入层
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # 编码器（下采样路径）
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = channels[i-1] if i > 0 else channels[0]
            out_ch = channels[i]
            is_last = i == len(channels) - 1
            
            self.down_blocks.append(
                DownBlock(
                    in_ch, out_ch, time_emb_dim,
                    num_layers=num_layers,
                    use_attention=attention_levels[i],
                    downsample=not is_last
                )
            )
        
        # 中间块（最深层）
        mid_ch = channels[-1]
        self.mid_block = nn.Sequential(
            ResidualBlock(mid_ch, mid_ch, time_emb_dim),
            AttentionBlock(mid_ch),
            ResidualBlock(mid_ch, mid_ch, time_emb_dim)
        )
        
        # 解码器（上采样路径）
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        reversed_attention = list(reversed(attention_levels))
        
        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i+1] if i < len(reversed_channels)-1 else channels[0]
            is_last = i == len(reversed_channels) - 1
            
            self.up_blocks.append(
                UpBlock(
                    in_ch, out_ch, time_emb_dim,
                    num_layers=num_layers + 1,  # +1因为要处理跳跃连接
                    use_attention=reversed_attention[i],
                    upsample=not is_last
                )
            )
        
        # 输出层
        self.output_norm = nn.GroupNorm(8, channels[0])
        self.output_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入图像 [batch_size, in_channels, height, width]
            timesteps: 时间步 [batch_size]
            
        返回:
            noise_pred: 预测的噪声 [batch_size, out_channels, height, width]
        """
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)
        
        # 输入投影
        x = self.input_conv(x)
        
        # 编码器前向传播，收集跳跃连接
        skip_connections_list = []
        for down_block in self.down_blocks:
            x, skip_connections = down_block(x, time_emb)
            skip_connections_list.extend(skip_connections)
        
        # 中间块
        for layer in self.mid_block:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:  # AttentionBlock
                x = layer(x)
        
        # 解码器前向传播，使用跳跃连接
        # 将跳跃连接按解码器块分组
        skip_idx = len(skip_connections_list)
        for up_block in self.up_blocks:
            # 为每个上采样块分配对应的跳跃连接
            block_skip_connections = []
            num_layers = len([l for l in up_block.layers if isinstance(l, ResidualBlock)])
            
            # 从后往前取跳跃连接
            for _ in range(num_layers - 1):  # -1因为最后一层不需要跳跃连接
                if skip_idx > 0:
                    skip_idx -= 1
                    block_skip_connections.append(skip_connections_list[skip_idx])
            
            x = up_block(x, block_skip_connections, time_emb)
        
        # 输出层
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)
        
        return x


# 测试函数
def test_unet():
    """
    测试UNet网络的功能
    """
    print("🧪 测试UNet网络...")
    
    # 创建模型
    model = UNet(
        in_channels=3,
        out_channels=3,
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True]
    )
    
    # 创建测试数据
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # 前向传播
    with torch.no_grad():
        output = model(x, timesteps)
    
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print(f"✅ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    test_unet()