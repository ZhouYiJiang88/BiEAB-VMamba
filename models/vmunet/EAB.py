import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelEAB(nn.Module):
    """
    并行增强注意力块 (Parallel Enhanced Attention Block)
    符合论文描述的并行双路径结构
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim

        # 并行双路径：通道注意力路径
        self.channel_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

        # 并行双路径：空间注意力路径（使用7x7大卷积核）
        self.spatial_path = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),  # 7x7卷积核
            nn.Sigmoid()
        )

        # 自适应融合权重（可学习参数）
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))  # 通道路径权重
        self.fusion_beta = nn.Parameter(torch.tensor(0.5))  # 空间路径权重

        # 细化卷积
        self.refine_conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入x的形状: [B, H, W, C]
        x_permuted = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        B, C, H, W = x_permuted.shape

        # ==================== 并行双路径处理 ====================
        # 路径1：通道注意力（并行）
        ca_weights = self.channel_path(x_permuted)  # [B, C, 1, 1]
        ca_output = x_permuted * ca_weights  # 通道注意力调制

        # 路径2：空间注意力（并行）
        # 生成空间注意力输入
        avg_out = torch.mean(x_permuted, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x_permuted, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        sa_weights = self.spatial_path(spatial_input)  # [B, 1, H, W]
        sa_output = x_permuted * sa_weights  # 空间注意力调制

        # ==================== 自适应融合 ====================
        # 使用可学习的权重进行融合
        alpha = torch.sigmoid(self.fusion_alpha)
        beta = torch.sigmoid(self.fusion_beta)

        # 归一化权重，确保总和为1
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total

        # 并行路径融合
        fused_output = alpha * ca_output + beta * sa_output

        # ==================== 细化处理 ====================
        refined = self.refine_conv(fused_output)
        refined = self.norm(refined)
        refined = self.act(refined)

        # 残差连接
        output = refined + x_permuted

        return output.permute(0, 2, 3, 1)  # [B, H, W, C]

    def extra_repr(self):
        return f'dim={self.dim}, fusion_alpha={self.fusion_alpha.item():.3f}, fusion_beta={self.fusion_beta.item():.3f}'