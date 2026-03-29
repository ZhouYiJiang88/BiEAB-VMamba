import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models.vmunet.EAB import ParallelEAB

# 在文件开头添加导入

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    # 估算选择性扫描（SSM核心操作）的计算量（FLOPs）
    # 通过模拟矩阵运算路径，计算 einsum 等操作的浮点运算次数
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


#通过此模块，图像被转为合适的Transformer处理的序列形式，每个patch对应一个高维向量，为后续的自注意力计算提供基础
class PatchEmbed2D(nn.Module):  #将2D图像分割为固定大小的块并映射到嵌入空间，
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self,
                 patch_size=4, #  patch_size=4 表示将图像划分为 4×4 像素的块，输出特征图尺寸为原图的 1/4（如 224×224 → 56×56）。
                 in_chans=3, #输入通道数 RGB为3  灰度图为1
                 embed_dim=96, #输出嵌入向量的维度，决定每个patch的表示能力
                 norm_layer=None, #可选归一化层，用于稳定训练
                 **kwargs):

        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)  #确保patch-size为元组形式
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  #卷积投影层（self.proj）
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):  #前向传播
        x = self.proj(x).permute(0, 2, 3, 1) # [B, C, H, W] → [B, H, W, C]
        if self.norm is not None:
            x = self.norm(x)
        return x

#下采样   将特征图尺寸从H×W降至H/2×W/2     输入通道C→合并后4C→线性投影至2C，实现特征压缩与信息聚合
class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim   # 输入通道数C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)    # 4C→2C的线性投影
        self.norm = norm_layer(4 * dim)   # 归一化层（默认LayerNorm）

    def forward(self, x):
        B, H, W, C = x.shape    # 输入形状：[Batch, Height, Width, Channels]

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):   # 处理奇数分辨率
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2  # 强制对齐到偶数尺寸

#通过start::step语法实现网格采样，等效于将图像划分为2×2网格
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C  # 通道维度拼接：B H/2 W/2 4C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
#上采样  将输入特征图从H×W扩展到(H*2)×(W*2)   通道调整：通过线性投影调整通道维度，同时保持特征信息完整性
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2   # 输入通道数×2（假设来自跳跃连接）
        self.dim_scale = dim_scale   # 上采样倍数（默认2倍）
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)  # 通道扩展
        self.norm = norm_layer(self.dim // dim_scale)   # 归一化层

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)   # [B,H,W,C] → [B,H,W,2*C]（当dim_scale=2时）
        #空间重组
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)  # 对重组后的特征归一化

        return x
    
#多倍率上采样
class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入通道数（如Swin-Tiny中Stage4的768维
        self.dim_scale = dim_scale # 上采样倍数（默认4倍）
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)  #通道扩展
        self.norm = norm_layer(self.dim // dim_scale)   #归一化层

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)   # [B,H,W,C] → [B,H,W,4*C]（当dim_scale=4时）  为后续空间重组做准备
        #空间重组（核心操作）
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)  #对重组后的特征归一化

        return x

#核心模块 SS2D是一个基于状态空间模型（SSM）的二维视觉特征处理模块，通过结合深度可分离卷积多方向选择性扫描，实现了对图像特征的全局建模与局部感知融合。其核心创新点包括
#四方向扫描机制：水平、垂直、正/反对角线四个方向的序列建模（类似Swin Transformer的窗口注意力但更高效）
#参数化状态空间：通过可学习的A_logs、Bs、Cs、Ds参数实现动态特征演化
#轻量化设计：深度可分离卷积减少计算量，dt_rank压缩时间步参数
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,     #输入特征维度
        d_state=16,  #状态空间维度
        # d_state="auto", # 20240109
        d_conv=3,   #卷积核大小
        expand=2,   #内部扩展倍数
        dt_rank="auto",  #时间步参数秩
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)   # 维度扩展：d_inner = expand * d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影：分成两部分（x和z）其中x用于后续的局部特征提取和状态空间建模，z用于最终的特征缩放（类似 Transformer 中的门控机制）。
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(   # 深度可分离卷积（增强局部性）
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  # 分组数=输入通道数→深度卷积
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU() #非线性激活（self.act）


        # 四组状态空间参数投影（对应四个扫描方向）
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        # 将4个线性层的权重堆叠为一个参数张量（形状：[4, 输出维度, 输入维度]）
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj # 删除原始线性层，仅保留堆叠后的权重参数

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        # 堆叠4个方向的权重和偏置为参数张量
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs # 删除原始投影层，保留堆叠后的参数

        # 状态空间参数初始化 # 状态转移矩阵A的对数形式（4个方向，合并为一个参数）
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)  # 状态矩阵（对数形式）
        # 跳跃连接参数D（4个方向，合并为一个参数）
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)  )  # 跳跃连接参数

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn #绑定选择性扫描的底层实现函数（高效计算 SSM 的核心操作）。
        
        B, C, H, W = x.shape
        L = H * W
        K = 4
        # 四方向扫描：水平、垂直、对角线
        x_hwwh = torch.stack(
            [x.view(B, -1, L), # 水平方向：展平为 [B, C, L]（H×W顺序）
             torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)  # 垂直方向：转置后展平为 [B, C, L]（W×H顺序）
             ], dim=1).view(B, 2, -1, L)# 合并为 [B, 2, C, L]

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l) # 正对角线和反对角线（通过翻转水平/垂直序列实现）

        # 1. 投影得到dt_rank、B_state、C_state（合并4个方向计算）
        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L),  # [B,4,C,L]
                             self.x_proj_weight)    # [4, dt_rank+2*d_state, C]   # 输出 [B,4, dt_rank+2*d_state, L]

        # # 2. 拆分参数：dt_rank（时间步参数）、B_state（输入矩阵）、C_state（输出矩阵）
        dts, Bs, Cs = torch.split(x_dbl,
                                  [self.dt_rank, self.d_state, self.d_state],  # 拆分维度
                                  dim=2) # 按通道维度拆分 # 分别得到 [B,4,dt_rank,L]、[B,4,d_state,L]、[B,4,d_state,L]

        # 3. 映射dt_rank到通道维度（调整时间步参数维度）
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L),  # [B,4,dt_rank,L]
                           self.dt_projs_weight)   # [4, C, dt_rank] # 输出 [B,4,C,L]（时间步参数适配通道维度）
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)


        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # 选择性状态扫描
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        #将 4 个方向的输出重新映射回原始空间维度，并合并为最终特征：
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4
        # 四方向特征展开：水平、垂直、正/反对角线
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)   # (B, 4, d_inner, L)

        # 投影得到时间步(dt)、状态参数(B/C)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)  # (B, 4, d_inner, L)
        assert out_y.dtype == torch.float
         # 方向特征重组
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        # --------------------------
        # 1. 正向扫描（原始逻辑）
        # --------------------------
        x_forward = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        x_forward = self.act(self.conv2d(x_forward))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x_forward)
        assert y1.dtype == torch.float32
        y_forward = y1 + y2 + y3 + y4  # 正向特征融合

        # --------------------------
        # 2. 反向扫描（新增逻辑）
        # --------------------------
        # 对输入进行空间翻转（水平+垂直）
        x_reverse = x.flip(dims=[1, 2])  # 翻转高度和宽度维度 [B, H, W, C]
        x_reverse = x_reverse.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x_reverse = self.act(self.conv2d(x_reverse))
        y1_rev, y2_rev, y3_rev, y4_rev = self.forward_core(x_reverse)
        y_reverse = y1_rev + y2_rev + y3_rev + y4_rev

        # 将反向扫描结果翻转回原始空间方向
        y_reverse = y_reverse.flip(dims=[-2, -1])  # 翻转最后两个空间维度

        # --------------------------
        # 3. 双向特征融合
        # --------------------------
        # 可选择不同的融合策略：

        # 策略1: 简单相加
        y = y_forward + y_reverse

        # 策略2: 加权融合（可调节权重）
        # forward_weight, reverse_weight = 0.6, 0.4
        # y = y_forward * forward_weight + y_reverse * reverse_weight

        # 策略3: 自适应门控融合
        # gate = torch.sigmoid((y_forward + y_reverse) / 2)  # 自适应门控
        # y = y_forward * gate + y_reverse * (1 - gate)

        # --------------------------
        # 4. 后续处理（保持不变）
        # --------------------------
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        use_lsb: bool = True,  # 新增参数控制是否使用LSB
        lsb_reduction: int = 4,  # LSB的reduction ratio
        use_parallel: bool = True,  # 新增：控制使用并行还是串行
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        # 根据use_parallel选择使用并行或串行EAB
        self.use_parallel = use_parallel
        if self.use_parallel:
            # 使用并行EAB
            self.eab = ParallelEAB(dim=hidden_dim, reduction=lsb_reduction)
            print(
                f"使用并行EAB，融合权重: alpha={self.eab.fusion_alpha.item():.3f}, beta={self.eab.fusion_beta.item():.3f}")


        self.eab_norm = norm_layer(hidden_dim)
        self.eab_drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # 原始VSS块的前向传播
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))

        # EAB处理（并行或串行）
        x = x + self.eab_drop_path(self.eab(self.eab_norm(x)))

        return x

class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        use_lsb=True,  # 新增参数
        lsb_reduction=4,  # 新增参数
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_lsb=use_lsb,  # 传递参数
                lsb_reduction=lsb_reduction,  # 传递参数
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        use_lsb=True,  # 新增参数
        lsb_reduction=4,  # 新增参数
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_lsb=use_lsb,  # 传递参数
                lsb_reduction=lsb_reduction,  # 传递参数
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in `VSS`M!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    

#编码器-解码器结构
class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96],d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,   #原来：
                 norm_layer=nn.LayerNorm, patch_norm=True,use_checkpoint=False,  use_lsb=True,    lsb_reduction=4, use_parallel_eab=True,   **kwargs):

        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0

        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_lsb=use_lsb,  # 传递参数
                lsb_reduction=lsb_reduction,  # 传递参数
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                use_lsb=use_lsb,  # 传递参数
                lsb_reduction=lsb_reduction,  # 传递参数
            )
            self.layers_up.append(layer)

        # 在VSSM的__init__中，替换attention_fusions初始化部分
        self.attention_fusions = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                self.attention_fusions.append(None)  # 第0层无需融合
            else:
                # 解码器当前层的通道数为dims_decoder[i_layer]
                # 输入为x（dims_decoder[i_layer]）和skip（dims[i_layer-1]）的拼接
                # 注意：编码器dims与解码器dims_decoder的对应关系是反转的
                in_channels = dims_decoder[i_layer] + dims[-(i_layer)]  # 关键修正
                out_channels = dims_decoder[i_layer]
                self.attention_fusions.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            bias=False
                        ),
                        nn.Sigmoid()
                    )
                )



        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)



    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):  #编码器部分
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)   #这里存储跳跃链接
            x = layer(x)
        return x, skip_list

    # 在 vmamba.py 的 VSSM 类中修改 forward_features_up 方法
    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # 检查并调整跳跃连接的特征通道数
                skip = skip_list[-inx]
                if x.size(1) != skip.size(1):
                    # 若通道数不匹配，使用1x1卷积调整
                    skip = nn.Conv2d(skip.size(1), x.size(1), kernel_size=1).to(skip.device)(skip)
                x = layer_up(x + skip)  # 或使用拼接后通过卷积调整：x = layer_up(torch.cat([x, skip], dim=1))
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        
        return x




    


