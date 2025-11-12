import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import numpy as np

# -------------------------- 基础组件（复用你的代码） --------------------------
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x),** kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------- 窗口注意力相关（复用你的代码） --------------------------
def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size **2, window_size** 2)
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask


def get_relative_distances(window_size):
    assert window_size > 0, f"窗口大小必须为正整数，当前输入：{window_size}"
    indices = torch.tensor([[x, y] for x in range(window_size) for y in range(window_size)], dtype=torch.long)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size=4, relative_pos_embedding=True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim **-0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = self.window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size=self.window_size, displacement=displacement,
                            upper_lower=True, left_right=False),
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size=self.window_size, displacement=displacement,
                            upper_lower=False, left_right=True),
                requires_grad=False
            )

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(self.window_size) + self.window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * self.window_size - 1, 2 * self.window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(self.window_size** 2, self.window_size **2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        assert nw_h * self.window_size == n_h and nw_w * self.window_size == n_w, \
            f"特征图尺寸({n_h}×{n_w})不能被窗口大小({self.window_size}×{self.window_size})整除！"

        q, k, v = map(
            lambda t: rearrange(
                t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                h=h, w_h=self.window_size, w_w=self.window_size
            ), qkv
        )

        dots = einsum(q, k, 'b h w i d, b h w j d -> b h w i j') * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            window_indices = torch.arange(nw_h * nw_w, device=x.device).reshape(nw_h, nw_w)
            upper_lower_window_idx = window_indices[-1, :].flatten()
            dots[:, :, upper_lower_window_idx] += self.upper_lower_mask
            left_right_window_idx = window_indices[:, -1].flatten()
            dots[:, :, left_right_window_idx] += self.left_right_mask

        attn = dots.softmax(dim=-1)
        out = einsum(attn, v, 'b h w i j, b h w j d -> b h w i d')

        out = rearrange(
            out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
            h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w
        )
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


# -------------------------- Swin块与Stage（复用你的代码） --------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size=4, relative_pos_embedding=True):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(
            dim=dim, heads=heads, head_dim=head_dim, shifted=shifted,
            window_size=window_size, relative_pos_embedding=relative_pos_embedding
        )))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(
            dim=dim, hidden_dim=mlp_dim, dropout=0.3
        )))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor=2):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(
            kernel_size=downscaling_factor,
            stride=downscaling_factor,
            padding=0
        )
        self.linear = nn.Linear(in_channels * (downscaling_factor **2), out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.downscaling_factor
        new_h, new_w = h // k, w // k
        assert new_h * k == h and new_w * k == w, \
            f"输入尺寸（h={h}, w={w}）必须能被下采样因子{k}整除"

        x = self.patch_merge(x)
        x = x.permute(0, 2, 1).reshape(b, new_h, new_w, c * k * k)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, num_heads, head_dim,
                 downscaling_factor=2, window_size=4, relative_pos_embedding=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage层数必须为偶数'

        self.patch_partition = PatchMerging(
            in_channels=in_channels,
            out_channels=hidden_dimension,
            downscaling_factor=downscaling_factor
        )

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(
                    dim=hidden_dimension, heads=num_heads, head_dim=head_dim,
                    mlp_dim=hidden_dimension * 4, shifted=False,
                    window_size=window_size, relative_pos_embedding=relative_pos_embedding
                ),
                SwinBlock(
                    dim=hidden_dimension, heads=num_heads, head_dim=head_dim,
                    mlp_dim=hidden_dimension * 4, shifted=True,
                    window_size=window_size, relative_pos_embedding=relative_pos_embedding
                ),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)  # (b, h, w, c) → (b, c, h, w)


# -------------------------- 目标检测适配：添加检测头 --------------------------
class DetectionHead(nn.Module):
    """检测头：输出类别分数和边界框（水平框：xmin, ymin, xmax, ymax）"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 分类头（输出每个位置的类别概率）
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        # 回归头（输出边界框坐标偏移）
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 4, kernel_size=1)  # 4个坐标
        )

    def forward(self, x):
        cls_logits = self.cls_head(x)  # (b, num_classes, h, w)
        bbox_preds = self.bbox_head(x)  # (b, 4, h, w)
        return cls_logits, bbox_preds


# -------------------------- DOTA专用Swin检测模型 --------------------------
class DroneSwinDetector(nn.Module):
    """适配DOTA的Swin目标检测模型"""
    def __init__(self,
                 hidden_dim=64,
                 layers=(2, 2, 4, 2),
                 heads=(2, 4, 8, 16),
                 channels=3,
                 num_classes=15,  # DOTA有15类（不含背景）
                 head_dim=32,
                 window_size=4,
                 img_size=640):  # 输入图像尺寸
        super().__init__()
        self.downscaling_factors = (2, 2, 2, 2)
        self.num_classes = num_classes
        self.img_size = img_size

        # 主干网络（4个Stage）
        self.stage1 = StageModule(
            in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
            downscaling_factor=self.downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
            window_size=window_size
        )
        self.stage2 = StageModule(
            in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
            downscaling_factor=self.downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
            window_size=window_size
        )
        self.stage3 = StageModule(
            in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
            downscaling_factor=self.downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
            window_size=window_size
        )
        self.stage4 = StageModule(
            in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
            downscaling_factor=self.downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
            window_size=window_size
        )

        # 检测头（使用最后一个Stage的输出）
        self.detection_head = DetectionHead(
            in_channels=hidden_dim * 8,
            num_classes=num_classes
        )

    def forward(self, img):
        # 输入尺寸校验
        b, c, h, w = img.shape
        assert h == w == self.img_size, f"输入尺寸必须为{self.img_size}×{self.img_size}"

        # 主干网络特征提取
        x1 = self.stage1(img)       # (b, 64, 320, 320)  下采样2倍
        x2 = self.stage2(x1)        # (b, 128, 160, 160) 下采样4倍
        x3 = self.stage3(x2)        # (b, 256, 80, 80)   下采样8倍
        x4 = self.stage4(x3)        # (b, 512, 40, 40)   下采样16倍

        # 检测头输出
        cls_logits, bbox_preds = self.detection_head(x4)  # (b, 15, 40, 40), (b, 4, 40, 40)
        return cls_logits, bbox_preds