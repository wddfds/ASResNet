import math

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)


        q, k = F.interpolate(q, scale_factor=0.5), F.interpolate(q, scale_factor=0.5)
        h, w = h//2, w//2
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        h, w = h * 2, w * 2
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    All bias set false
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)

        ffn_channel = ffn_expansion_factor * dim
        self.conv_ff1 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
                                  groups=1, bias=False)
        self.dwconv = nn.Conv2d(ffn_channel, ffn_channel, kernel_size=3, stride=1, padding=1, groups=ffn_channel, bias=False)
        self.sg = SimpleGate()
        self.conv_ff2 = nn.Conv2d(in_channels=int(ffn_channel) // 2, out_channels=dim, kernel_size=1, padding=0,
                                  stride=1, groups=1, bias=False)

        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        input = x
        y = self.beta * x + self.attn(self.norm1(x))  # rescale

        y = self.conv_ff1(self.norm2(y))
        y = self.dwconv(y)
        y = self.sg(y)
        y = self.gamma * y + self.conv_ff2(x)

        return y + input