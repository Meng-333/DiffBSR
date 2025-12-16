import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from litsr.utils.registry import ArchRegistry
from torchvision.models import vgg19
from torch.autograd import Variable
from math import exp


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            # Mish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super().__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(
                64, channels_in * self.kernel_size * self.kernel_size, bias=False
            ),
        )
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, k_v):
        """
        :param x: feature map: B * C * H * W
        :param k_v: degradation representation: B * C
        """
        b, c, h, w = x.size()

        # branch 1
        kernel = self.kernel(k_v).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(
            F.conv2d(
                x.view(1, -1, h, w),
                kernel,
                groups=b * c,
                padding=(self.kernel_size - 1) // 2,
            )
        )
        out = out.view(b, -1, h, w)

        # branch 2
        # out = out + self.ca(x, k_v)

        return out + self.conv(x)


class DABlock(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            # Mish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        )
        self.da_conv = DA_conv(dim, dim_out, 3)

    def forward(self, x, k_v):
        out = self.block(x)
        out = self.da_conv(out, k_v)
        return out



class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = DABlock(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, k_v):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h, k_v)
        return h + self.res_conv(x)


##########################  通道注意力和空间注意力  ############################
class CLAM(nn.Module):
    def __init__(self, in_planes, ratio=16, pool_mode='Avg|Max'):
        super(CLAM, self).__init__()
        self.pool_mode = pool_mode
        if pool_mode.find('Avg') != -1:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool_mode.find('Max') != -1:
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        # self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        if self.pool_mode == 'Avg':
            out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        elif self.pool_mode == 'Max':
            out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        elif self.pool_mode == 'Avg|Max':
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out


class SLAM(nn.Module):
    def __init__(self, kernel_size=7, pool_mode='Avg|Max'):
        super(SLAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.pool_mode = pool_mode
        if pool_mode == 'Avg|Max':
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.pool_mode == 'Avg':
            out = torch.mean(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Max':
            out, _ = torch.max(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Avg|Max':
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out)) * x
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        self.conv = nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=True)
        if with_attn:
            # self.attn = SelfAttention(dim_out, norm_groups=norm_groups)
            self.ca = CLAM(dim_out, pool_mode='Avg|Max')
            self.sa = SLAM(kernel_size=7, pool_mode='Avg|Max')

    def forward(self, x, time_emb, k_v):
        x = self.res_block(x, time_emb, k_v)
        if (self.with_attn):
            # x = self.attn(x)
            x = self.ca(x)
            x = self.sa(x)
            # x = self.conv(x)
        return x


@ArchRegistry.register()
class DiffSR_UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                # Mish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            # use_attn = (now_res in attn_res)
            use_attn = False  # True
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            # use_attn = (now_res in attn_res)
            use_attn = False  # True
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False), nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, time, k_v=None):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        k_v = self.compress(k_v)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, k_v)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, k_v)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t, k_v)
            else:
                x = layer(x)

        return self.final_conv(x)