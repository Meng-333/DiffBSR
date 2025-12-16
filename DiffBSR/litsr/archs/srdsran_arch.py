import torch
import math
from litsr.utils.registry import ArchRegistry
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import Parameter

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

class SGAM(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(SGAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class CGAM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, light=False):
        super(CGAM, self).__init__()
        self.chanel_in = in_dim
        self.light = light
        if light:
            self.conv1x1 = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, stride=1, bias=True)
            self.relu = nn.ReLU(True)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        if self.light:
            x_avg = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_max = torch.nn.functional.adaptive_max_pool2d(x, 1)
            x_pool = self.relu(self.conv1x1(torch.cat([x_avg, x_max], 1)))
            proj_query = x_pool.view(m_batchsize, C, -1)
            proj_key = x_pool.view(m_batchsize, C, -1).permute(0, 2, 1)
            energy = torch.bmm(proj_query, proj_key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)
            # out = attention*x
        else:
            proj_query = x.view(m_batchsize, C, -1)
            proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
            energy = torch.bmm(proj_query, proj_key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
            attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class RAB(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=True, dilation=1, act_type='lrelu',
                 la_mode='CA-SA', pool_mode='Avg|Max', addconv=True):
        super(RAB, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = torch.nn.Conv2d(inplanes, 4 * planes, kernel_size, stride, padding, bias=bias, dilation=dilation)
        self.conv2 = torch.nn.Conv2d(4 * planes, planes, kernel_size, stride, padding, bias=bias, dilation=dilation)
        self.la_mode = la_mode
        self.addconv = addconv
        if self.la_mode.find('CA') != -1:
            self.ca = CLAM(in_planes=planes, pool_mode=pool_mode)
        if self.la_mode.find('SA') != -1:
            self.sa = SLAM(kernel_size=7, pool_mode=pool_mode)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(planes * 2, planes, kernel_size=1, bias=True)
        if self.la_mode.find('-') != -1 and addconv:
            self.conv = nn.Conv2d(planes, planes, kernel_size=1, bias=True)

        if act_type == 'relu':
            self.act = torch.nn.ReLU(True)
        elif act_type == 'prelu':
            self.act = torch.nn.PReLU()
        elif act_type == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif act_type == 'tanh':
            self.act = torch.nn.Tanh()
        elif act_type == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout, saout], dim=1))
        out += x
        return out

class ResGroup(nn.Module):
    def __init__(self, block, n_blocks=3, nc=64, kernel_size=3, stride=1, bias=True, padding=1,
                 act_type='lrelu', mode='CNA', rla_mode='CA-SA', bla_mode='CA-SA', pool_mode='Avg|Max', addconv=True):
        super(ResGroup, self).__init__()

        layer = []
        for i in range(n_blocks):
            layer.append(block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding,
                               act_type='lrelu', la_mode=bla_mode, pool_mode=pool_mode, addconv=addconv))
        self.RG = nn.Sequential(*layer)
        self.la_mode = rla_mode
        self.addconv = addconv
        if self.la_mode.find('CA') != -1:
            self.ca = CLAM(in_planes=nc, pool_mode=pool_mode)
        if self.la_mode.find('SA') != -1:
            self.sa = SLAM(kernel_size=7, pool_mode=pool_mode)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)
        if self.la_mode.find('-') != -1 and addconv:
            self.conv = nn.Conv2d(nc, nc, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.RG(x)
        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout, saout], dim=1))
        out += x
        return out

# multiScale block
class MSB(nn.Module):
    def __init__(self, inplanes, planes):
        super(MSB, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(inplanes, planes, 1, 1, 0), nn.Conv2d(planes, planes, 3, 1, 1))
        self.conv3 = torch.nn.Conv2d(inplanes, planes, 1, 1, 0)
        self.conv = torch.nn.Conv2d(planes * 3, planes, 1, 1, 0)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = self.conv(torch.cat([out1, out2, out3], dim=1))
        out = self.lrelu(out)
        return out

class GAB_UP(nn.Module):
    def __init__(self, ga_mode='CA-SA', addconv=True, upscale_factor=4):
        super(GAB_UP, self).__init__()
        self.ga_mode = ga_mode

        if self.ga_mode.find('CA') != -1:
            self.ca = CGAM(64)
        if self.ga_mode.find('SA') != -1:
            self.sa = SGAM(64)
        if self.ga_mode.find('-') != -1 and addconv:
            self.conv = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        if self.ga_mode.find('|') != -1:
            self.conv = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.addconv = addconv
        # Upsampling layers
        upsampling = []
        upsampling_two = [nn.Conv2d(64, 64 * 4, 3, 1, 1),
                          nn.PixelShuffle(upscale_factor=2),
                          nn.LeakyReLU(inplace=True)]
        upsampling_three = [nn.Conv2d(64, 64 * 9, 3, 1, 1),
                            nn.PixelShuffle(upscale_factor=3),
                            nn.LeakyReLU(inplace=True)]
        if (upscale_factor & (upscale_factor - 1)) == 0:  # Is scale = 2^n
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling += upsampling_two
        elif upscale_factor % 3 == 0:
            for _ in range(int(math.log(upscale_factor, 3))):
                upsampling += upsampling_three

        self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x):
        out = x
        if self.ga_mode == 'CA':
            out = self.ca(out)
        elif self.ga_mode == 'SA':
            out = self.sa(out)
        elif self.ga_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
            if self.addconv:
                out = self.conv(out)
        elif self.ga_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
            if self.addconv:
                out = self.conv(out)
        elif self.ga_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout, saout], dim=1))

        out = self.upsampling(out)
        return out

@ArchRegistry.register()
class SRDSRAN_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=12, n_basic_blocks=3,
                 rla_mode='CA-SA', bla_mode='CA-SA', ga_mode='CA-SA', pool_mode='Avg|Max', addconv=True, upscale_factor=4):
        super(SRDSRAN_Net, self).__init__()

        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        # residual groups
        layer = []
        for i in range(n_residual_blocks):
            layer.append(ResGroup(RAB, n_blocks=n_basic_blocks, nc=64, kernel_size=3, stride=1, padding=1,
                                        act_type='lrelu', mode='CNA', rla_mode=rla_mode, bla_mode=bla_mode,
                                        pool_mode=pool_mode, addconv=addconv))
        self.res_groups = nn.Sequential(*layer)

        # Global Attention Block and Upsampling
        self.GAB_UP = GAB_UP(ga_mode=ga_mode, addconv=addconv, upscale_factor=upscale_factor)

        # Multi-scale block
        self.MSB = MSB(inplanes=in_channels, planes=64)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, 3, 1, 1))

    def forward(self, x):
        msb = self.MSB(x)
        out = self.conv1(x)
        out_all = msb + out

        for res_group in self.res_groups:
            y = res_group(out)
            out_all += y
            out = y

        out_all = self.GAB_UP(out_all)
        return self.conv3(out_all)
