import torch
import torch.nn as nn
import math
from torch.nn import init
from torch.nn.parameter import Parameter
## 基于afpn的基础，先进行cat，在conv，split，然后在最后两层乘以权重时，在乘两层自身的权重，再乘ca得到的权重

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

        # carafe
        # from mmcv.ops import CARAFEPack
        # self.upsample = nn.Sequential(
        #     BasicConv(in_channels, out_channels, 1),
        #     CARAFEPack(out_channels, scale_factor=scale_factor)
        # )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class CoordAtt_im(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_im, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.gn = nn.GroupNorm(inp // self.mip, inp // self.mip)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(inp // self.mip, inp // self.mip, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h

        x1_group = out.reshape(n * self.mip, -1, h, w)
        x1 = self.gn(x1_group)
        x2_group = x.reshape(n * self.mip, -1, h, w)
        x2 = self.conv3x3(x2_group)
        x11 = self.softmax(self.agp(x1).reshape(n * self.mip, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(n * self.mip, c // self.mip, -1)  # b, c, hw
        x21 = self.softmax(self.agp(x2).reshape(n * self.mip, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(n * self.mip, c // self.mip, -1)  # b, c, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(n * self.mip, 1, h, w)
        out = (x2_group * weights.sigmoid()).reshape(n, c, h, w)
        return out


class CoordAtt_af(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_af, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.gn = nn.GroupNorm(inp // self.mip, inp // self.mip)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(inp // self.mip, inp // self.mip, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(-1)
        self.weight_levels = nn.Conv2d((inp // self.mip) * 2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h

        x1_group = out.reshape(n * self.mip, -1, h, w)
        x1 = self.gn(x1_group)
        x2_group = x.reshape(n * self.mip, -1, h, w)
        x2 = self.conv3x3(x2_group)
        levels_weight_v = torch.cat((x1, x2), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = nn.functional.softmax(levels_weight, dim=1)

        out = x * levels_weight[:, 0:1, :, :] + \
              out * levels_weight[:, 1:2, :, :]

        return out


class CoTBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(CoTBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = CoT(c_, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CoT(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = nn.functional.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2



class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))


# 三个分支add操作
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))



# 两个分支concat操作
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

class BiFPN_Concat4(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat4, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2], weight[3] * x[3]]
        return torch.cat(x, self.d)


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight



class Bifusion_1(nn.Module):
    def __init__(self, in_channels, channels=[128, 64, 32]):
        super(Bifusion_1, self).__init__()
        out_channels = in_channels
        self.upsample = Upsample(out_channels, out_channels)
        self.downsample = Downsample(out_channels, out_channels)
        self.cv1 = Conv(channels[0], out_channels, 1, 1)
        self.cv2 = Conv(channels[2], out_channels, 1, 1)
        self.cv3 = Conv(out_channels * 3, out_channels, 1, 1)

    def forward(self, x):
        input1, input2, input3 = x
        input1 = self.upsample(self.cv1(input1))
        input3 = self.downsample(self.cv2(input3))


        out = self.cv3(torch.cat((input1, input2, input3), dim=1))


        return out


class Bifusion_2(nn.Module):
    def __init__(self, in_channels, channels=[64, 64, 32]):
        super(Bifusion_2, self).__init__()
        out_channels = in_channels


        self.upsample = Upsample(out_channels, out_channels, scale_factor=2)
        self.downsample = Downsample(out_channels, out_channels, scale_factor=2)
        self.cv1 = Conv(channels[0], out_channels, 1, 1)
        self.cv2 = Conv(channels[1], out_channels, 1, 1)
        self.cv3 = Conv(out_channels * 3, out_channels, 1, 1)

    def forward(self, x):
        input1, input2, input3 = x
        input1 = self.upsample(self.cv1(input1))
        input2 = self.upsample(self.cv2(input2))



        out = self.cv3(torch.cat((input1, input2, input3), dim=1))


        return out




class AUFPN_fusion_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
        super(AUFPN_fusion_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.weight_levels_im2 = nn.Conv2d(self.inter_dim * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level
        # self.CA_2 = CA_Block_2(inter_dim*2, level)
        self.cord = CoordAtt(self.inter_dim*2, self.inter_dim*2)
        self.ca = CoordAtt(self.inter_dim*2, self.inter_dim*2)
        self.ca_im2 = CoordAtt_im(self.inter_dim * 2, self.inter_dim * 2)
        self.se_weight = SEWeightModule(self.inter_dim)
        self.bincat2 = BiFPN_Concat2(dimension=1)
        self.conv_2 = Conv(self.inter_dim * 2, self.inter_dim * 2, 3, 1)
        self.cot_2 = CoTBottleneck(self.inter_dim * 2, self.inter_dim * 2, shortcut=True, g=1, e=1.0)
        self.conv_sample = Conv(self.inter_dim*2, self.inter_dim, 3, 1)
        # self.sk2 = SKAttention2(self.inter_dim)
        # self.sa = ShuffleAttention(self.inter_dim*2)


    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        # feats = torch.cat((input1, input2), dim=1)
        # input1 = self.weight_level_1(input1)
        # input2 = self.weight_level_2(input2)

        input_all = [input1, input2]
        feats = self.bincat2(input_all)  # cat操作
        # feats_conv = self.conv_2(feats)
        # feats_conv = self.ca(feats_conv)
        # fused_out_reduced = self.ca(feats)
        fused_out_reduced = self.ca_im2(feats)
        # x_1, x_2 = torch.split(feats_conv, self.inter_dim, dim=1)
        # x1_out = self.ca(x_1)
        # x2_out = self.ca(x_2)
        #
        # x1_weight = self.weight_level_1(x_1)
        # x2_weight = self.weight_level_2(x_2)
        # levels_weight = torch.cat((x1_weight, x2_weight), dim=1)
        # levels_weight = self.weight_levels(levels_weight)
        # levels_weight = nn.functional.softmax(levels_weight, dim=1)

        # fused_out_reduced = self.sk2(x_1, x_2)

        # fused_out_reduced = x_1 * levels_weight[:, 0:1, :, :] + \
        #                     x_2 * levels_weight[:, 1:2, :, :]
        #
        #
        # out = self.conv(fused_out_reduced)
        out = self.conv_sample(fused_out_reduced)

        return out


class AUFPN_fusion_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(AUFPN_fusion_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.weight_levels_im3 = nn.Conv2d(self.inter_dim * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level
        # self.CA_3 = CA_Block_3(inter_dim*3, level)
        self.cord = CoordAtt(self.inter_dim * 3, self.inter_dim * 3)
        self.bincat3 = BiFPN_Concat3(dimension=1)
        self.ca3 = CoordAtt(self.inter_dim*3, self.inter_dim*3)
        self.ca_im3 = CoordAtt_im(self.inter_dim * 3, self.inter_dim * 3)
        self.se_weight3 = SEWeightModule(self.inter_dim)
        self.conv_sample = Conv(self.inter_dim * 3, self.inter_dim, 3, 1)
        self.conv_3 = Conv(self.inter_dim * 3, self.inter_dim*3, 3, 1)
        self.cot_3 = CoTBottleneck(self.inter_dim*3, self.inter_dim*3, shortcut=True, g=1, e=1.0)
        # self.sk3 = SKAttention3(self.inter_dim)
        # self.sa = ShuffleAttention(self.inter_dim * 3)
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
        # input1 = self.weight_level_1(input1)
        # input2 = self.weight_level_2(input2)
        # input3 = self.weight_level_3(input3)
        #
        # levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        # levels_weight = self.weight_levels(levels_weight_v)
        # levels_weight = nn.functional.softmax(levels_weight, dim=1)
        input_all = [input1, input2, input3]
        feats = self.bincat3(input_all)
        # feats = torch.cat((input1, input2, input3), dim=1)
        # fused_out_reduced = self.cord(feats)

        # feats_conv = self.conv_3(feats)
        # x_1, x_2, x_3 = torch.split(feats_conv, self.inter_dim, dim=1)
        # fused_out_reduced = self.sk3(x_1, x_2, x_3)
        # feats_conv = self.ca3(feats_conv)
        # fused_out_reduced = self.ca3(feats_conv)
        fused_out_reduced = self.ca_im3(feats)
        # out = self.conv_sample(fused_out_reduced)
        # x_1, x_2, x_3 = torch.split(feats_conv, self.inter_dim, dim=1)
        # x1_out = self.ca3(x_1)
        # x2_out = self.ca3(x_2)
        # x3_out = self.ca3(x_3)
        #
        # x1_weight = self.weight_level_1(x_1)
        # x2_weight = self.weight_level_2(x_2)
        # x3_weight = self.weight_level_3(x_3)
        #
        # levels_weight = torch.cat((x1_weight, x2_weight, x3_weight), dim=1)
        # levels_weight = self.weight_levels(levels_weight)
        # levels_weight = nn.functional.softmax(levels_weight, dim=1)
        out = self.conv_sample(fused_out_reduced)
        #
        # fused_out_reduced = x_1 * levels_weight[:, 0:1, :, :] + \
        #                     x_2 * levels_weight[:, 1:2, :, :] + \
        #                     x_3 * levels_weight[:, 2:, :, :]
        #
        #
        # out = self.conv(fused_out_reduced)

        return out



class AUFPN_fusion_4(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[32, 64, 128, 256]):
        super(AUFPN_fusion_4, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_4 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.bincat4 = BiFPN_Concat4(dimension=1)
        self.cord4 = CoordAtt(self.inter_dim * 4, self.inter_dim * 4)
        self.ca_im4 = CoordAtt_im(self.inter_dim * 4, self.inter_dim * 4)
        self.conv_sample = Conv(self.inter_dim * 4, self.inter_dim, 3, 1)

        self.level = level
        if self.level == 0:
            self.upsample8x = Upsample(channel[3], channel[0], scale_factor=8)
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample4x1 = Upsample(channel[3], channel[1], scale_factor=4)
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)
            self.upsample2x2 = Upsample(channel[3], channel[2], scale_factor=2)
        elif self.level == 3:
            self.downsample2x = Downsample(channel[2], channel[3], scale_factor=2)
            self.downsample4x = Downsample(channel[1], channel[3], scale_factor=4)
            self.downsample8x = Downsample(channel[0], channel[3], scale_factor=8)

    def forward(self, x):
        input1, input2, input3, input4 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            input4 = self.upsample8x(input4)
        elif self.level == 1:
            input4 = self.upsample4x1(input4)
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
            input4 = self.upsample2x2(input4)
        elif self.level == 3:
            input3 = self.downsample2x(input3)
            input2 = self.downsample4x(input2)
            input1 = self.downsample8x(input1)
        # level_1_weight_v = self.weight_level_1(input1)
        # level_2_weight_v = self.weight_level_2(input2)
        # level_3_weight_v = self.weight_level_3(input3)
        # level_4_weight_v = self.weight_level_4(input4)

        input_all = [input1, input2, input3, input4]
        feats = self.bincat4(input_all)

        fused_out_reduced = self.ca_im4(feats)
        out = self.conv_sample(fused_out_reduced)

        # levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1)
        # levels_weight_v = self.bincat3([level_1_weight_v, level_2_weight_v, level_3_weight_v])
        # levels_weight = self.weight_levels(levels_weight_v)
        # levels_weight = nn.functional.softmax(levels_weight, dim=1)
        #
        # fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
        #                     input2 * levels_weight[:, 1:2, :, :] + \
        #                     input3 * levels_weight[:, 2:3, :, :] + \
        #                     input4 * levels_weight[:, 3:, :, :]
        #
        # out = self.conv(fused_out_reduced)

        return out