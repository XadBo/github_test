import torch
import torch.nn as nn
import math

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

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

# class ASFF_Bifpn_2(nn.Module):
#     def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
#         super(ASFF_Bifpn_2, self).__init__()
#
#         self.inter_dim = inter_dim
#         compress_c = 8
#
#         self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
#
#         self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
#         self.upsample = Upsample(channel[1], channel[0])
#         self.downsample = Downsample(channel[0], channel[1])
#         self.level = level
#         # self.CA_2 = CA_Block_2(inter_dim*2, level)
#         self.bi_add = BiFPN_Add2(self.inter_dim, self.inter_dim)
#         self.conv_sample = Conv(self.inter_dim*2, self.inter_dim, 3, 1)
#
#
#     def forward(self, x):
#         input1, input2 = x
#         if self.level == 0:
#             input2 = self.upsample(input2)
#         elif self.level == 1:
#             input1 = self.downsample(input1)
#
#         # feats = torch.cat((input1, input2), dim=1)
#
#         input_all = [input1, input2]
#         fused_out_reduced = self.bi_add(input_all)
#
#         # fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
#         #                     input2 * levels_weight[:, 1:2, :, :]
#
#         # out = self.conv(fused_out_reduced)
#         out = self.conv(fused_out_reduced)
#
#         return out
#
#
# class ASFF_Bifpn_3(nn.Module):
#     def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
#         super(ASFF_Bifpn_3, self).__init__()
#
#         self.inter_dim = inter_dim
#         compress_c = 8
#
#         self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
#
#         self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
#
#         self.level = level
#         # self.CA_3 = CA_Block_3(inter_dim*3, level)
#         self.bi_add3 = BiFPN_Add3(self.inter_dim, self.inter_dim)
#         self.conv_sample = Conv(self.inter_dim * 3, self.inter_dim, 3, 1)
#         if self.level == 0:
#             self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
#             self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
#         elif self.level == 1:
#             self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
#             self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
#         elif self.level == 2:
#             self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
#             self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)
#
#     def forward(self, x):
#         input1, input2, input3 = x
#         if self.level == 0:
#             input2 = self.upsample2x(input2)
#             input3 = self.upsample4x(input3)
#         elif self.level == 1:
#             input3 = self.upsample2x1(input3)
#             input1 = self.downsample2x1(input1)
#         elif self.level == 2:
#             input1 = self.downsample4x(input1)
#             input2 = self.downsample2x(input2)
#         # level_1_weight_v = self.weight_level_1(input1)
#         # level_2_weight_v = self.weight_level_2(input2)
#         # level_3_weight_v = self.weight_level_3(input3)
#         #
#         # levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
#         # levels_weight = self.weight_levels(levels_weight_v)
#         # levels_weight = nn.functional.softmax(levels_weight, dim=1)
#         # feats = torch.cat((input1, input2, input3), dim=1)
#
#         input_all = [input1, input2, input3]
#         fused_out_reduced = self.bi_add3(input_all)
#         out = self.conv(fused_out_reduced)
#
#         # fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
#         #                     input2 * levels_weight[:, 1:2, :, :] + \
#         #                     input3 * levels_weight[:, 2:, :, :]
#
#         # out = self.conv(fused_out_reduced)
#
#         return out


class ASFF_BifpnCA_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
        super(ASFF_BifpnCA_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level
        # self.CA_2 = CA_Block_2(inter_dim*2, level)
        self.cord = CoordAtt(self.inter_dim*2, self.inter_dim*2)
        self.bincat2 = BiFPN_Concat2(dimension=1)
        self.spdcov = space_to_depth(dimension=1)
        self.conv_sample = Conv(self.inter_dim*2, self.inter_dim, 3, 1)


    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        # feats = torch.cat((input1, input2), dim=1)
        input_all = [input1, input2]
        feats = self.bincat2(input_all)

        fused_out_reduced = self.cord(feats)

        # fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
        #                     input2 * levels_weight[:, 1:2, :, :]

        # out = self.conv(fused_out_reduced)
        # fused_out_reduced = self.spdcov(fused_out_reduced)
        out = self.conv_sample(fused_out_reduced)

        return out


class ASFF_BifpnCA_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(ASFF_BifpnCA_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level
        # self.CA_3 = CA_Block_3(inter_dim*3, level)
        self.cord = CoordAtt(self.inter_dim * 3, self.inter_dim * 3)
        self.bincat3 = BiFPN_Concat3(dimension=1)
        self.spdcov = space_to_depth(dimension=1)
        self.conv_sample = Conv(self.inter_dim * 3, self.inter_dim, 3, 1)
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
        # level_1_weight_v = self.weight_level_1(input1)
        # level_2_weight_v = self.weight_level_2(input2)
        # level_3_weight_v = self.weight_level_3(input3)
        #
        # levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        # levels_weight = self.weight_levels(levels_weight_v)
        # levels_weight = nn.functional.softmax(levels_weight, dim=1)
        input_all = [input1, input2, input3]
        feats = self.bincat3(input_all)
        # feats = torch.cat((input1, input2, input3), dim=1)
        fused_out_reduced = self.cord(feats)
        # fused_out_reduced = self.spdcov(fused_out_reduced)
        out = self.conv_sample(fused_out_reduced)

        # fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
        #                     input2 * levels_weight[:, 1:2, :, :] + \
        #                     input3 * levels_weight[:, 2:, :, :]

        # out = self.conv(fused_out_reduced)

        return out
