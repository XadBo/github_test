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
        out = (x1_group * weights.sigmoid()).reshape(n, c, h, w)
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
        self.weight_levels2 = nn.Conv2d(self.mip * 2, 2, kernel_size=1, stride=1, padding=0)

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
        # levels_weight_v = torch.cat((x1, x2), 1)
        # levels_weight = self.weight_levels(levels_weight_v).reshape(n, self.mip * 2, h, w)
        # levels_weight = self.weight_levels2(levels_weight)
        # levels_weight = nn.functional.softmax(levels_weight, dim=1)
        #
        #
        # out = x * levels_weight[:, 0:1, :, :] + \
        #       out * levels_weight[:, 1:2, :, :]

        # a new
        levels_weight_v = torch.add(x1, x2)
        x_all = self.softmax(self.agp(levels_weight_v).reshape(n * self.mip, -1, 1).permute(0, 2, 1))
        x_all2 = levels_weight_v.reshape(n * self.mip, c // self.mip, -1)
        weights = torch.matmul(x_all, x_all2).reshape(n * self.mip, 1, h, w)
        out = (x1_group * weights.sigmoid()).reshape(n, c, h, w)

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


class SKAttention2(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3], reduction=2, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                    nn.BatchNorm2d(channel),
                    nn.ReLU()
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x1, x2):
        bs, c, _, _ = x1.size()
        conv_outs = []
        conv_outs.append(x1)
        conv_outs.append(x2)
        # ### split
        # for conv in self.convs:
        #     conv_outs.append(conv(x))
        feats = torch.stack((x1, x2), 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1
        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class SKAttention3(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5], reduction=2, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                    nn.BatchNorm2d(channel),
                    nn.ReLU()
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x1, x2, x3):
        bs, c, _, _ = x1.size()
        conv_outs = []
        conv_outs.append(x1)
        conv_outs.append(x2)
        conv_outs.append(x3)
        # ### split
        # for conv in self.convs:
        #     conv_outs.append(conv(x))
        feats = torch.stack((x1, x2, x3), 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1
        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b*self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2,dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight*x_channel+self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0*self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1) # bs*G,c//(2*G),h,w
        x_spatial = self.sweight*x_spatial+self.sbias # bs*G,c//(2*G),h,w
        x_spatial = x_1*self.sigmoid(x_spatial) # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel,x_spatial],dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = Conv(2, 1, kernel_size, s=1, p=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


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

class ASFF_CAimprove_1(nn.Module):
    def __init__(self, inter_dim=512, channel=[64]):
        super(ASFF_CAimprove_1, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.weight_levels_im2 = nn.Conv2d(self.inter_dim * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)


        # self.CA_2 = CA_Block_2(inter_dim*2, level)
        self.cord = CoordAtt(self.inter_dim*2, self.inter_dim*2)
        self.ca = CoordAtt(self.inter_dim, self.inter_dim)
        self.se_weight = SEWeightModule(self.inter_dim)
        self.bincat2 = BiFPN_Concat2(dimension=1)
        self.conv_2 = Conv(self.inter_dim * 2, self.inter_dim * 2, 3, 1)
        self.cot_2 = CoTBottleneck(self.inter_dim * 2, self.inter_dim * 2, shortcut=True, g=1, e=1.0)
        self.conv_sample = Conv(self.inter_dim*2, self.inter_dim, 3, 1)
        self.sk2 = SKAttention2(self.inter_dim)
        self.sa = ShuffleAttention(self.inter_dim*2)


    def forward(self, x):
        input1 = x
        # feats = torch.cat((input1, input2), dim=1)
        # input1 = self.weight_level_1(input1)
        # input2 = self.weight_level_2(input2)


        feats_conv = self.conv(input1)
        # feats_conv = self.ca(feats_conv)
        fused_out_reduced = self.ca(feats_conv)
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
        out = self.conv(fused_out_reduced)
        # out = self.conv_sample(fused_out_reduced)

        return out


class ASFF_newCA_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
        super(ASFF_newCA_2, self).__init__()

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
        self.new_ca2 = CoordAtt_af(self.inter_dim * 2, self.inter_dim * 2)
        self.se_weight = SEWeightModule(self.inter_dim)
        self.bincat2 = BiFPN_Concat2(dimension=1)
        self.conv_2 = Conv(self.inter_dim * 2, self.inter_dim * 2, 3, 1)
        self.cot_2 = CoTBottleneck(self.inter_dim * 2, self.inter_dim * 2, shortcut=True, g=1, e=1.0)
        self.conv_sample = Conv(self.inter_dim*2, self.inter_dim, 3, 1)
        self.sk2 = SKAttention2(self.inter_dim)
        self.sa = ShuffleAttention(self.inter_dim*2)
        self.gate = AttentionGate()


    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        # feats = torch.cat((input1, input2), dim=1)
        # input1 = self.weight_level_1(input1)
        # input2 = self.weight_level_2(input2)

        input1 = self.gate(input1)
        input2 = self.gate(input2)

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


class ASFF_newCA_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(ASFF_newCA_3, self).__init__()

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
        self.new_ca3 = CoordAtt_af(self.inter_dim * 3, self.inter_dim * 3)
        self.se_weight3 = SEWeightModule(self.inter_dim)
        self.conv_sample = Conv(self.inter_dim * 3, self.inter_dim, 3, 1)
        self.conv_3 = Conv(self.inter_dim * 3, self.inter_dim*3, 3, 1)
        self.cot_3 = CoTBottleneck(self.inter_dim*3, self.inter_dim*3, shortcut=True, g=1, e=1.0)
        self.sk3 = SKAttention3(self.inter_dim)
        self.sa = ShuffleAttention(self.inter_dim * 3)
        self.gate = AttentionGate()
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

        input1 = self.gate(input1)
        input2 = self.gate(input2)
        input3 = self.gate(input3)

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
