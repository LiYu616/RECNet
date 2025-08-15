import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import MyLibForSteerCNN as ML
# import scipy.io as sio
import math
from PIL import Image

class Fconv_PCA(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True,
                 Smooth=True, iniScale=1.0, Bscale=1.0):

        super(Fconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis * Bscale)  # .cuda())
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=bias)

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
        else:
            _filter = self.filter
            _bias = self.bias
        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + _bias

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)
            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
            self.register_buffer("filter", _filter)
            self.register_buffer("bias", _bias)

        return super(Fconv_PCA, self).train(mode)


class Fconv_PCA_out(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Fconv_PCA_out, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())

        iniw = Getini_reg(Basis.size(3), inNum, outNum, 1, weight) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=bias)

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + _bias

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)

            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
        return super(Fconv_PCA_out, self).train(mode)


class Fconv_1X1(nn.Module):

    def __init__(self, inNum, outNum, tranNum=8, ifIni=0, bias=True, Smooth=True, iniScale=1.0):

        super(Fconv_1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(1, inNum, outNum, self.expand) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.c = torch.zeros(1, outNum, 1, 1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1, tranNum, 1, 1, 1, 1])

        Num = tranNum // expand
        tempWList = [
            torch.cat([tempW[:, i * Num:(i + 1) * Num, :, -i:, ...], tempW[:, i * Num:(i + 1) * Num, :, :-i, ...]],
                      dim=3) for i in range(expand)]
        tempW = torch.cat(tempWList, dim=1)

        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, 1, 1])

        bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])  # .cuda()

        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + bias


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, tranNum=8, inP=None,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, Smooth=True, iniScale=1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP=inP, padding=(kernel_size - 1) // 2, bias=bias,
                     Smooth=Smooth, iniScale=iniScale))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


def Getini_reg(nNum, inNum, outNum, expand, weight=1):
    A = (np.random.rand(outNum, inNum, expand, nNum) - 0.5) * 2 * 2.4495 / np.sqrt((inNum) * nNum) * np.expand_dims(
        np.expand_dims(np.expand_dims(weight, axis=0), axis=0), axis=0)
    return torch.FloatTensor(A)


def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth=True):
    if inP == None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(Mask, 2)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    #    theta = torch.FloatTensor(theta)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    #    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)

    BasisC = np.reshape(BasisC, [sizeP * sizeP * tranNum, inP * inP])
    BasisS = np.reshape(BasisS, [sizeP * sizeP * tranNum, inP * inP])

    BasisR = np.concatenate((BasisC, BasisS), axis=1)

    U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))

    Rank = np.sum(S > 0.0001)
    BasisR = np.matmul(np.matmul(BasisR, U[:, :Rank]), np.diag(1 / np.sqrt(S[:Rank] + 0.0000000001)))
    BasisR = np.reshape(BasisR, [sizeP, sizeP, tranNum, Rank])

    temp = np.reshape(BasisR, [sizeP * sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis=0) ** 2, axis=0) + np.std(np.sum(temp ** 2 * sizeP * sizeP, axis=0),
                                                              axis=0)) / np.mean(
        np.sum(temp, axis=0) ** 2 + np.sum(temp ** 2 * sizeP * sizeP, axis=0), axis=0)
    Trod = 1
    Ind = var < Trod
    Rank = np.sum(Ind)
    Weight = 1 / np.maximum(var, 0.04) / 25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight, 0), 0), 0) * BasisR

    return torch.FloatTensor(BasisR), Rank, Weight


def MaskC(SizeP):
    p = (SizeP - 1) / 2
    x = np.arange(-p, p + 1) / p
    X, Y = np.meshgrid(x, x)
    C = X ** 2 + Y ** 2

    Mask = np.ones([SizeP, SizeP])
    #        Mask[C>(1+1/(4*p))**2]=0
    Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)

    return X, Y, Mask


class PointwiseAvgPoolAntialiased(nn.Module):

    def __init__(self, sizeF, stride, padding=None):
        super(PointwiseAvgPoolAntialiased, self).__init__()
        sigma = (sizeF - 1) / 2 / 3
        self.kernel_size = (sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if padding is None:
            padding = int((sizeF - 1) // 2)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = torch.arange(sizeF).repeat(sizeF).view(sizeF, sizeF)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())
        _filter = torch.exp(r / (2 * variance))
        _filter /= torch.sum(_filter)
        _filter = _filter.view(1, 1, sizeF, sizeF)
        self.filter = nn.Parameter(_filter, requires_grad=False)
        # self.register_buffer("filter", _filter)

    def forward(self, input):
        _filter = self.filter.repeat((input.shape[1], 1, 1, 1))
        output = F.conv2d(input, _filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return output


class PointwiseSobelAntialiased(nn.Module):
    def __init__(self, stride, padding=None):
        super(PointwiseSobelAntialiased, self).__init__()
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Register Sobel filters as non-trainable parameters
        self.sobel_x = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y, requires_grad=False)

        self.kernel_size = (3, 3)  # Sobel kernel is fixed at 3x3
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        if padding is None:
            padding = 1  # Default padding for Sobel to maintain dimensions

        self.padding = (padding, padding) if isinstance(padding, int) else padding

    def forward(self, input):
        # Apply Sobel filters in both directions
        sobel_x_filter = self.sobel_x.repeat((input.shape[1], 1, 1, 1))
        sobel_y_filter = self.sobel_y.repeat((input.shape[1], 1, 1, 1))

        grad_x = F.conv2d(input, sobel_x_filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        grad_y = F.conv2d(input, sobel_y_filter, stride=self.stride, padding=self.padding, groups=input.shape[1])

        # Compute gradient magnitude
        output = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return output


class F_BN(nn.Module):
    def __init__(self, channels, tranNum=8):
        super(F_BN, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.tranNum = tranNum

    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1) / self.tranNum), self.tranNum * X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum * X.size(1), int(X.size(2) / self.tranNum), X.size(3)])


class F_Dropout(nn.Module):
    def __init__(self, zero_prob=0.5, tranNum=8):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout2d(zero_prob)

    def forward(self, X):
        X = self.Dropout(X.reshape([X.size(0), int(X.size(1) / self.tranNum), self.tranNum * X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum * X.size(1), int(X.size(2) / self.tranNum), X.size(3)])


def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s - 1) / 2
    t = (c - margin / 100. * c) ** 2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r) / sig ** 2)
            else:
                mask[..., x, y] = 1.
    return mask


class MaskModule(nn.Module):

    def __init__(self, S: int, margin: float = 0.):
        super(MaskModule, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask(S, margin=margin), requires_grad=False)

    def forward(self, input):
        assert input.shape[2:] == self.mask.shape[2:]

        out = input * self.mask
        return out


class GroupPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum

    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)])
        output = torch.max(output, 2).values
        return output


class GroupMeanPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupMeanPooling, self).__init__()
        self.tranNum = tranNum

    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)])
        output = torch.mean(output, 2)
        return output


class Fconv(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True):
        super(Fconv, self).__init__()

        # 默认使用 CUDA 设备，如果可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP

        BasisC, BasisS = GetBasis(sizeP, tranNum, inP)
        self.register_buffer("Basis", torch.cat([BasisC, BasisS], 3).to(self.device))  # 移动到设备上

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum

        iniw = Getini(inP, inNum, outNum, self.expand)
        self.weights = nn.Parameter(iniw.to(self.device), requires_grad=True)  # 移动到设备上

        if padding == None:
            self.padding = 0
        else:
            self.padding = padding

        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1).to(self.device), requires_grad=True)  # 移动到设备上
        else:
            self.c = torch.zeros(1, outNum, 1, 1).to(self.device)

        self.register_buffer("filter", torch.zeros(outNum * tranNum, inNum * self.expand, sizeP, sizeP).to(self.device))

    def forward(self, input):
        input = input.to(self.device)  # 确保输入数据在正确的设备上

        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand

        tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

        for i in range(expand):
            ind = np.hstack((np.arange(expand - i, expand), np.arange(expand - i)))
            tempW[:, i, :, :, ...] = tempW[:, i, :, ind, ...]

        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])

        bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])

        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + bias


class FCNN_reg(nn.Module):

    def __init__(self, sizeP, inNum, outNum, Basisin, tranNum=8, inP=None, padding=None, ifIni=0, bias=True):

        super(FCNN_reg, self).__init__()
        #        self.sampled_basis = Basisin.cuda()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.register_buffer("Basis", Basisin.cuda())

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3)) * 0.03
        self.weights = nn.Parameter(iniw, requires_grad=True)
        self.padding = padding

    def forward(self, input):

        #        _filter = self._filter
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

        for i in range(expand):
            ind = np.hstack((np.arange(expand - i, expand), np.arange(expand - i)))
            tempW[:, i, :, :, ...] = tempW[:, i, :, ind, ...]
        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])

        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)

        return output


def Getini(sizeP, inNum, outNum, expand):
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0, 0), 0), 4), 0)
    y = Y0[:, 1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y, 0), 0), 3), 0)

    orlW = np.zeros([outNum, inNum, expand, sizeP, sizeP, 1, 1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(
                    Image.fromarray(((np.random.randn(3, 3)) * 2.4495 / np.sqrt((inNum) * sizeP * sizeP))).resize(
                        (sizeP, sizeP)))
                orlW[i, j, k, :, :, 0, 0] = temp

    v = np.pi / sizeP * (sizeP - 1)
    k = np.reshape((np.arange(sizeP)), [1, 1, 1, 1, 1, sizeP, 1])
    l = np.reshape((np.arange(sizeP)), [1, 1, 1, 1, 1, sizeP])

    tempA = np.sum(np.cos(k * v * X0) * orlW, 4) / sizeP
    tempB = -np.sum(np.sin(k * v * X0) * orlW, 4) / sizeP
    A = np.sum(np.cos(l * v * y) * tempA + np.sin(l * v * y) * tempB, 3) / sizeP
    B = np.sum(np.cos(l * v * y) * tempB - np.sin(l * v * y) * tempA, 3) / sizeP
    A = np.reshape(A, [outNum, inNum, expand, sizeP * sizeP])
    B = np.reshape(B, [outNum, inNum, expand, sizeP * sizeP])
    iniW = np.concatenate((A, B), axis=3)
    return torch.FloatTensor(iniW).cuda(0)


def GetBasis(sizeP, tranNum=8, inP=None):
    if inP == None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(Mask, 2)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    #    theta = torch.FloatTensor(theta)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    #    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    return torch.FloatTensor(BasisC).cuda(0), torch.FloatTensor(BasisS).cuda(0)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 64),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)

            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)

            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FConvEncoder(nn.Module):
    def __init__(self, tranNum=8):
        super().__init__()

        zero_prob = 0.25
        # 定义 F-Conv 的层次结构，与 ResNet18 输出匹配
        self.stage1 = nn.Sequential(
            Fconv(7, inNum=3, outNum=8, tranNum=tranNum, inP=12, padding=3, ifIni=1, bias=False),
            F_BN(8, tranNum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1)  # 确保输出尺寸缩小一半
        )
        self.stage2 = nn.Sequential(
            Fconv(7, inNum=8, outNum=16, tranNum=tranNum, inP=12, padding=3, ifIni=0, bias=False),
            F_BN(16, tranNum),
            nn.ReLU(inplace=True),
            F_Dropout(zero_prob, tranNum),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 添加降采样
        )
        self.stage3 = nn.Sequential(
            Fconv(7, inNum=16, outNum=32, tranNum=tranNum, inP=12, padding=3, ifIni=0, bias=False),
            F_BN(32, tranNum),
            nn.ReLU(inplace=True),
            F_Dropout(zero_prob, tranNum),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 添加降采样
        )
        self.stage4 = nn.Sequential(
            Fconv(5, inNum=32, outNum=64, tranNum=tranNum, inP=12, padding=2, ifIni=0, bias=False),
            F_BN(64, tranNum),
            nn.ReLU(inplace=True),
            F_Dropout(zero_prob, tranNum),
            PointwiseAvgPoolAntialiased(5, 2, padding=2),
            GroupPooling(tranNum)  # 添加降采样
        )

    def forward(self, x):
        res1 = self.stage1(x)  # 输出与 ResNet18 的第一层一致
        res2 = self.stage2(res1)  # 第二层
        res3 = self.stage3(res2)  # 第三层
        res4 = self.stage4(res3)  # 第四层
        return res1, res2, res3, res4

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',  # 修改了 backbone_name 的默认值
                 window_size=8,
                 num_classes=6,
                 tranNum=8  # 添加 FConvEncoder 需要的参数
                 ):
        super().__init__()

        # 根据 backbone_name 选择不同的 encoder
        self.fconv_encoder = FConvEncoder(tranNum)
        encoder_channels = [64, 128, 256, 64]  # FConv 对应的输出通道数

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.fconv_encoder(x)
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetFormer().to(device)  # 将模型移动到 GPU 或 CPU
x = torch.randn(1, 3, 1024, 1024).to(device)  # 将输入张量移动到相同设备
output = model(x)