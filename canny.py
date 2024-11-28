#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author  : xcRt
@time    : 2024/11/23 04:00
@function: Implementation of Differentiable Canny Edge Detector
@version : V1
"""
from functools import cache
from typing import Optional

import torch
import torch.nn.functional as f
from torch import nn
from torchvision.utils import save_image


@cache
def gaussian_kernel(kernel_size: int, sigma: Optional[float] = None):
    """高斯卷积核"""
    assert kernel_size % 2 == 1
    if sigma is None or sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) / 2 - 1) + 0.8
    x = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
    gauss = torch.exp(-(x ** 2) / (2 * (sigma ** 2)))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel[None, None, :, :]
    return kernel


@cache
def sobel_kernel():
    """sobel卷积核"""
    kernel_x = torch.Tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ])
    kernel_y = torch.Tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ])
    return kernel_x.unsqueeze(0).unsqueeze(0), kernel_y.unsqueeze(0).unsqueeze(0)


@cache
def hysteresis_kernel():
    """滞后阈值卷积核，检测弱边缘邻域是否有强边缘"""
    kernel = torch.ones(3, 3)
    return kernel.unsqueeze(0).unsqueeze(0)


def _watch_grad_fn(locals_):
    """查看所有变量的grad_fn，保证所有变量都可导"""
    for name, param in locals_.items():
        if isinstance(param, torch.Tensor):
            print(name, param.grad_fn)


class CannyEdgeDetector(nn.Module):
    def __init__(
            self,
            gaussian_kernel_size: int,
            gaussian_sigma: float,
            high_threshold: float,
            low_threshold: float,
            binary: bool = False
    ):
        """
        Canny边缘检测
        :param gaussian_kernel_size: 高斯平滑卷积核大小
        :param gaussian_sigma: 高斯分布的标准差
        :param high_threshold: 强阈值
        :param low_threshold: 弱阈值
        :param binary: 是否二值化，二值化后不可导

        Example::

            >>> import torch
            >>> import torch.nn.functional as f
            >>> from PIL import Image
            >>> from torchvision.utils import save_image
            >>> from torchvision.transforms.functional import to_tensor
            >>> from canny import CannyEdgeDetector
            >>> path =  'Lena.png'
            >>> img = Image.open(path).convert('L')
            >>> img = to_tensor(img).unsqueeze(0)
            >>> canny = CannyEdgeDetector(5, 1, 0.1, 0.25)
            >>> edge = canny(img)
            image None
            img <ReflectionPad2DBackward0 object at 0x00000244EE7EFF70>
            out <ConvolutionBackward0 object at 0x00000244EE7EFF70>
            image <ConvolutionBackward0 object at 0x00000244EE7EFF70>
            padded_img <ReflectionPad2DBackward0 object at 0x00000244EE7EFF70>
            grad_x <ConvolutionBackward0 object at 0x00000244EE7EFF70>
            grad_y <ConvolutionBackward0 object at 0x00000244EE7EFF70>
            grad <SqrtBackward0 object at 0x00000244EE7EFF70>
            grad_direction <WhereBackward0 object at 0x00000244EE7EFF70>
            grad <SqrtBackward0 object at 0x00000244EE7EFF70>
            grad_direction <WhereBackward0 object at 0x00000244EE7EFF70>
            padded_grad <ReflectionPad2DBackward0 object at 0x00000244EE7EFF70>
            suppressed_grad <WhereBackward0 object at 0x00000244EE7EFF70>
            edges <WhereBackward0 object at 0x00000244EE7EFF70>
            max_value <MaxBackward1 object at 0x00000244EE7EFF70>
            min_value <MinBackward1 object at 0x00000244EE7EFF70>
            low_threshold <AddBackward0 object at 0x00000244EE7EFF70>
            high_threshold <AddBackward0 object at 0x00000244EE7EFF70>
            strong_mask None
            weak_mask None
            has_strong_neighbor None
            mask None
            filtered_edges <MulBackward0 object at 0x00000244EE7EFF70>
            >>> save_image(edge, 'edge.png')
        """
        super(CannyEdgeDetector, self).__init__()
        assert gaussian_kernel_size % 2 == 1
        assert gaussian_sigma >= 0
        assert 0 < high_threshold < 1
        assert 0 < low_threshold < 1

        self.gaussian_kernel_size = gaussian_kernel_size
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.binary = binary
        self.register_buffer('gaussian_kernel', gaussian_kernel(gaussian_kernel_size, gaussian_sigma))
        self.register_buffer('sobel_kernel_x', sobel_kernel()[0])
        self.register_buffer('sobel_kernel_y', sobel_kernel()[1])
        self.register_buffer('hysteresis_kernel', hysteresis_kernel())

    def forward(self, image):
        smoothed_image = self.gaussian_smoothing(image)
        grad, grad_direction = self.gradient_calculation(smoothed_image)
        edges = self.non_maximum_suppression(grad, grad_direction)
        edges = self.double_threshold(edges)
        return edges

    def gaussian_smoothing(self, image):
        """高斯平滑"""
        img = f.pad(image, pad=tuple([self.gaussian_kernel_size // 2] * 4), mode='reflect')
        out = f.conv2d(img, self.gaussian_kernel)
        _watch_grad_fn(locals())
        return out

    def gradient_calculation(self, image):
        """计算梯度和梯度方向"""
        padded_img = f.pad(image, pad=(1, 1, 1, 1), mode='reflect')  # 填充，保证图片大小相同

        grad_x = f.conv2d(padded_img, self.sobel_kernel_x)
        grad_y = f.conv2d(padded_img, self.sobel_kernel_y)

        # 计算梯度
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        # 计算梯度方向
        grad_direction = torch.atan2(grad_y, grad_x)
        grad_direction = torch.rad2deg(grad_direction)
        grad_direction = torch.where(grad_direction < 0, grad_direction + 180, grad_direction)
        _watch_grad_fn(locals())
        return grad, grad_direction

    def non_maximum_suppression(self, grad, grad_direction):
        """非极大值抑制"""
        padded_grad = f.pad(grad, (1, 1, 1, 1), mode='reflect')  # 填充周围一圈，让中间的值可以和周围8个值比较而不会超出索引范围

        suppressed_grad = torch.zeros_like(grad)

        # 保留符合梯度方向并且大于梯度方向的梯度值
        suppressed_grad = torch.where(
            ((grad_direction < 22.5) | (grad_direction >= 157.5)) &
            (grad > padded_grad[:, :, 1:-1, 2:]) &
            (grad > padded_grad[:, :, 1:-1, :-2]),
            grad,
            suppressed_grad
        )
        suppressed_grad = torch.where(
            (22.5 <= grad_direction) & (grad_direction < 67.5) &
            (grad > padded_grad[:, :, 2:, :-2]) &
            (grad > padded_grad[:, :, :-2, 2:]),
            grad,
            suppressed_grad
        )
        suppressed_grad = torch.where(
            (67.5 <= grad_direction) & (grad_direction < 112.5) &
            (grad > padded_grad[:, :, :-2, 1:-1]) &
            (grad > padded_grad[:, :, 2:, 1:-1]),
            grad,
            suppressed_grad
        )
        suppressed_grad = torch.where(
            (112.5 <= grad_direction) & (grad_direction < 157.5) &
            (grad > padded_grad[:, :, 2:, 2:]) &
            (grad > padded_grad[:, :, :-2, :-2]),
            grad,
            suppressed_grad
        )
        _watch_grad_fn(locals())
        return suppressed_grad

    def double_threshold(self, edges):
        """双阈值处理和滞后阈值处理"""
        max_value = torch.max(edges)
        min_value = torch.min(edges)
        low_threshold = min_value + (max_value - min_value) * self.low_threshold
        high_threshold = min_value + (max_value - min_value) * self.high_threshold

        strong_mask = torch.where(edges >= high_threshold, torch.ones_like(edges), torch.zeros_like(edges))
        weak_mask = torch.where(torch.logical_and(edges >= low_threshold, edges < high_threshold), torch.ones_like(edges), torch.zeros_like(edges))

        has_strong_neighbor = f.conv2d(f.pad(strong_mask, pad=(1, 1, 1, 1), mode='reflect'), self.hysteresis_kernel)

        mask = torch.where(torch.logical_or(strong_mask, torch.logical_and(weak_mask, has_strong_neighbor)), torch.ones_like(strong_mask), torch.zeros_like(strong_mask))

        filtered_edges = edges * mask.detach()

        _watch_grad_fn(locals())
        return filtered_edges


def canny_edge_detector(
        img: torch.Tensor,
        gaussian_kernel_size: int,
        gaussian_sigma: float,
        high_threshold: float,
        low_threshold: float,
):
    """
    Canny边缘检测
    :param img: 图片, dim=4
    :param gaussian_kernel_size: 高斯平滑卷积核大小
    :param gaussian_sigma: 高斯分布的标准差
    :param high_threshold: 强阈值
    :param low_threshold: 弱阈值

    Example::

        >>> import torch
        >>> import torch.nn.functional as f
        >>> from PIL import Image
        >>> from torchvision.utils import save_image
        >>> from torchvision.transforms.functional import to_tensor
        >>> from canny import canny_edge_detector
        >>> path =  'Lena.png'
        >>> img = Image.open(path).convert('L')
        >>> img = to_tensor(img).unsqueeze(0)
        >>> edge = canny_edge_detector(img, 5, 1, 0.1, 0.25)
        image None
        img <ReflectionPad2DBackward0 object at 0x00000244EE7EFF70>
        out <ConvolutionBackward0 object at 0x00000244EE7EFF70>
        image <ConvolutionBackward0 object at 0x00000244EE7EFF70>
        padded_img <ReflectionPad2DBackward0 object at 0x00000244EE7EFF70>
        grad_x <ConvolutionBackward0 object at 0x00000244EE7EFF70>
        grad_y <ConvolutionBackward0 object at 0x00000244EE7EFF70>
        grad <SqrtBackward0 object at 0x00000244EE7EFF70>
        grad_direction <WhereBackward0 object at 0x00000244EE7EFF70>
        grad <SqrtBackward0 object at 0x00000244EE7EFF70>
        grad_direction <WhereBackward0 object at 0x00000244EE7EFF70>
        padded_grad <ReflectionPad2DBackward0 object at 0x00000244EE7EFF70>
        suppressed_grad <WhereBackward0 object at 0x00000244EE7EFF70>
        edges <WhereBackward0 object at 0x00000244EE7EFF70>
        max_value <MaxBackward1 object at 0x00000244EE7EFF70>
        min_value <MinBackward1 object at 0x00000244EE7EFF70>
        low_threshold <AddBackward0 object at 0x00000244EE7EFF70>
        high_threshold <AddBackward0 object at 0x00000244EE7EFF70>
        strong_mask None
        weak_mask None
        has_strong_neighbor None
        mask None
        filtered_edges <MulBackward0 object at 0x00000244EE7EFF70>
        >>> save_image(edge, 'edge.png')
    """
    canny = CannyEdgeDetector(gaussian_kernel_size, gaussian_sigma, high_threshold, low_threshold).to(img.device)
    return canny(img)


def binaryzation_edge(edges: torch.Tensor) -> torch.Tensor:
    return torch.sign(edges).abs()
