#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author  : xcRt
@time    : 2024/11/23 04:02
@function: Example of Canny Edge Detector
@version : V1
"""
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from canny import CannyEdgeDetector, canny_edge_detector, binaryzation_edge


def main():
    path = r"Lena.png"
    img = Image.open(path).convert('L')
    img = to_tensor(img).unsqueeze(0).to('cuda')
    img.requires_grad = True
    # canny = CannyEdgeDetector(5, 1, 0.15, 0.3).to('cuda')
    # edges: torch.Tensor = canny(img)
    edges: torch.Tensor = canny_edge_detector(img, 5, 1, 0.23, 0.3)
    save_image(edges, fp='./edge.png')
    edges.sum().backward()
    binary_edges = binaryzation_edge(edges)  # non-differentiable
    save_image(binary_edges, fp='./binary_edge.png')


if __name__ == '__main__':
    main()