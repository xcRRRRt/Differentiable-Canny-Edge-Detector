# 可导的Canny边缘检测

# Differentiable Canny Edge Detector

### Example:

<div style="width: 98%; margin: auto;">
    <!-- 前两张图片并排 -->
    <div style="display: flex; width: 100%; justify-content: center;">
        <div style="width: 50%; text-align: center;">
            <b>Original</b>
            <img src="Lena.png" alt="Lena" style="width: 100%; height: auto;">
        </div>
        <div style="width: 50%; text-align: center;">
            <b>Differentiable</b>
            <img src="edge.png" alt="Edge" style="width: 100%; height: auto;">
        </div>
    </div>
    <div style="width: 100%; text-align: center; margin-top: -6px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
        <img src="binary_edge.png" alt="Binary Edge" style="width: 100%; height: auto;">
        <b>Binary</b>
    </div>
</div>



### Usage:

```python
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from canny import CannyEdgeDetector


def main():
    path = r"Lena.png"
    img = Image.open(path).convert('L')
    img = to_tensor(img).unsqueeze(0).to('cuda')
    img.requires_grad = True
    canny = CannyEdgeDetector(5, 1, 0.23, 0.3).to('cuda')
    edges: torch.Tensor = canny(img)
    edges.sum().backward()
    save_image(edges, fp='edge.png')


if __name__ == '__main__':
    main()
```

to see **binary** edges:

```python
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from canny import CannyEdgeDetector, binaryzation_edge


def main():
    path = r"Lena.png"
    img = Image.open(path).convert('L')
    img = to_tensor(img).unsqueeze(0).to('cuda')
    img.requires_grad = True
    canny = CannyEdgeDetector(5, 1, 0.23, 0.3).to('cuda')
    edges: torch.Tensor = canny(img)
    edges.sum().backward()
    save_image(edges, fp='edge.png')
    binary_edges = binaryzation_edge(edges)
    save_image(binary_edges, 'binary_edge.png')


if __name__ == '__main__':
    main()
```
