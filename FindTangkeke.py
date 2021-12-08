import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

img = plt.imread("tkk.jpg")  # 被寻找的图片
kernel = plt.imread("kernel.jpg")  # 卷积核图片

transform = transforms.ToTensor()  # 转换形式

# 打开被寻找的图片
plt.imshow(img)
plt.show()

# 转换为张量
img_ts = transform(img)

# print(img_ts)

# 打开卷积核图片
plt.imshow(kernel)
plt.show()

# 转换为张量
kernel = transform(kernel)


# print(kernel)

def findTangkeke(pic, kernel):
    H = kernel.shape[1]  # 卷积核高
    W = kernel.shape[2]  # 卷积核宽
    # print(H, W)

    conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=H, padding=0, stride=H)
    conv.weight.data = kernel.view(1, 3, H, W)  # 自定义卷积核
    x = conv(pic)
    return x.detach().numpy()  # 转换为 numpy 数组


H = img_ts.shape[1]
W = img_ts.shape[2]
img_ts = img_ts.view(1, 3, H, W)  # 保持维度相同

np1 = findTangkeke(img_ts, kernel)
np1 = np.squeeze(np1)  # 保持维度相同
pos = np.unravel_index(np.argmax(np1), np1.shape)
# print(pos)

plt.imshow(np1)
plt.show()
