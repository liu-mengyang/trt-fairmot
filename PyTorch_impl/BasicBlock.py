import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch
from torch import nn

from Config import BN_MOMENTUM

from TRT_Constructor import TRT_Constructor

# Residual Block
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

    def TRT_export(self, constructor: TRT_Constructor, x):
        if residual is None:
            residual = x

        out = constructor.Conv2d(self.conv1, x)
        out = constructor.BatchNorm2d(self.bn1, out)
        out = constructor.ReLU(self.relu, out)

        out = constructor.Conv2d(self.conv2, out)
        out = constructor.BatchNorm2d(self.bn2, out)

        out += residual
        out = constructor.ReLU(self.relu, out)
        return out

if __name__ == '__main__':

    # 以下为TensorRT对比测试代码

    input_channel = 2
    output_channel = 2
    m = BasicBlock(input_channel, output_channel) # Pytorch构建的模型

    data        = np.arange(2*input_channel*3*3,dtype=np.float32).reshape(2,input_channel,3,3)
    inputH0     = np.ascontiguousarray(data.reshape(-1))

    outputH0_torch = m(torch.Tensor(data))
    print("outputH0 in Pytorch:", outputH0_torch.shape)
    print(outputH0_torch)

