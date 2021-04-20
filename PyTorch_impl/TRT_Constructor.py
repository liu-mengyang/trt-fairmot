from torch import nn
import tensorrt as trt
import numpy as np


class TRT_Constructor:
    def __init__(self, network: trt.tensorrt.INetworkDefinition):
        self.network = network

    def MaxPool2d(self, pool: nn.MaxPool2d, x):
        stride = pool.stride
        y = self.network.add_pooling(
            input=x,
            type=trt.PoolingType.MAX,
            window_size=stride
        )
        y.stride = pool.stride
        y.padding = pool.padding
        return y

    def Conv2d(self, conv: nn.Conv2d, x):
        y = self.network.add_convolution(
            input=x,
            num_output_maps=conv.out_channels,
            kernel_shape=conv.kernel_size,
            kernel=conv.weight,
            bias=conv.bias
        )
        y.stride = conv.stride
        y.padding = conv.padding
        y.dilation = conv.dilation
        return y

    def BatchNorm2d(self, bn: nn.BatchNorm2d, x):
        eps = bn.eps
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        scale = gamma / np.sqrt(var**2+eps)
        shift = - mean * scale + beta
        y = self.network.add_scale(
            input=x,
            mode=trt.ScaleMode.CHANNEL,
            shift=shift,
            scale=scale
        )
        return y

    def ReLU(self, relu: nn.ReLU, x):
        y = self.network.add_parametric_relu(x)
        return y
