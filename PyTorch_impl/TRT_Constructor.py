from torch import nn
import tensorrt as trt
import numpy as np
from typing import List

class TRT_Constructor:
    def __init__(self, network: trt.tensorrt.INetworkDefinition):
        self.network = network

    def MaxPool2d(self, pool: nn.MaxPool2d, x: trt.tensorrt.ITensor):
        stride = pool.stride
        y = self.network.add_pooling(
            input=x,
            type=trt.PoolingType.MAX,
            window_size=stride
        )
        y.stride = pool.stride
        y.padding = pool.padding
        return y.get_output(0)

    def Conv2d(self, conv: nn.Conv2d, x: trt.tensorrt.ITensor):
        y = self.network.add_convolution(
            input=x,
            num_output_maps=conv.out_channels,
            kernel_shape=conv.kernel_size,
            kernel=conv.weight.detach().numpy(),
            bias=conv.bias.detach().numpy() if conv.bias is not None else None
        )
        y.stride = conv.stride
        y.padding = conv.padding
        y.dilation = conv.dilation
        return y.get_output(0)

    def BatchNorm2d(self, bn: nn.BatchNorm2d, x: trt.tensorrt.ITensor):
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
            shift=shift.detach().numpy(),
            scale=scale.detach().numpy()
        )
        return y.get_output(0)

    def ReLU(self, relu: nn.ReLU, x: trt.tensorrt.ITensor):
        y = self.network.add_activation(x, trt.ActivationType.RELU)
        return y.get_output(0)

    def Elementwise(self, a: trt.tensorrt.ITensor, b: trt.tensorrt.ITensor, op: trt.tensorrt.ElementWiseOperation):
        y = self.network.add_elementwise(a, b, op)
        return y.get_output(0)

    def Add(self, a: trt.tensorrt.ITensor, b: trt.tensorrt.ITensor):
        return self.Elementwise(a, b, trt.tensorrt.ElementWiseOperation.SUM)

    def Concat(self, i: List[trt.tensorrt.ITensor]):
        y = self.network.add_concatenation(i)
        return y.get_output(0)
