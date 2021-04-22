import torch
from torch import nn
import tensorrt as trt
import numpy as np
from typing import List
import ctypes

ctypes.cdll.LoadLibrary('./build/DCNv2Plugin.so')

def get_plugin_creator(plugin_name):
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

class TRT_Constructor:
    def __init__(self, network: trt.tensorrt.INetworkDefinition):
        self.network = network

    def MaxPool2d(self, pool: nn.MaxPool2d, x: trt.tensorrt.ITensor):
        stride, padding, window_size = pool.stride, pool.padding, pool.kernel_size
        tlist = [stride, padding, window_size]
        tlist = [[a, a] if type(a) is int else list(a) for a in tlist]
        tlist = [trt.tensorrt.DimsHW(a) for a in tlist]
        stride, padding, window_size = tlist
        if type(window_size) is int:
            window_size = [window_size, window_size]
        else:
            window_size = list(window_size)
        y = self.network.add_pooling(
            input=x,
            type=trt.PoolingType.MAX,
            window_size=window_size
        )
        y.stride = stride
        y.padding = padding
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

    def Sigmoid(self, x: trt.tensorrt.ITensor):
        y = self.network.add_activation(x, trt.ActivationType.SIGMOID)
        return y.get_output(0)

    def Elementwise(self, a: trt.tensorrt.ITensor, b: trt.tensorrt.ITensor, op: trt.tensorrt.ElementWiseOperation):
        y = self.network.add_elementwise(a, b, op)
        return y.get_output(0)

    def Add(self, a: trt.tensorrt.ITensor, b: trt.tensorrt.ITensor):
        return self.Elementwise(a, b, trt.tensorrt.ElementWiseOperation.SUM)

    def Concat(self, i: List[trt.tensorrt.ITensor]):
        y = self.network.add_concatenation(i)
        return y.get_output(0)

    def Slice(self, x: trt.tensorrt.ITensor, start: trt.tensorrt.Dims, shape: trt.tensorrt.Dims, stride: trt.tensorrt.Dims):
        y = self.network.add_slice(x, start, shape, stride)
        return y.get_output(0)

    def Constant(self, shape: trt.tensorrt.Dims, weights: trt.tensorrt.Weights):
        y = self.network.add_constant(shape, weights)
        return y.get_output(0)

    def DCNv2(self, x: trt.tensorrt.ITensor, out_channels, offset: trt.tensorrt.ITensor, mask: trt.tensorrt.ITensor, weight: trt.tensorrt.ITensor, bias: trt.tensorrt.ITensor):
        plugin_creator = get_plugin_creator('DCNv2Plugin')
        if plugin_creator == None:
            print('Plugin DCNv2Plugin not found. Exiting')
            exit()
        fc = trt.PluginFieldCollection()
        fc.append(trt.PluginField("out_channel", np.array([out_channels], np.int32)))
        y = self.network.add_plugin_v2([x, offset, mask, weight, bias], 
            plugin_creator.create_plugin('DCNv2Plugin', fc))
        return y.get_output(0)
