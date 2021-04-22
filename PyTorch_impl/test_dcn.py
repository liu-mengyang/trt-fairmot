import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch
from torch import nn

from TRT_Constructor import TRT_Constructor

# 用于测试的函数
input_channel = 2
output_channel = 2
h = 50
w = 50

def test_fun(m: nn.Module):  # 输入待测试的nn.Module，主要测其中的m.TRT_export方法
    batch_size = 1
    m.eval()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
    profile = builder.create_optimization_profile()         #
    config = builder.create_builder_config()               #
    config.max_workspace_size = 1 << 30                                       #
    config.flags = 0                                             #

    inputT0 = network.add_input(
        'inputT0', trt.DataType.FLOAT, (-1, input_channel, h, w))
    profile.set_shape(inputT0.name, (1, input_channel, h, w),
                      (10, input_channel, h, w), (100, input_channel, h, w))
    config.add_optimization_profile(profile)

    constructor = TRT_Constructor(network)
    output = m.TRT_export(constructor, inputT0)

    network.mark_output(output)
    engine = builder.build_engine(
        network, config)                              #
    if engine == None:
        exit()

    context = engine.create_execution_context()
    context.set_binding_shape(0, (batch_size, input_channel, h, w))
    print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))  # 这里是
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))

    stream = cuda.Stream()

    data = np.arange(1*input_channel*h*w,
                     dtype=np.float32).reshape(batch_size, input_channel, h, w)*10+10
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    outputH0 = np.empty(context.get_binding_shape(
        1), dtype=trt.nptype(engine.get_binding_dtype(1)))    #
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    print("execute")
    context.execute_async_v2(
        [int(inputD0), int(outputD0)], stream.handle) 
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()

    print("inputH0:", data.shape, engine.get_binding_dtype(0))
    print(data)
    print("outputH0:", outputH0.shape, engine.get_binding_dtype(1))
    print(outputH0)

    outputH0_torch = m(torch.tensor(data))
    print("outputH0 in Pytorch:", outputH0_torch.shape)
    print(outputH0_torch)
    print(outputH0_torch.cpu().detach().numpy())

    diff = outputH0_torch.cpu().detach().numpy()-outputH0
    print("Average absolute difference between Pytorch and TRT:",
          np.mean(np.abs(diff)))
    print("Average relative difference between Pytorch and TRT:",
          np.mean(np.abs(diff/outputH0)))
    print(diff)
