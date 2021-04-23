import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch
from torch import nn

from TRT_Constructor import TRT_Constructor

# 用于测试的函数
input_channel = 3
output_channel = 16
h = 608
w = 1088


def test_fun(m: nn.Module):  # 输入待测试的nn.Module，主要测其中的m.TRT_export方法

    batch_size = 1
    m.eval()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
    profile = builder.create_optimization_profile()         #
    config = builder.create_builder_config()               #
    # config.max_workspace_size = 1 << 30                                       #
    config.flags = 0                                             #

    inputT0 = network.add_input(
        'inputT0', trt.DataType.FLOAT, (-1, input_channel, -1, -1))
    profile.set_shape(inputT0.name, (1, input_channel, h, w),
                      (10, input_channel, h, w), (100, input_channel, h, w))
    config.add_optimization_profile(profile)

    constructor = TRT_Constructor(network)
    output = m.TRT_export(constructor, inputT0)

    if type(output) is list:
        for o in output:
            network.mark_output(o)
    else:
        network.mark_output(output)
    engine = builder.build_engine(
        network, config)                              #
    if engine == None:
        exit()

    context = engine.create_execution_context()
    context.set_binding_shape(0, (batch_size, input_channel, h, w))
    print("Bind0->", engine.get_binding_shape(0),
          context.get_binding_shape(0))  # 这里是
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))

    print("Bind2->", engine.get_binding_shape(2), context.get_binding_shape(2))
    print("Bind3->", engine.get_binding_shape(3), context.get_binding_shape(3))
    print("Bind4->", engine.get_binding_shape(4), context.get_binding_shape(4))
    print("Bind5->", engine.get_binding_shape(5), context.get_binding_shape(5))
    print("Bind6->", engine.get_binding_shape(6), context.get_binding_shape(6))


    stream = cuda.Stream()

    data = np.arange(1*input_channel*h*w,
                     dtype=np.float32).reshape(batch_size, input_channel, h, w)*10+10
    # data        = np.random.randn(batch_size,input_channel,h,w)*10+40
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)

    outputH1 = np.empty(context.get_binding_shape(
        1), dtype=trt.nptype(engine.get_binding_dtype(1)))    #
    outputD1 = cuda.mem_alloc(outputH1.nbytes)

    outputH2 = np.empty(context.get_binding_shape(
        2), dtype=trt.nptype(engine.get_binding_dtype(2)))    #
    outputD2 = cuda.mem_alloc(outputH2.nbytes)

    outputH3 = np.empty(context.get_binding_shape(
        3), dtype=trt.nptype(engine.get_binding_dtype(3)))    #
    outputD3 = cuda.mem_alloc(outputH3.nbytes)

    outputH4 = np.empty(context.get_binding_shape(
        4), dtype=trt.nptype(engine.get_binding_dtype(4)))    #
    outputD4 = cuda.mem_alloc(outputH4.nbytes)

    outputH5 = np.empty(context.get_binding_shape(
        5), dtype=trt.nptype(engine.get_binding_dtype(5)))    #
    outputD5 = cuda.mem_alloc(outputH5.nbytes)

    outputH6 = np.empty(context.get_binding_shape(
        6), dtype=trt.nptype(engine.get_binding_dtype(6)))    #
    outputD6 = cuda.mem_alloc(outputH6.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async_v2(
        [int(inputD0), int(outputD1), int(outputD2), int(outputD3), int(outputD4), int(outputD5), int(outputD6)], stream.handle)          #
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    stream.synchronize()

    print("inputH0:", data.shape, engine.get_binding_dtype(0))
    print(data)
    print("outputH1:", outputH1.shape, engine.get_binding_dtype(1))
    print(outputH1)

    outputH0_torch = m(torch.Tensor(data))
    print("outputH0 in Pytorch:", outputH0_torch[0].shape)
    print(outputH0_torch[0])

    diff = outputH0_torch[0].detach().numpy()-outputH1
    print("Average absolute difference between Pytorch and TRT:",
          np.mean(np.abs(diff)))
    print("Average relative difference between Pytorch and TRT:",
          np.nansum(np.abs(diff/outputH0_torch[0].cpu().detach().numpy())) / np.size(diff))
    print(diff)
