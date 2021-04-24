import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch
from torch import nn

from TRT_Constructor import TRT_Constructor

np.seterr(divide='ignore',invalid='ignore')

# 用于测试的函数
input_channel = 3
output_channel = 256
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
    config.max_workspace_size = 1 << 30                                       #
    config.flags = 0                                             #

    inputT0 = network.add_input(
        'inputT0', trt.DataType.FLOAT, (-1, input_channel, h, w))
    profile.set_shape(inputT0.name, (1, input_channel, h, w),
                      (10, input_channel, h, w), (100, input_channel, h, w))
    config.add_optimization_profile(profile)

    constructor = TRT_Constructor(network)
    output = m.TRT_export(constructor, inputT0)

    if type(output) is list:
        for o in output:
            network.mark_output(o)
    engine = builder.build_engine(
        network, config)                              #
    if engine == None:
        exit()

    context = engine.create_execution_context()
    context.set_binding_shape(0, (batch_size, input_channel, h, w))
    print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))  # 这里是
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))
    print("Bind2->", engine.get_binding_shape(2), context.get_binding_shape(2))
    print("Bind3->", engine.get_binding_shape(3), context.get_binding_shape(3))

    stream = cuda.Stream()

    data = np.arange(1*input_channel*h*w,
                     dtype=np.float32).reshape(batch_size, input_channel, h, w)/input_channel/h/w*255+10
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    outputH0 = np.empty(context.get_binding_shape(
        1), dtype=trt.nptype(engine.get_binding_dtype(1)))    #
    outputD0 = cuda.mem_alloc(outputH0.nbytes)
    outputH1 = np.empty(context.get_binding_shape(
        2), dtype=trt.nptype(engine.get_binding_dtype(2)))    #
    outputD1 = cuda.mem_alloc(outputH1.nbytes)
    outputH2 = np.empty(context.get_binding_shape(
        3), dtype=trt.nptype(engine.get_binding_dtype(3)))    #
    outputD2 = cuda.mem_alloc(outputH2.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    print("execute")
    context.execute_async_v2(
        [int(inputD0), int(outputD0), int(outputD1), int(outputD2)], stream.handle) 
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    cuda.memcpy_dtoh_async(outputH2, outputD2, stream)
    stream.synchronize()

    # print("inputH0:", data.shape, engine.get_binding_dtype(0))
    # print(data)
    # print("outputH0:", outputH0.shape, engine.get_binding_dtype(1))
    # print(outputH0)

    outputH0_torch = m(torch.tensor(data))
    # print("outputH0 in Pytorch:", outputH0_torch[0].shape)
    # print(outputH0_torch[0])

    diff = outputH0_torch[0].cpu().detach().numpy()-outputH0
    print("1 Average absolute difference between Pytorch and TRT:",
          np.mean(np.abs(diff)))
    print("1 Average relative difference between Pytorch and TRT:",
          np.nansum(np.abs(diff/outputH0_torch[0].cpu().detach().numpy())) / np.size(diff)
          )
    diff = outputH0_torch[1].cpu().detach().numpy()-outputH1
    print("2 Average absolute difference between Pytorch and TRT:",
          np.mean(np.abs(diff)))
    print("2 Average relative difference between Pytorch and TRT:",
          np.nansum(np.abs(diff/outputH0_torch[0].cpu().detach().numpy())) / np.size(diff)
          )
    diff = outputH0_torch[2].cpu().detach().numpy()-outputH2
    print("3 Average absolute difference between Pytorch and TRT:",
          np.mean(np.abs(diff)))
    print("3 Average relative difference between Pytorch and TRT:",
          np.nansum(np.abs(diff/outputH0_torch[0].cpu().detach().numpy())) / np.size(diff)
          )
    # print(diff)
