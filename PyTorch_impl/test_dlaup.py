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

channels = [16, 32, 64, 128, 256, 512]

def test_fun(m: nn.Module):  # 输入待测试的nn.Module，主要测其中的m.TRT_export方法
    batch_size = 1
    m.eval()

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
    profile = builder.create_optimization_profile()         #
    config = builder.create_builder_config()               #
    config.max_workspace_size = 1 << 30                                       #
    config.flags = 0                                             #

    inputT0 = network.add_input(
        'inputT0', trt.DataType.FLOAT, (-1, channels[0], int(h), int(w)))
    inputT1 = network.add_input(
        'inputT1', trt.DataType.FLOAT, (-1, channels[1], int(h/2), int(w/2)))
    inputT2 = network.add_input(
        'inputT2', trt.DataType.FLOAT, (-1, channels[2], int(h/4), int(w/4)))
    inputT3 = network.add_input(
        'inputT3', trt.DataType.FLOAT, (-1, channels[3], int(h/8), int(w/8)))
    inputT4 = network.add_input(
        'inputT4', trt.DataType.FLOAT, (-1, channels[4], int(h/16), int(w/16)))
    inputT5 = network.add_input(
        'inputT5', trt.DataType.FLOAT, (-1, channels[5], int(h/32), int(w/32)))

    profile.set_shape(inputT0.name, (1, channels[0], int(h), int(w)),
                      (1, channels[0], int(h), int(w)), (1, channels[0], int(h), int(w)))
    profile.set_shape(inputT1.name, (1, channels[1], int(h/2), int(w/2)),
                      (1, channels[1], int(h/2), int(w/2)), (1, channels[1], int(h/2), int(w/2)))
    profile.set_shape(inputT2.name, (1, channels[2], int(h/4), int(w/4)),
                      (1, channels[2], int(h/4), int(w/4)), (1, channels[2], int(h/4), int(w/4)))
    profile.set_shape(inputT3.name, (1, channels[3], int(h/8), int(w/8)),
                      (1, channels[3], int(h/8), int(w/8)), (1, channels[3], int(h/8), int(w/8)))
    profile.set_shape(inputT4.name, (1, channels[4], int(h/16), int(w/16)),
                      (1, channels[4], int(h/16), int(w/16)), (1, channels[4], int(h/16), int(w/16)))
    profile.set_shape(inputT5.name, (1, channels[5], int(h/32), int(w/32)),
                      (1, channels[5], int(h/32), int(w/32)), (1, channels[5], int(h/32), int(w/32)))
    config.add_optimization_profile(profile)

    constructor = TRT_Constructor(network)
    output = m.TRT_export(constructor, [inputT2, inputT3, inputT4, inputT5])

    network.mark_output(output[0])
    engine = builder.build_engine(
        network, config)                              #
    if engine == None:
        exit()

    context = engine.create_execution_context()
    # context.set_binding_shape(0, (batch_size, channels[-6], int(h), int(w)))
    # context.set_binding_shape(1, (batch_size, channels[-5], int(h/2), int(w/2)))
    context.set_binding_shape(0, (batch_size, channels[-4], int(h/4), int(w/4)))
    context.set_binding_shape(1, (batch_size, channels[-3], int(h/8), int(w/8)))
    context.set_binding_shape(2, (batch_size, channels[-2], int(h/16), int(w/16)))
    context.set_binding_shape(3, (batch_size, channels[-1], int(h/32), int(w/32)))
    # print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))  # 这里是
    # print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))
    print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))
    print("Bind2->", engine.get_binding_shape(2), context.get_binding_shape(2))
    print("Bind3->", engine.get_binding_shape(3), context.get_binding_shape(3))
    print("Bind4->", engine.get_binding_shape(4), context.get_binding_shape(4))
    stream = cuda.Stream()

    data1 = np.arange(1*channels[0]*int(h)*int(w),
                     dtype=np.float32).reshape(batch_size, channels[0], int(h), int(w))/channels[0]/int(h)+10
    data2 = np.arange(1*channels[1]*int(h/2)*int(w/2),
                     dtype=np.float32).reshape(batch_size, channels[1], int(h/2), int(w/2))/channels[1]/int(h/2)+10
    data3 = np.arange(1*channels[2]*int(h/4)*int(w/4),
                     dtype=np.float32).reshape(batch_size, channels[2], int(h/4), int(w/4))/channels[2]/int(h/4)+10
    data4 = np.arange(1*channels[3]*int(h/8)*int(w/8),
                     dtype=np.float32).reshape(batch_size, channels[3], int(h/8), int(w/8))/channels[3]/int(h/8)+10
    data5 = np.arange(1*channels[4]*int(h/16)*int(w/16),
                     dtype=np.float32).reshape(batch_size, channels[4], int(h/16), int(w/16))/channels[4]/int(h/16)+10
    data6 = np.arange(1*channels[5]*int(h/32)*int(w/32),
                     dtype=np.float32).reshape(batch_size, channels[5], int(h/32), int(w/32))/channels[5]/int(h/32)+10
    # inputH0 = np.ascontiguousarray(data1.reshape(-1))
    # inputD0 = cuda.mem_alloc(inputH0.nbytes)
    # inputH1 = np.ascontiguousarray(data2.reshape(-1))
    # inputD1 = cuda.mem_alloc(inputH1.nbytes)
    inputH2 = np.ascontiguousarray(data3.reshape(-1))
    inputD2 = cuda.mem_alloc(inputH2.nbytes)
    inputH3 = np.ascontiguousarray(data4.reshape(-1))
    inputD3 = cuda.mem_alloc(inputH3.nbytes)
    inputH4 = np.ascontiguousarray(data5.reshape(-1))
    inputD4 = cuda.mem_alloc(inputH4.nbytes)
    inputH5 = np.ascontiguousarray(data6.reshape(-1))
    inputD5 = cuda.mem_alloc(inputH5.nbytes)

    outputH0 = np.empty(context.get_binding_shape(
        4), dtype=trt.nptype(engine.get_binding_dtype(4)))    #
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    # cuda.memcpy_htod_async(inputD0, inputH0, stream)
    # cuda.memcpy_htod_async(inputD1, inputH1, stream)
    cuda.memcpy_htod_async(inputD2, inputH2, stream)
    cuda.memcpy_htod_async(inputD3, inputH3, stream)
    cuda.memcpy_htod_async(inputD4, inputH4, stream)
    cuda.memcpy_htod_async(inputD5, inputH5, stream)
    
    print("execute")
    context.execute_async_v2(
        [int(inputD2), int(inputD3), int(inputD4), int(inputD5), int(outputD0)], stream.handle) 
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()

    print("inputH0:", data1.shape, engine.get_binding_dtype(0))
    print(data1)
    print("outputH0:", outputH0.shape, engine.get_binding_dtype(1))
    print(outputH0)

    outputH0_torch = m([torch.tensor(data1),torch.tensor(data2),torch.tensor(data3),torch.tensor(data4),torch.tensor(data5),torch.tensor(data6)])
    print("outputH0 in Pytorch:", outputH0_torch[0].shape)
    print(outputH0_torch[0])

    diff = outputH0_torch[0].cpu().detach().numpy()-outputH0
    print("Average absolute difference between Pytorch and TRT:",
          np.mean(np.abs(diff)))
    print("Average relative difference between Pytorch and TRT:",
          np.nansum(np.abs(diff/outputH0_torch[0].cpu().detach().numpy())) / np.size(diff)
          )
    print(diff)
