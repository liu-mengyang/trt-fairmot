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

channels = [64, 128, 256]

def test_fun(m: nn.Module):  # 输入待测试的nn.Module，主要测其中的m.TRT_export方法
    with torch.no_grad():
        batch_size = 1
        m.eval()

        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 20

        # network = builder.create_network(
        #     1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
        # profile = builder.create_optimization_profile()         #
        # config = builder.create_builder_config()               #
        # config.max_workspace_size = 1 << 30                                       #
        # config.flags = 0                                             #

        network = builder.create_network()
        inputT0 = network.add_input(
            'inputT0', trt.DataType.FLOAT, (channels[0], int(h/4), int(w/4)))
        inputT1 = network.add_input(
            'inputT1', trt.DataType.FLOAT, (channels[1], int(h/8), int(w/8)))
        inputT2 = network.add_input(
            'inputT2', trt.DataType.FLOAT, (channels[2], int(h/16), int(w/16)))

        # profile.set_shape(inputT0.name, (1, channels[0], int(h/4), int(w/4)),
        #                   (1, channels[0], int(h/4), int(w/4)), (1, channels[0], int(h/4), int(w/4)))
        # profile.set_shape(inputT1.name, (1, channels[1], int(h/8), int(w/8)),
        #                   (1, channels[1], int(h/8), int(w/8)), (1, channels[1], int(h/8), int(w/8)))
        # profile.set_shape(inputT2.name, (1, channels[2], int(h/16), int(w/16)),
        #                   (1, channels[2], int(h/16), int(w/16)), (1, channels[2], int(h/16), int(w/16)))
        # config.add_optimization_profile(profile)

        constructor = TRT_Constructor(network)
        output = m.TRT_export(constructor, [inputT0, inputT1, inputT2], 0, 3)

        network.mark_output(output)
        engine = builder.build_cuda_engine(network)
        # engine = builder.build_engine(
        #     network, config)                              #
        if engine == None:
            exit()

        context = engine.create_execution_context()
        context.set_binding_shape(0, (channels[0], int(h/4), int(w/4)))
        context.set_binding_shape(1, (channels[1], int(h/8), int(w/8)))
        context.set_binding_shape(2, (channels[2], int(h/16), int(w/16)))
        print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))  # 这里是
        print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))
        print("Bind2->", engine.get_binding_shape(2), context.get_binding_shape(2))
        print("Bind3->", engine.get_binding_shape(3), context.get_binding_shape(3))
        stream = cuda.Stream()

        data1 = np.arange(1*channels[0]*int(h/4)*int(w/4),
                        dtype=np.float32).reshape(channels[0], int(h/4), int(w/4))/channels[0]+10
        data2 = np.arange(1*channels[1]*int(h/8)*int(w/8),
                        dtype=np.float32).reshape(channels[1], int(h/8), int(w/8))/channels[1]+10
        data3 = np.arange(1*channels[2]*int(h/16)*int(w/16),
                        dtype=np.float32).reshape(channels[2], int(h/16), int(w/16))/channels[2]+10
        np.random.shuffle(data1.reshape(-1))
        np.random.shuffle(data2.reshape(-1))
        np.random.shuffle(data3.reshape(-1))
        inputH0 = np.ascontiguousarray(data1)
        inputD0 = cuda.mem_alloc(inputH0.nbytes)
        inputH1 = np.ascontiguousarray(data2)
        inputD1 = cuda.mem_alloc(inputH1.nbytes)
        inputH2 = np.ascontiguousarray(data3)
        inputD2 = cuda.mem_alloc(inputH2.nbytes)
        outputH0 = np.empty(context.get_binding_shape(
            3), dtype=trt.nptype(engine.get_binding_dtype(3)))    #
        outputD0 = cuda.mem_alloc(outputH0.nbytes)

        cuda.memcpy_htod(inputD0, inputH0)
        cuda.memcpy_htod(inputD1, inputH1)
        cuda.memcpy_htod(inputD2, inputH2)
        print("execute")
        context.execute_async(
            1, [int(inputD0), int(inputD1), int(inputD2), int(outputD0)], stream.handle) 
        stream.synchronize()
        cuda.memcpy_dtoh(outputH0, outputD0)
        stream.synchronize()

        print("inputH0:", data1.shape, engine.get_binding_dtype(0))
        print(data1)
        print("outputH0:", outputH0.shape, engine.get_binding_dtype(3))
        print(outputH0)
        with torch.no_grad():
            outputH0_torch = m([torch.tensor(data1.reshape(1, channels[0], int(h/4), int(w/4))),torch.tensor(data2.reshape(1, channels[1], int(h/8), int(w/8))),torch.tensor(data3.reshape(1, channels[2], int(h/16), int(w/16)))], 0, 3)
        print("outputH0 in Pytorch:", outputH0_torch.shape)
        print(outputH0_torch)

        diff = outputH0_torch.numpy()-outputH0
        print("Average absolute difference between Pytorch and TRT:",
            np.mean(np.abs(diff)))
        print("Average relative difference between Pytorch and TRT:",
            np.nansum(np.abs(diff/outputH0_torch.numpy())) / np.size(diff)
            )
        print(diff)
