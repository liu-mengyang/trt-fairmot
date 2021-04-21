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

    def TRT_export(self, constructor: TRT_Constructor, x, residual=None):
        if residual is None:
            residual = x

        out = constructor.Conv2d(self.conv1, x)
        out = constructor.BatchNorm2d(self.bn1, out)
        out = constructor.ReLU(self.relu, out)

        out = constructor.Conv2d(self.conv2, out)
        out = constructor.BatchNorm2d(self.bn2, out)

        out = constructor.Add(out, residual)
        out = constructor.ReLU(self.relu, out)
        return out

if __name__ == '__main__':
    import os
    import numpy as np
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
    import torch

    # 以下为TensorRT对比测试代码

    input_channel = 2
    output_channel = 2
    h = 5
    w = 5
    batch_size = 2
    m = BasicBlock(input_channel, output_channel) # Pytorch构建的模型

    logger = trt.Logger(trt.Logger.INFO)
    builder                     = trt.Builder(logger)
    network                     = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
    profile                     = builder.create_optimization_profile()         #
    config                      = builder.create_builder_config()               #
    config.max_workspace_size   = 1 << 30                                       #
    config.flags                = 0                                             #

    inputT0     = network.add_input('inputT0', trt.DataType.FLOAT, (-1, input_channel, -1, -1))
    profile.set_shape(inputT0.name, (1,input_channel,1,1),(10,input_channel,10,10),(100,input_channel,100,100))
    config.add_optimization_profile(profile)
    
    constructor = TRT_Constructor(network)
    output = m.TRT_export(constructor, inputT0)

    network.mark_output(output)
    engine = builder.build_engine(network, config)                              #
    if engine == None:
        exit()
    
    context         = engine.create_execution_context()
    context.set_binding_shape(0,(batch_size,input_channel,h,w))
    print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0)) # 这里是
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))

    stream          = cuda.Stream()

    data        = np.arange(2*input_channel*h*w,dtype=np.float32).reshape(batch_size,input_channel,h,w)*100+100
    # data        = np.random.randn(batch_size,input_channel,h,w)*10+40
    inputH0     = np.ascontiguousarray(data.reshape(-1))
    inputD0     = cuda.mem_alloc(inputH0.nbytes)
    outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))    #
    outputD0    = cuda.mem_alloc(outputH0.nbytes)
        
    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream.handle)          #
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()
    
    print("inputH0:", data.shape,engine.get_binding_dtype(0))
    print(data)
    print("outputH0:", outputH0.shape,engine.get_binding_dtype(1))
    print(outputH0)

    outputH0_torch = m(torch.Tensor(data))
    print("outputH0 in Pytorch:", outputH0_torch.shape)
    print(outputH0_torch)

    diff = outputH0_torch.detach().numpy()-outputH0
    print("Average absolute difference between Pytorch and TRT:", np.mean(np.abs(diff)))
    print("Average relative difference between Pytorch and TRT:", np.mean(np.abs(diff/outputH0)))
    print(diff)
