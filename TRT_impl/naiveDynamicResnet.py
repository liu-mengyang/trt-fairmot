import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
def add_conv3x3(network, inputTensor, output_channel=1):
    input_channel = inputTensor.shape[1]
    conv_w = np.asarray(np.random.rand(input_channel*3*3*output_channel), dtype = np.float32)
    conv_b = np.ones((output_channel,), np.float32)
    conv = network.add_convolution(inputTensor, output_channel, (3, 3), conv_w, conv_b)
    conv.stride = (1, 1)
    conv.padding = (1, 1)
    return conv


if __name__ == '__main__':

    input_channel = 4
    output_channel = 6

    logger = trt.Logger(trt.Logger.INFO)
    if os.path.isfile('./engine.trt'):
        with open('./engine.trt', 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine( f.read() )
            if engine == None:
                exit()
    else:
        builder                     = trt.Builder(logger)
        network                     = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
        profile                     = builder.create_optimization_profile()         #
        config                      = builder.create_builder_config()               #
        config.max_workspace_size   = 1 << 30                                       #
        config.flags                = 0                                             #

        input_channel = 3
        inputT0     = network.add_input('inputT0', trt.DataType.FLOAT, (-1, input_channel, -1, -1))
        profile.set_shape(inputT0.name, (1,input_channel,4,5),(10,input_channel,12,16),(100,input_channel,15,20))
        config.add_optimization_profile(profile)

        conv1 = add_conv3x3(network, inputT0, output_channel)
        print('conv', conv1.get_output(0).shape)

        network.mark_output(conv1.get_output(0))
        engine = builder.build_engine(network, config)                              #
        if engine == None:
            exit() 
        with open('./engine.trt', 'wb') as f:
            f.write( engine.serialize() )
    
    context         = engine.create_execution_context()
    context.set_binding_shape(0,(20,input_channel,12,12))
    print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0)) # 这里是
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))

    stream          = cuda.Stream()

    data        = np.arange(20*input_channel*12*12,dtype=np.float32).reshape(20,input_channel,12,12)
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


