import torch
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from build_model import build_fairmot, load_model
from TRT_Constructor import TRT_Constructor

torch.cuda.init()

with torch.no_grad():
    net = build_fairmot()
    model = load_model(net, "../weights/fairmot_dla34.pth")
    model = model.to(torch.device('cuda'))
    model.eval()

    ### TRT
    batch_size = 1
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 20
    network = builder.create_network()  #
    
    input_channel = 3
    h = 608
    w = 1088

    inputT0 = network.add_input(
        'inputT0', trt.DataType.FLOAT, (input_channel, h, w))

    constructor = TRT_Constructor(network, cuda=True)
    output = model.TRT_export(constructor, inputT0)

    network.mark_output(output[0])
    network.mark_output(output[1])
    network.mark_output(output[2])
    engine = builder.build_cuda_engine(
        network)                              #
    if engine == None:
        exit()
    with open("fairmot.trt", 'wb') as f:
        f.write(engine.serialize())