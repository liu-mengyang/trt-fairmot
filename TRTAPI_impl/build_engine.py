import torch
import time
import numpy as np
import tensorrt as trt

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
    input_channel = 3
    h = 608
    w = 1088

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    config.flags = 0
    
    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (1, input_channel, h, w))
    profile.set_shape(inputT0.name, (1,input_channel,h,w),(1,input_channel,h,w),(1,input_channel,h,w))                    #
    config.add_optimization_profile(profile)

    constructor = TRT_Constructor(network, cuda=True)
    output = model.TRT_export(constructor, inputT0)

    network.mark_output(output[0])
    network.mark_output(output[1])
    network.mark_output(output[2])
    engine = builder.build_engine(network, config)                              #
    if engine == None:
        exit()
    with open("fairmot.trt", 'wb') as f:
        f.write(engine.serialize())