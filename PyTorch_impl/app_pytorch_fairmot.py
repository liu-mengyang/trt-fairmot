import torch
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from build_model import build_fairmot, load_model
from TRT_Constructor import TRT_Constructor

net = build_fairmot()
model = load_model(net, "../weights/fairmot_dla34.pth")
model = model.to(torch.device('cuda'))
model.eval()

im_blob = torch.randn([1,3,608,1088]).cuda().float()
output_torch = model(im_blob)

# 10 rounds of PyTorch FairMOT
# nRound = 10
# torch.cuda.synchronize()
# t0 = time.time()
# for i in range(nRound):
#     model(im_blob)
# torch.cuda.synchronize()
# time_pytorch = (time.time() - t0) / nRound
# print('PyTorch time:', time_pytorch)

### TRT
batch_size = 1
logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
profile = builder.create_optimization_profile()         #
config = builder.create_builder_config()               #
config.max_workspace_size = 1 << 30                                      #
config.flags = 0                                             #

input_channel = 3
h = 608
w = 1088

inputT0 = network.add_input(
    'inputT0', trt.DataType.FLOAT, (1, input_channel, h, w))
profile.set_shape(inputT0.name, (1, input_channel, h, w),
                    (1, input_channel, h, w), (1, input_channel, h, w))
config.add_optimization_profile(profile)

constructor = TRT_Constructor(network, cuda=True)
output = model.TRT_export(constructor, inputT0)

network.mark_output(output[0])
network.mark_output(output[1])
network.mark_output(output[2])
engine = builder.build_engine(
    network, config)                              #
#trt = TrtLite(build_engine_dynamic)
#trt.print_info()
#trt.save_to_file("edvr.trt")
if engine == None:
    exit()

context = engine.create_execution_context()
context.set_binding_shape(0, (batch_size, input_channel, h, w))
print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))  # 这里是
print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))
print("Bind2->", engine.get_binding_shape(2), context.get_binding_shape(2))
print("Bind3->", engine.get_binding_shape(3), context.get_binding_shape(3))

stream = cuda.Stream()

inputH0 = np.ascontiguousarray(im_blob.cpu().reshape(-1))
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
# print("execute")
context.execute_async_v2(
    [int(inputD0), int(outputD0), int(outputD1), int(outputD2)], stream.handle) 
cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
cuda.memcpy_dtoh_async(outputH2, outputD2, stream)
stream.synchronize()

diff = output_torch[0].cpu().detach().numpy()-outputH0
print("1 Average absolute difference between Pytorch and TRT:",
        np.mean(np.abs(diff)))
print("1 Average relative difference between Pytorch and TRT:",
        np.nansum(np.abs(diff/output_torch[0].cpu().detach().numpy())) / np.size(diff)
        )
diff = output_torch[1].cpu().detach().numpy()-outputH1
print("2 Average absolute difference between Pytorch and TRT:",
        np.mean(np.abs(diff)))
print("2 Average relative difference between Pytorch and TRT:",
        np.nansum(np.abs(diff/output_torch[1].cpu().detach().numpy())) / np.size(diff)
        )
diff = output_torch[2].cpu().detach().numpy()-outputH2
print("3 Average absolute difference between Pytorch and TRT:",
        np.mean(np.abs(diff)))
print("3 Average relative difference between Pytorch and TRT:",
        np.nansum(np.abs(diff/output_torch[2].cpu().detach().numpy())) / np.size(diff)
        )