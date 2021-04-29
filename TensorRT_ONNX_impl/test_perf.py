import torch
import torchvision
from multiprocessing import Process
import time
import ctypes
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from build_model import build_fairmot, load_model

ctypes.cdll.LoadLibrary('./build/DCNv2PluginDyn.so')

net = build_fairmot()
model = load_model(net, "../weights/fairmot_dla34.pth")
model = model.to(torch.device('cuda'))
model.eval()

batch_size = 1
input_channel = 3
height = 608
width = 1088
im_blob = torch.randn([batch_size, input_channel, height, width]).cuda().float()
print('****  ',batch_size,'  ****')
print('====', 'fairmot-pytorch', '===')
output = model(im_blob)

# 30 rounds of PyTorch FairMOT
nRound = 30
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    model(im_blob)
torch.cuda.synchronize()
time_pytorch = (time.time() - t0) * 1000 / nRound
print('PyTorch time:', time_pytorch)
throughout_pytorch = 1000 / time_pytorch * batch_size
print('PyTorch throughout:', throughout_pytorch)
from trt_lite import TrtLite
import os


class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor
    def get_pointer(self):
        return self.tensor.data_ptr()

for engine_file_path in ['fairmot.trt', 'fairmot_fp16.trt']:
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1) 
    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    i2shape = {0: (batch_size, input_channel, height, width)}
    io_info = trt.get_io_info(i2shape)
    d_buffers = trt.allocate_io_buffers(i2shape, True)    
    output1_data_trt = np.zeros(io_info[1][2], dtype=np.float32)
    output2_data_trt = np.zeros(io_info[2][2], dtype=np.float32)
    output3_data_trt = np.zeros(io_info[3][2], dtype=np.float32)

    # input from device to device
    cuda.memcpy_dtod(d_buffers[0], PyTorchTensorHolder(im_blob), im_blob.nelement() * im_blob.element_size())
    trt.execute(d_buffers, i2shape)
    
    cuda.memcpy_dtoh(output1_data_trt, d_buffers[1])
    cuda.memcpy_dtoh(output2_data_trt, d_buffers[2])
    cuda.memcpy_dtoh(output3_data_trt, d_buffers[3])

    cuda.Context.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute(d_buffers, i2shape)
    cuda.Context.synchronize()
    time_trt = (time.time() - t0) * 1000 / nRound
    print('TensorRT time:', time_trt)
    throughout_trt = 1000 / time_trt * batch_size
    print('TensorRT throughout:', throughout_trt)
    print('Latency speedup:', time_pytorch / time_trt)
    print('Throughout speedup:', throughout_trt / throughout_pytorch)