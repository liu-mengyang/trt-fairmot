import torch
import torchvision
from torchsummary import summary
import time
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes

from build_model import build_fairmot, load_model

ctypes.cdll.LoadLibrary('./build/DCNv2Plugin.so')

net = build_fairmot()
model = load_model(net, "../weights/fairmot_dla34.pth")
model = model.to(torch.device('cuda'))
model.eval()
im_blob = torch.randn([1,3,608,1088]).cuda().float()
output = model(im_blob)


# print(output)

output1_data_pytorch = output[0]['hm'].cpu().detach().numpy()
print(np.mean(output1_data_pytorch))
output2_data_pytorch = output[0]['wh'].cpu().detach().numpy()
output3_data_pytorch = output[0]['id'].cpu().detach().numpy()
# print(output1_data_pytorch.shape)
# print(output2_data_pytorch.shape)
# print(output3_data_pytorch.shape)
print(output1_data_pytorch)
print(output2_data_pytorch)
print(output3_data_pytorch)

# 10 rounds of PyTorch FairMOT
# nRound = 10
# torch.cuda.synchronize()
# t0 = time.time()
# for i in range(nRound):
#     model(im_blob)
# torch.cuda.synchronize()
# time_pytorch = (time.time() - t0) / nRound
# print('PyTorch time:', time_pytorch)


from trt_lite import TrtLite
import numpy as np
import os

class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor
    def get_pointer(self):
        return self.tensor.data_ptr()

for engine_file_path in ['fairmot.trt']:
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)
    
    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    i2shape = {0: (1, 3, 608, 1088)}
    io_info = trt.get_io_info(i2shape)
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    output1_data_trt = np.zeros(io_info[1][2], dtype=np.float32)
    output2_data_trt = np.zeros(io_info[2][2], dtype=np.float32)
    output3_data_trt = np.zeros(io_info[3][2], dtype=np.float32)
    
    # print(output1_data_trt.shape)
    # print(output2_data_trt.shape)
    # print(output3_data_trt.shape)

    #利用PyTorch和PyCUDA的interop，保留数据始终在显存上
    cuda.memcpy_dtod(d_buffers[0], PyTorchTensorHolder(im_blob), im_blob.nelement() * im_blob.element_size())
    #下面一行的作用跟上一行一样，不过它是把数据拷到cpu再拷回gpu，效率低。作为注释留在这里供参考
    #cuda.memcpy_htod(d_buffers[0], input_data.cpu().detach().numpy())
    trt.execute(d_buffers, i2shape)
    
    cuda.memcpy_dtoh(output1_data_trt, d_buffers[1])
    cuda.memcpy_dtoh(output2_data_trt, d_buffers[2])
    cuda.memcpy_dtoh(output3_data_trt, d_buffers[3])
    # print(np.mean(output1_data_trt))
    
    print(output1_data_trt)
    print(output2_data_trt)
    print(output3_data_trt)

    # cuda.Context.synchronize()
    # t0 = time.time()
    # for i in range(nRound):
    #     trt.execute(d_buffers, i2shape)
    # cuda.Context.synchronize()
    # time_trt = (time.time() - t0) / nRound
    # print('TensorRT time:', time_trt)

    # print('Speedup:', time_pytorch / time_trt)
    # print('Average diff percentage1:', np.mean(np.abs(output1_data_pytorch - output1_data_trt) / np.abs(output1_data_pytorch)))
    # print('Average diff percentage2:', np.mean(np.abs(output2_data_pytorch - output2_data_trt) / np.abs(output2_data_pytorch)))
    # print('Average diff percentage3:', np.mean(np.abs(output3_data_pytorch - output3_data_trt) / np.abs(output3_data_pytorch)))
