import torch
import time
import numpy as np
import tensorrt as trt

from build_model import build_fairmot, load_model
from trt_lite2 import TrtLite
from TRT_Constructor import TRT_Constructor

torch.cuda.init()

with torch.no_grad():
    net = build_fairmot()
    model = load_model(net, "../weights/fairmot_dla34.pth")
    model = model.to(torch.device('cuda'))
    model.eval()
    
    input_channel = 3
    h = 608
    w = 1088
    input_data = torch.randn(3, 608, 1088, dtype=torch.float32, device='cuda')
    # input_data = np.arange(1*input_channel*h*w,
    #                     dtype=np.float32).reshape(input_channel, h, w)/input_channel/h/w*255+10
    output_torch = model(torch.tensor(input_data.reshape(1, input_channel, h, w), device='cuda'))

    # 10 rounds of PyTorch FairMOT
    nRound = 10
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        model(torch.tensor(input_data.reshape(1, input_channel, h, w), device='cuda'))
    torch.cuda.synchronize()
    time_pytorch = (time.time() - t0) / nRound
    print('PyTorch time:', time_pytorch)

    batch_size = 1

    trt = TrtLite(engine_file_path="fairmot.trt")
    io_info = trt.get_io_info(batch_size)
    if io_info is None:
        exit()
    print(io_info)
    h_buffers = trt.allocate_io_buffers(batch_size, False)
    d_buffers = trt.allocate_io_buffers(batch_size, True)

    #利用PyTorch和PyCUDA的interop，保留数据始终在显存上
    # cuda.memcpy_dtod(d_buffers[0], PyTorchTensorHolder(input_data), input_data.nelement() * input_data.element_size())
    # output1_data_trt = np.zeros(())
    #下面一行的作用跟上一行一样，不过它是把数据拷到cpu再拷回gpu，效率低。作为注释留在这里供参考
    # cuda.memcpy_htod(d_buffers[0], input_data)
    d_buffers[0] = input_data
    torch.cuda.synchronize()
    trt.execute([t.data_ptr() for t in d_buffers], batch_size)
    torch.cuda.synchronize()
    # cuda.Context.synchronize()
    for i, info in enumerate(io_info):
        if not info[1]:
            h_buffers[i] = d_buffers[i].cpu().numpy()
    print("execute successfully")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute([t.data_ptr() for t in d_buffers], batch_size)
    torch.cuda.synchronize()
    time_trt = (time.time() - t0) / nRound
    print('TensorRT time:', time_trt)

    print('Speedup:', time_pytorch / time_trt)
    diff1 = np.abs(output_torch[0].cpu().numpy() - h_buffers[1])
    diff2 = np.abs(output_torch[1].cpu().numpy() - h_buffers[2])
    diff3 = np.abs(output_torch[2].cpu().numpy() - h_buffers[3])
    print("pytorch output")
    print(output_torch[0])
    print("trt output")
    print(h_buffers[1])
    print('1 Average diff percentage:', np.mean(diff1 / np.abs(output_torch[0].cpu().numpy())))
    print("diff1:")
    print(diff1)
    print("pytorch output")
    print(output_torch[1])
    print("trt output")
    print(h_buffers[2])
    print('2 Average diff percentage:', np.mean(diff2 / np.abs(output_torch[1].cpu().numpy())))
    print("diff2:")
    print(diff2)
    print("pytorch output")
    print(output_torch[2])
    print("trt output")
    print(h_buffers[3])
    print('3 Average diff percentage:', np.mean(diff3 / np.abs(output_torch[2].cpu().numpy())))
    print("diff3:")
    print(diff3)
