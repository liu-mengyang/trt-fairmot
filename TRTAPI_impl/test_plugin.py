#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import glob
import torch
from functools import reduce
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
from trt_lite import TrtLite
from torch.utils.cpp_extension import load


from TRT_Constructor import TRT_Constructor

np.set_printoptions(threshold=np.inf)

ctypes.cdll.LoadLibrary('./build/DCNv2Plugin.so')
torch.cuda.init()

input_channel = 64
output_channel = 64
h = int(608 / 4)
w = int(1088 / 4)


filename = "dcn_v2_cuda.cpp" if torch.cuda.is_available() else "dcn_v2.cpp"

extensions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DCN/src")
main_file = glob.glob(os.path.join(extensions_dir, filename))
source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
source_cuda = (
    glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    if torch.cuda.is_available()
    else []
)


_backend = load(
    name="DCNv2_" + ("gpu" if torch.cuda.is_available() else "cpu"),
    sources=main_file + source_cpu + source_cuda,
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
)

batch_size = 1
input_shape = (batch_size, 64, 608, 1088)
n = reduce(lambda x, y: x * y, input_shape)
data_input = np.asarray(range(n), dtype=np.float32).reshape(input_shape)
data_offset = np.arange(18*608*1088, dtype=np.float32).reshape(1, 18, 608, 1088)
data_mask = np.arange(9*608*1088, dtype=np.float32).reshape(1, 9,608,1088)
data_weight = np.arange(64*64*3*3, dtype=np.float32).reshape(64,64,3,3)
data_bias = np.arange(64, dtype=np.float32)

def get_plugin_creator(plugin_name):
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

def build_engine(builder, input_shape):
    print("building")
    plugin_creator = get_plugin_creator('DCNv2Plugin')
    print("plugin created")
    if plugin_creator == None:
        print('Plugin not found. Exiting')
        exit()

    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 20
    network = builder.create_network()
    print("network created")
    tensor = network.add_input('data', trt.DataType.FLOAT, input_shape)
    offset = network.add_input('offset', trt.DataType.FLOAT, (18, 608, 1088))
    mask = network.add_input('mask', trt.DataType.FLOAT, (9,608,1088))
    weight = network.add_input('weight', trt.DataType.FLOAT, (64,64,3,3))
    bias = network.add_input('bias', trt.DataType.FLOAT, (64,))
    print("input added")
    
    fc = trt.PluginFieldCollection()
    fc.append(trt.PluginField("out_channel", np.array([64], np.int32)))
    print("fc appendded")
    y = network.add_plugin_v2([tensor, offset, mask, weight, bias], 
        plugin_creator.create_plugin('DCNv2Plugin', fc))
    print("plugin added")
    network.mark_output(y.get_output(0))

    return builder.build_cuda_engine(network)

def run_engine():
    print("run")
    batch_size = 1
    input_shape = (batch_size, 64, 608, 1088)
    n = reduce(lambda x, y: x * y, input_shape)
    
    
    trt = TrtLite(build_engine, (input_shape[1:],))
    trt.print_info()

    d_buffers = trt.allocate_io_buffers(batch_size, True)
    h_buffers = trt.allocate_io_buffers(batch_size, False)

    h_buffers[0][:] = data_input.reshape(input_shape[1:])
    h_buffers[1][:] = data_offset.reshape(18, 608, 1088)
    h_buffers[2][:] = data_mask.reshape(9,608,1088)
    h_buffers[3][:] = data_weight.reshape(64,64,3,3)
    h_buffers[4][:] = data_bias
    # h_buffers[5][:] = np.zeros(input_shape[1:], dtype=np.float32)
    
    print("execute")

    io_info = trt.get_io_info(batch_size)
    if io_info is None:
        return
    print(io_info)

    for i, info in enumerate(io_info):
        if info[1]:
            cuda.memcpy_htod(d_buffers[i], h_buffers[i])
    
    trt.execute(d_buffers, batch_size)
    print("output:")
    for i, info in enumerate(io_info):
        if not info[1]:
            cuda.memcpy_dtoh(h_buffers[i], d_buffers[i])
            print(np.mean(h_buffers[i]))
            print("got")

def run_torch():
    
    np.random.shuffle(data_input)
    np.random.shuffle(data_offset)
    np.random.shuffle(data_mask)
    np.random.shuffle(data_weight)
    np.random.shuffle(data_bias)
    input = torch.tensor(data_input)
    offset = torch.tensor(data_offset)
    mask = torch.tensor(data_mask)
    weight = torch.tensor(data_weight)
    bias = torch.tensor(data_bias)

    output = _backend.dcn_v2_forward(
            input.cuda(),
            weight.cuda(),
            bias.cuda(),
            offset.cuda(),
            mask.cuda(),
            3,
            3,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        )
    return output

if __name__ == '__main__':
    run_engine()
    output = run_torch()
    print("torch:")
    print(torch.mean(output))