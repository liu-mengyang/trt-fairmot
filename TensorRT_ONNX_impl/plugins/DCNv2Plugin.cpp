#include "DCNv2Plugin.h"
// #include <torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh> 
// #include <THC/THCDeviceUtils.cuh>

at::Tensor
dcn_v2_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int deformable_group);

int DCNv2Plugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                         const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) 
{
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    nvinfer1::Dims inputDim = inputDesc[0].dims;
    nvinfer1::Dims outputDim = outputDesc[0].dims;
    
    // const int batch = inputDim.d[0];
    const int channels_in = inputDim.d[1];
    const int height_in = inputDim.d[2];
    const int width_in = inputDim.d[3];
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    const int dilation_h = 1;
    const int dilation_w = 1;
    const int deformable_group = 1;

    const int channels_out = outputDim.d[1];
    const int height_out = outputDim.d[2];
    const int width_out = outputDim.d[3];

    // std::cout << "input shape: (" << channels_in << "," << height_in << "," << width_in << ")" << std::endl;
    // std::cout << "output shape: (" << channels_out << "," << height_out << "," << width_out << ")" << std::endl;
    
    at::Tensor input = at::from_blob(const_cast<void *>(inputs[0]), {1, channels_in, height_in, width_in}, options);
    at::Tensor offset = at::from_blob(const_cast<void *>(inputs[1]), {1, 2 * kernel_h * kernel_w * deformable_group, height_in, width_in}, options);
    at::Tensor mask = at::from_blob(const_cast<void *>(inputs[2]), {1, kernel_h * kernel_w * deformable_group, height_in, width_in}, options);
    at::Tensor weight = at::from_blob(const_cast<void *>(inputs[3]), {channels_out, channels_in, kernel_h, kernel_w}, options);
    at::Tensor bias = at::from_blob(const_cast<void *>(inputs[4]), {channels_out}, options);
    at::Tensor output = at::from_blob(const_cast<void *>(outputs[0]), {1, channels_out, height_out, width_out}, options);

    // std::cout << input << std::endl;

    output = dcn_v2_cuda_forward(input, weight, bias, offset, mask, 
                                 kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                                 dilation_h, dilation_w, deformable_group);
    std::cout << at::mean(output) << std::endl;
    return 0;
}

nvinfer1::PluginFieldCollection DCNv2PluginCreator::fc;
REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);