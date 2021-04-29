#include "DCNv2PluginDyn.h"
#include <torch/script.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


void modulated_deform_conv_forward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor output, at::Tensor columns,
    int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const bool with_bias);

int DCNv2PluginDyn::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) 
{
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    nvinfer1::Dims inputDim = inputDesc[0].dims;
    nvinfer1::Dims outputDim = m.outputDim;

    const int nBatch = inputDim.d[0];
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
    
    at::Tensor input = torch::from_blob(const_cast<void *>(inputs[0]), {nBatch, channels_in, height_in, width_in}, options);
    at::Tensor offset = torch::from_blob(const_cast<void *>(inputs[1]), {nBatch, 2 * kernel_h * kernel_w * deformable_group, height_in, width_in}, options);
    at::Tensor mask = torch::from_blob(const_cast<void *>(inputs[2]), {nBatch, kernel_h * kernel_w * deformable_group, height_in, width_in}, options);
    at::Tensor weight = torch::from_blob(const_cast<void *>(inputs[3]), {channels_out, channels_in, kernel_h, kernel_w}, options);
    at::Tensor bias = torch::from_blob(const_cast<void *>(inputs[4]), {channels_out}, options);
    at::Tensor output = torch::from_blob(outputs[0], {nBatch, channels_out, height_out, width_out}, options);
    at::Tensor ones = at::Tensor();
    at::Tensor columns = at::Tensor();
    modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output, columns, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, 1, deformable_group, true);
    return 0;
}

nvinfer1::PluginFieldCollection DCNv2PluginDynCreator::fc;
REGISTER_TENSORRT_PLUGIN(DCNv2PluginDynCreator);