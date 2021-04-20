#include "DCNv2Plugin.h"
// #include <torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh> 
// #include <THC/THCDeviceUtils.cuh>

int DCNv2Plugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
    using scalar_t = float;
    auto options = at::TensorOptions().device(c10::kCUDA).dtype(c10::kFloat);
    nvinfer1::Dims inputDim = inputDesc[0].dims;
    nvinfer1::Dims outputDim = outputDesc[0].dims;
    
    const int batch = inputDim.d[0];
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
    // const int height_out = (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    // const int width_out = (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    at::Tensor input = at::from_blob(const_cast<void *>(inputs[0]), {1, channels_in, height_in, width_in}, options);
    at::Tensor offset = at::from_blob(const_cast<void *>(inputs[1]), {1, 2 * kernel_h * kernel_w * deformable_group, height_in, width_in}, options);
    at::Tensor mask = at::from_blob(const_cast<void *>(inputs[2]), {1, kernel_h * kernel_w * deformable_group, height_in, width_in}, options);
    at::Tensor weight = at::from_blob(const_cast<void *>(inputs[3]), {channels_out, channels_in, kernel_h, kernel_w}, options);
    at::Tensor bias = at::from_blob(const_cast<void *>(inputs[4]), {channels_out}, options);
    at::Tensor output = at::from_blob(const_cast<void *>(outputs[0]), {1, channels_out, height_out, width_out}, options);

    // std::cout << input << std::endl;
    // std::cout << offset.size(0) << " " << offset.size(1) << " " << offset.size(2) << " " << offset.size(3) << std::endl;

    at::Tensor ones = at::ones({batch, channels_out, height_out, width_out}, options);
    at::Tensor columns = at::empty({batch, channels_in * kernel_h * kernel_w, 1 * height_out * width_out}, options);
    output = output.view({batch, channels_out, height_out, width_out}).zero_();

    // Add biases to output tensor
    // torch implementation
    auto ones_T = at::transpose(ones.contiguous(), 3, 1);
    ones_T = at::mul(ones_T, bias.contiguous());
    ones_T = at::transpose(ones_T, 3, 1);
    output = at::add(output, ones_T);

    modulated_deformable_im2col_cuda(stream,
                                 input.data_ptr<scalar_t>(),
                                 offset.data_ptr<scalar_t>(),
                                 mask.data_ptr<scalar_t>(),
                                 batch, channels_in, height_in, width_in,
                                 height_out, width_out, kernel_h, kernel_w,
                                 pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                 deformable_group,
                                 columns.data_ptr<scalar_t>());
    

    // Scale columns and add to output
    // torch implementation
    auto weight_flat = weight.view({channels_out, channels_in * kernel_h * kernel_w});
    auto product = at::matmul(weight_flat, columns);
    output = at::add(output, product.view({batch, channels_out, height_out, width_out}));

    // std::cout << at::mean(output) << std::endl;
    // std::cout << output << std::endl;

    return 0;
}

nvinfer1::PluginFieldCollection DCNv2PluginCreator::fc;
REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);