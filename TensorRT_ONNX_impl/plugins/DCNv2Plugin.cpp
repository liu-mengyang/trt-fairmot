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
    at::Tensor input = at::from_blob(const_cast<void *>(inputs[0]), {1, inputDim.d[1], inputDim.d[2], inputDim.d[3]}, options);
    at::Tensor offset = at::from_blob(const_cast<void *>(inputs[1]), {1, 2 * 9 * 1, inputDim.d[2], inputDim.d[3]}, options);
    at::Tensor mask = at::from_blob(const_cast<void *>(inputs[2]), {1, 9 * 1, inputDim.d[2], inputDim.d[3]}, options);
    at::Tensor weight = at::from_blob(const_cast<void *>(inputs[3]), {outputDim.d[1], inputDim.d[1], 3, 3}, options);
    at::Tensor bias = at::from_blob(const_cast<void *>(inputs[4]), {outputDim.d[1]}, options);
    at::Tensor output = at::from_blob(const_cast<void *>(outputs[0]), {1, outputDim.d[1], outputDim.d[2], outputDim.d[3]}, options);

    std::cout << input << std::endl;

    const int batch = 1;
    const int channels = inputDim.d[1];
    const int height = inputDim.d[2];
    const int width = inputDim.d[3];

    const int channels_out = outputDim.d[1];
    // const int channels_kernel = weight.size(1);
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    const int dilation_h = 1;
    const int dilation_w = 1;
    const int deformable_group = 1;

    // AT_ASSERTM(channels == channels_kernel,
    //            "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    // const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    // const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int height_out = outputDim.d[2];
    const int width_out = outputDim.d[3];

    at::Tensor ones = at::ones({batch, bias.sizes()[0], height_out, width_out}, options);
    at::Tensor columns = at::empty({batch, channels * kernel_h * kernel_w, 1 * height_out * width_out}, options);
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
                                 batch, channels, height, width,
                                 height_out, width_out, kernel_h, kernel_w,
                                 pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                 deformable_group,
                                 columns.data_ptr<scalar_t>());
    

    // Scale columns and add to output
    // torch implementation
    auto weight_flat = weight.view({channels_out, channels * kernel_h * kernel_w});
    auto product = at::matmul(weight_flat, columns);
    output = at::add(output, product.view({batch, channels_out, height_out, width_out}));

    // std::cout << at::mean(output) << std::endl;

    return 0;
}

nvinfer1::PluginFieldCollection DCNv2PluginCreator::fc;
REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);