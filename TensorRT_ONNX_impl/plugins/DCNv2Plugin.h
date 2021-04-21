#include <NvInfer.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <cstring>
// #include "dcn_v2_im2col_cuda.h"

// using namespace std;

class DCNv2Plugin: public nvinfer1::IPluginV2DynamicExt {
public:

    DCNv2Plugin(int output_channel) {
        dlopen("/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda.so", RTLD_LAZY);
        m.out_channel = output_channel;
    }

    DCNv2Plugin(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }

    virtual size_t getSerializationSize() const override {
        return sizeof(m);
    }

    virtual void serialize(void *buffer) const override {
        memcpy(buffer, &m, sizeof(m));
    }

    nvinfer1::IPluginV2DynamicExt* clone() const override {
        return new DCNv2Plugin(&m, sizeof(m));
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInput, int nbOutputs) override {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    int getNbOutputs() const override {
        return 1;
    }

    nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* pInputDim, int nInputDim, nvinfer1::IExprBuilder &exprBuilder) override {
        nvinfer1::DimsExprs output(pInputDim[0]);
        auto input_h = output.d[2]->getConstantValue();
        auto input_w = output.d[3]->getConstantValue();
        auto output_h = (input_h + 2 * 1 - (1 * (3 - 1) + 1)) / 1 + 1;
        auto output_w = (input_w + 2 * 1 - (1 * (3 - 1) + 1)) / 1 + 1;
        output.d[1] = exprBuilder.constant(m.out_channel);
        output.d[2] = exprBuilder.constant(output_h);
        output.d[3] = exprBuilder.constant(output_w);
        return output;
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {
        return nvinfer1::DataType::kFLOAT;
    }
    
    virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInput, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutput) override {
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const override {return 0;}
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "DCNv2Plugin";}
    const char* getPluginVersion() const override {return "1";}
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) {}
    void detachFromContext() {}

private:
    struct {
        int out_channel;
    } m;
    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;
};


class DCNv2PluginCreator : public nvinfer1::IPluginCreator {
public:
    static nvinfer1::PluginFieldCollection fc;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    DCNv2PluginCreator() {
        mPluginAttributes.emplace_back(nvinfer1::PluginField("out_channel", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

        fc.nbFields = mPluginAttributes.size();
        fc.fields = mPluginAttributes.data();
    }
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new DCNv2Plugin(serialData, serialLength);
    }

    const char* getPluginName() const override {return "DCNv2Plugin";}
    const char* getPluginVersion() const override {return "1";}

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}

    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return &fc;
    } 
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        std::cout << __FUNCTION__ << std::endl;
        int out_channel = 0;
        for (int i = 0; i < fc->nbFields; i++) {
            if (!strcmp(fc->fields[i].name, "out_channel")) {
                out_channel = *(static_cast<const int*>(fc->fields[i].data));
            }
        }
        return new DCNv2Plugin({out_channel});
    }
    
};