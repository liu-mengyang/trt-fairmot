#include <NvInfer.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <assert.h>

inline std::string to_string(nvinfer1::Dims const &dim) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < dim.nbDims; i++) {
        oss << dim.d[i] << ", ";
    }
    oss << ")";
    return oss.str();
}

class DCNv2PluginDyn: public nvinfer1::IPluginV2DynamicExt {
public:
    DCNv2PluginDyn() {
        dlopen("/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda.so", RTLD_LAZY);
        dlopen("/usr/local/lib/python3.8/dist-packages/torch/lib/libc10_cuda.so", RTLD_LAZY);
    }

    DCNv2PluginDyn(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }

    virtual size_t getSerializationSize() const override {
        return sizeof(m);
    }

    virtual void serialize(void *buffer) const override {
        memcpy(buffer, &m, sizeof(m));
    }

    nvinfer1::IPluginV2DynamicExt* clone() const override {
        return new DCNv2PluginDyn(&m, sizeof(m));
    }
    
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInput, int nbOutputs) override {
        assert(nbInput == 5);
        assert(nbOutputs == 1);
        assert(pos < (nbInput + nbOutputs));
        const auto *in = inOut;
        const auto *out = inOut + nbInput;
        if (pos == nbInput) {
            return out[0].type == in[0].type &&
                    out[0].format == nvinfer1::TensorFormat::kLINEAR;
        } else {
            return in[pos].type == nvinfer1::DataType::kFLOAT &&
                    in[pos].format == nvinfer1::TensorFormat::kLINEAR;
        }
    }

    int getNbOutputs() const override {
        return 1;
    }

    nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* pInputDim, int nInputDim, nvinfer1::IExprBuilder &exprBuilder) override {
        assert(index == 0);
        assert(nInputDim == 5);
        auto outputDim = pInputDim[0];
        outputDim.d[1] = pInputDim[3].d[0];
        return outputDim;
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {
        return nvinfer1::DataType::kFLOAT;
    }
    
    virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInput, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutput) override {
        assert(nbInput == 5);
        assert(nbOutput == 1);
        m.inputDim = in[0].desc.dims;
        m.outputDim = in[0].desc.dims;
        m.outputDim.d[1] = in[3].desc.dims.d[0];
        // std::cout << "configurePlugin type=" << (int)out[0].desc.type << ", inputDim=" << to_string(m.inputDim) << ", outputDim=" << to_string(m.outputDim) << std::endl;
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
        nvinfer1::Dims inputDim;
        nvinfer1::Dims outputDim;
    } m;
    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;
};


class DCNv2PluginDynCreator : public nvinfer1::IPluginCreator {
public:
    static nvinfer1::PluginFieldCollection fc;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    DCNv2PluginDynCreator() {
        fc.nbFields = mPluginAttributes.size();
        fc.fields = mPluginAttributes.data();
    }
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new DCNv2PluginDyn(serialData, serialLength);
    }

    const char* getPluginName() const override {return "DCNv2Plugin";}
    const char* getPluginVersion() const override {return "1";}

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}

    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        // std::cout << __FUNCTION__ << std::endl;
        return &fc;
    } 
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        // std::cout << __FUNCTION__ << std::endl;
        return new DCNv2PluginDyn();
    }
    
};