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

class DCNv2Plugin: public nvinfer1::IPluginV2IOExt {
public:
    DCNv2Plugin() {
        dlopen("/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda.so", RTLD_LAZY);
        dlopen("/usr/local/lib/python3.8/dist-packages/torch/lib/libc10_cuda.so", RTLD_LAZY);
        // m.outputDim.d[0] = output_channel;
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

    nvinfer1::IPluginV2IOExt* clone() const override {
        return new DCNv2Plugin(&m, sizeof(m));
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInput, int nbOutputs) const override {
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

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* pInputDim, int nInputDim) override {
        assert(index == 0);
        assert(nInputDim == 5);
        auto outputDim = pInputDim[0];
        outputDim.d[0] = pInputDim[3].d[0];
        std::cout << "outputDim: " << to_string(outputDim) << std::endl;
        return outputDim;
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {
        return nvinfer1::DataType::kFLOAT;
    }
    
    virtual void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput) override {
        assert(nbInput == 5);
        assert(nbOutput == 1);
        m.inputDim = in[0].dims;
        m.outputDim = in[0].dims;
        m.outputDim.d[0] = in[3].dims.d[0];
        m.outputDim.d[1] = m.inputDim.d[1];
        m.outputDim.d[2] = m.inputDim.d[2];
        // out[0].dims = m.outputDim;
        std::cout << "configurePlugin type=" << (int)out[0].type << ", inputDim=" << to_string(m.inputDim) << ", outputDim=" << to_string(m.outputDim) << std::endl;
    }

    size_t getWorkspaceSize(int nMaxBatchSize) const override {return 0;}
    int enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "DCNv2Plugin";}
    const char* getPluginVersion() const override {return "1";}
    bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {return false;}
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


class DCNv2PluginCreator : public nvinfer1::IPluginCreator {
public:
    static nvinfer1::PluginFieldCollection fc;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    DCNv2PluginCreator() {
        // mPluginAttributes.emplace_back(nvinfer1::PluginField("out_channel", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

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
        // std::cout << __FUNCTION__ << std::endl;
        return &fc;
    } 
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        // std::cout << __FUNCTION__ << std::endl;
        // int out_channel = 0;
        // for (int i = 0; i < fc->nbFields; i++) {
        //     if (!strcmp(fc->fields[i].name, "out_channel")) {
        //         out_channel = *(static_cast<const int*>(fc->fields[i].data));
        //     }
        // }
        return new DCNv2Plugin();
    }
    
};