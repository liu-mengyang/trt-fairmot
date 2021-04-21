SHELL=/bin/bash
CUDA_PATH = /usr/local/cuda
TRT_PATH = /usr/local/tensorrt7.2-cuda11.1
TORCH_PATH = /usr/local/lib/python3.8/dist-packages/torch
GENCODE = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -DEBUG -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ 
INCLUDES := -I../include -isystem $(CUDA_PATH)/include -isystem $(TRT_PATH)/include -isystem $(TORCH_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(TORCH_PATH)/lib/ -L$(TRT_PATH)/lib -Wl,-rpath=$(CUDA_PATH)/lib64
LDFLAGS += -lcudart -lnvinfer -lnvonnxparser -ldl -lpthread -lcuda -ltorch -lc10 -ltorch_cuda -lc10_cuda -ltorch_cpu -ltorch_python

OBJ_DIR = obj
OBJ = $(shell find $(OBJ_DIR) -name *.o 2>/dev/null)
DEP = $(OBJ:.o=.d)

BIN = AppBasic
SO = DCNv2Plugin.so

all: $(BIN) $(SO)

AppBasic: $(addprefix $(OBJ_DIR)/, AppBasic.o)

DCNv2Plugin.so: $(addprefix $(OBJ_DIR)/, DCNv2Plugin.o dcn_v2_cuda.o dcn_v2_im2col_cuda.o)

-include $(DEP)

clean:
	rm -rf $(BIN) $(SO) $(DEP) $(OBJ_DIR) out.trt int8_cache.*

VPATH = ../include:../samples:../plugins

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC $(GENCODE) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<


$(SO):
	$(GCC) $(CCFLAGS) -fPIC -shared -o $@ $+ $(LDFLAGS)

$(CUDA_BIN):
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LDFLAGS)

$(BIN):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


