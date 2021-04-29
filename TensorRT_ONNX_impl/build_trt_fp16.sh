trtexec --onnx=fairmot_plugin.onnx\
        --explicitBatch \
        --minShapes="input":1x3x608x1088\
        --optShapes="input":8x3x608x1088\
        --maxShapes="input":16x3x608x1088\
        --shapes="input":1x3x608x1088\
        --saveEngine=fairmot_fp16.trt\
        --plugins=./build/DCNv2PluginDyn.so\
        --verbose\
        --fp16
