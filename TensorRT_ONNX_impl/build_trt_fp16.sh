trtexec --onnx=fairmot_plugin.onnx --saveEngine=fairmot_fp16.trt --plugins=./build/DCNv2PluginDyn.so --verbose --fp16
