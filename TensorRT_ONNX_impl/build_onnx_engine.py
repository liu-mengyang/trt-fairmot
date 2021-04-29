import torch

from build_model import build_fairmot, load_model

input_data = torch.randn(1, 3, 608, 1088, dtype=torch.float32, device='cuda')

net = build_fairmot()
model = load_model(net, "../weights/fairmot_dla34.pth")
model = model.to(torch.device('cuda'))
model.eval()

input_names = ['input']
output_names = ['output1', 'output2', 'output3']
torch.onnx.export(net, input_data, 'fairmot.onnx', input_names=input_names, output_names=output_names,
                  verbose=True, opset_version=12,
                  dynamic_axes={"input":{0: "batch_size"}, 
                                "output1":{0: "batch_size"}, 
                                "output2":{0: "batch_size"}, 
                                "output3":{0: "batch_size"}},
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)


