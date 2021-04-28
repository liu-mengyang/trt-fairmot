import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("fairmot.onnx"))

for node in graph.nodes:
    if node.op == 'DCNv2':
        node.name = "DCNv2Plugin"
        node.op = "DCNv2Plugin"

graph.cleanup()
onnx.save(gs.export_onnx(graph), "fairmot_plugin.onnx")