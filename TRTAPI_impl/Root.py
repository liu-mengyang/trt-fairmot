import torch
from torch import nn

from Config import BN_MOMENTUM

from TRT_Constructor import TRT_Constructor

# Aggregation Node
class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

    def TRT_export(self, constructor: TRT_Constructor, *x):
        children = x
        x = constructor.Conv2d(self.conv, constructor.Concat(x))
        x = constructor.BatchNorm2d(self.bn, x)
        if self.residual:
            x = constructor.Add(x, children[0])
        x = constructor.ReLU(self.relu, x)
        return x

if __name__ == '__main__':
    # 以下为TensorRT对比测试代码
    from test_fun import test_fun, input_channel, output_channel
    m = Root(input_channel, output_channel, kernel_size=1, residual=False) # Pytorch构建的模型
    test_fun(m)