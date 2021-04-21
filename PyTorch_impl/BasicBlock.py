from torch import nn

from Config import BN_MOMENTUM

from TRT_Constructor import TRT_Constructor

# Residual Block
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

    def TRT_export(self, constructor: TRT_Constructor, x, residual=None):
        if residual is None:
            residual = x

        out = constructor.Conv2d(self.conv1, x)
        out = constructor.BatchNorm2d(self.bn1, out)
        out = constructor.ReLU(self.relu, out)

        out = constructor.Conv2d(self.conv2, out)
        out = constructor.BatchNorm2d(self.bn2, out)

        out = constructor.Add(out, residual)
        out = constructor.ReLU(self.relu, out)
        return out

if __name__ == '__main__':
    # 以下为TensorRT对比测试代码
    from test_fun import test_fun, input_channel, output_channel
    m = BasicBlock(input_channel, output_channel) # Pytorch构建的模型
    test_fun(m)