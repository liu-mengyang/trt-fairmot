from torch import nn

from Config import BN_MOMENTUM
from DCN.dcn_v2 import DCN

from TRT_Constructor import TRT_Constructor

# Warpped DCN
class DeformConv(nn.Module):
    def __init__(self, chi, cho, test=False):
        super(DeformConv, self).__init__()
        self.test = test
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        if self.test == True:
            x = x.cpu()
        x = self.conv(x)
        if self.test == True:
            x = x.cpu()
        x = self.actf(x)
        return x

    def TRT_export(self, constructor: TRT_Constructor, x):
        x = self.conv.TRT_export(constructor, x)
        # print(x.shape)
        x = constructor.BatchNorm2d(self.actf[0], x)
        x = constructor.ReLU(self.actf[1], x)
        return x

if __name__ == '__main__':
    # 以下为TensorRT对比测试代码
    from test_dcn import test_fun, input_channel, output_channel
    m = DeformConv(input_channel, output_channel, test=True) # Pytorch构建的模型
    test_fun(m)