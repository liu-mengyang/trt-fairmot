import math
from torch import nn

from DeformConv import DeformConv
from TRT_Constructor import TRT_Constructor

# other used functions
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class IDA_pare(nn.Module):
    def __init__(self, o, c, f, test=False):
        super(IDA_pare, self).__init__()
        self.test = test
        self.proj = DeformConv(c, o, test=self.test)
        self.node = DeformConv(o, o, test=self.test)
        self.up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                padding=f // 2, output_padding=0,
                                groups=o, bias=False)
        fill_up_weights(self.up)
        
    def forward(self, x, pre_x=None):
        x = self.proj(x)
        x = self.up(x)
        if pre_x is None:
            pre_x = x
        x = self.node(x + pre_x)
        return x

    def TRT_export(self, constructor: TRT_Constructor, x, pre_x=None):
        x = self.proj.TRT_export(constructor, x)
        x = constructor.DeConv2d(self.up, x)
        if pre_x is None:
            pre_x = x
        x = constructor.Add(x, pre_x)
        x = self.node.TRT_export(constructor, x)
        return x

if __name__ == '__main__':
    # 以下为TensorRT对比测试代码
    from test_dcn import test_fun, input_channel, output_channel
    m = IDA_pare(output_channel, input_channel, 2, test=True) # Pytorch构建的模型
    test_fun(m)