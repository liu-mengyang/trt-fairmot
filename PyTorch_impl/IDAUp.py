import math
from torch import nn

from DeformConv import DeformConv
from IDA_pare import IDA_pare
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

# Iterative Deep Aggregation Upsample
class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f, test=True):
        super(IDAUp, self).__init__()
        self.test = test
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            ida_pare = IDA_pare(o, c, f, test=test)
            setattr(self, 'ida_pare_' + str(i), ida_pare)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            ida_pare = getattr(self, 'ida_pare_' + str(i - startp))
            layers[i] = ida_pare(layers[i], layers[i-1])
        return layers[-1]

    def TRT_export(self, constructor: TRT_Constructor, layers, startp, endp):
        for i in range(startp + 1, endp):
            layers[i] = getattr(self, 'ida_pare_{}'.format(i - startp)).TRT_export(constructor, layers[i], layers[i-1])
        y = layers[-1]
        return y
        # return layers

if __name__ == '__main__':
    # 以下为TensorRT对比测试代码
    from test_ida import test_fun
    channels = [16, 32, 64, 128, 256, 512]
    first_level = 2
    last_level = 5
    out_channel = channels[first_level]
    m = IDAUp(out_channel, channels[first_level:last_level], 
                            [2 ** i for i in range(last_level - first_level)],
                            test=True) # Pytorch构建的模型
    test_fun(m)