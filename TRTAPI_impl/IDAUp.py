import math
from torch import nn

from DeformConv import DeformConv
from IDA_pare import IDA_pare
from TRT_Constructor import TRT_Constructor

# Iterative Deep Aggregation Upsample
class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f, test=False):
        super(IDAUp, self).__init__()
        self.test = test
        for i in range(1, len(channels)):
            # c = channels[i]
            # f = int(up_f[i])
            # ida_pare = IDA_pare(o, c, f, test=test)
            # setattr(self, 'ida_pare_' + str(i), ida_pare)
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])
        # return layers[-1]

    def TRT_export(self, constructor: TRT_Constructor, layers, startp, endp):
        for i in range(startp + 1, endp):
            layers[i] = getattr(self, 'proj_' + str(i - startp)).TRT_export(constructor, layers[i])
            layers[i] = constructor.DeConv2d(getattr(self, 'up_' + str(i - startp)), layers[i])
            x = constructor.Add(layers[i], layers[i - 1])
            layers[i] = getattr(self, 'node_' + str(i - startp)).TRT_export(constructor, x)
        # return layers[-1]

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