from torch import nn
import numpy as np

from IDAUp import IDAUp
from TRT_Constructor import TRT_Constructor

# Deep Layer Aggregation Upsample
class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, test=False):
        super(DLAUp, self).__init__()
        self.test = test
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j], test=self.test))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out

    def TRT_export(self, constructor: TRT_Constructor, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - 1):
            y = getattr(self, 'ida_{}'.format(i)).TRT_export(constructor, layers, len(layers) - i - 2, len(layers))
            out.insert(0, y)
        return out


if __name__ == '__main__':
    # 以下为TensorRT对比测试代码
    from test_dlaup import test_fun
    channels = [16, 32, 64, 128, 256, 512]
    first_level = 2
    out_channel = channels[first_level]
    scales = [2 ** i for i in range(len(channels[first_level:]))]
    m = DLAUp(first_level, channels[first_level:], scales, test=True)
    test_fun(m)