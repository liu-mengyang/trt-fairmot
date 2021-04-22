from torch import nn
import numpy as np

from IDAUp import IDAUp

# Deep Layer Aggregation Upsample
class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
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
                          scales[j:] // scales[j]))
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
        for i in range(len(layers) - self.startp - 1):
            getattr(self, 'ida_{}'.format(i)).TRT_export(constructor, layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out
