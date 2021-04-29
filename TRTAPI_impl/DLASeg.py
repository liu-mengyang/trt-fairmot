from torch import nn
import numpy as np
import copy

from BasicBlock import BasicBlock
from DLA import DLA, _TRT_make_conv_level
from IDAUp import IDAUp
from DLAUp import DLAUp
from TRT_Constructor import TRT_Constructor

from TRT_Constructor import TRT_Constructor

# DLA-34
def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Deep Layer Segmention
class DLASeg(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, test=False):
        super(DLASeg, self).__init__()
        self.test = test
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = dla34(pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales, test=self.test)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)],
                            test=self.test)
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]

    def TRT_export(self, constructor: TRT_Constructor, x):
        x = self.base.TRT_export(constructor, x)
        x = self.dla_up.TRT_export(constructor, x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])
        self.ida_up.TRT_export(constructor, y, 0, len(y))

        z = []
        for head in self.heads:
            z_pare = _TRT_make_head(constructor, self.__getattr__(head), y[-1])
            z.append(z_pare)
        return z

def _TRT_make_head(constructor: TRT_Constructor, head, x):
    x = constructor.Conv2d(head[0], x)
    x = constructor.ReLU(head[1], x)
    x = constructor.Conv2d(head[2], x)
    return x

if __name__ == '__main__':
    # 以下为TensorRT对比测试代码
    from test_dlaseg import test_fun
    heads = {'hm': 1,
        'wh': 4,
        'id': 128}
    down_ratio = 4
    head_conv = 256
    num_layers = 34
    m = DLASeg(heads,
                     pretrained=True,
                     down_ratio=down_ratio,
                     final_kernel=1,
                     last_level=5,
                     head_conv=head_conv, test=True) # Pytorch构建的模型
    test_fun(m)
