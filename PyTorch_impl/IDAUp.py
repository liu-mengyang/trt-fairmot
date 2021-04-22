import math
from torch import nn

from DeformConv import DeformConv

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

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
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
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            layers[i] = node(layers[i] + layers[i - 1])

    def TRT_export(self, constructor: TRT_Constructor, layers, startp, endp):
        for i in range(startp + 1, endp):
            layers[i] = getattr(self, 'proj_{}'.format(i - startp)).TRT_export(constructor, layers[i])
            layers[i] = constructor.DeConv2d(getattr(self, 'up_{}'.format(i - startp)), layers[i])
            layers[i] = getattr(self, 'node_{}'.format(i - startp)).TRT_export(constructor, layers[i])