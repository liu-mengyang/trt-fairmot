import torch
import time

from build_model import build_fairmot, load_model

net = build_fairmot()
model = load_model(net, "../weights/fairmot_dla34.pth")
model = model.to(torch.device('cuda'))
model.eval()

im_blob = torch.randn([1,3,608,1088]).cuda().float()
output = model(im_blob)

# 10 rounds of PyTorch FairMOT
nRound = 10
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    model(im_blob)
torch.cuda.synchronize()
time_pytorch = (time.time() - t0) / nRound
print('PyTorch time:', time_pytorch)