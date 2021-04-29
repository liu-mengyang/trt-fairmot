import torch

from DLASeg import DLASeg

# Build model
def build_fairmot():
    heads = {'hm': 1,
        'wh': 4,
        'id': 128}
    net = DLASeg(heads,
                     pretrained=True,
                     down_ratio=4,
                     final_kernel=1,
                     last_level=5,
                     head_conv=256)
    return net

# load model
def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model.load_state_dict(state_dict, strict=False)
  return model