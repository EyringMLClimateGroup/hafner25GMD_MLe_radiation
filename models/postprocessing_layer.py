
import torch
import torch.nn as nn
import torch.nn.functional as F

class Postprocessing(nn.Module):
    def __init__(self, out_vars, in_vars, var_len=47):
        super(Postprocessing, self).__init__()
        self.out_vars = out_vars
        self.var_len = var_len
        self.in_vars = in_vars
        
    def forward(self, y, x):
        start = 0
        new_y = torch.empty(0)
        n = 4 if "extra_3d_qs" in self.in_vars else 3
            
        # Loop over the output variables
        # and denormalize them
        for i, v in enumerate(self.out_vars):
            if "tend_ta" in v:
                l = self.var_len
                denorm = y[:,start:start+l]/86400 # Transform to K/s
            elif v in ["rsds", "rsut", "rpds_dir", "rpds_dif", "rvds_dir", "rvds_dif", "rnds_dir", "rnds_dif"]:
                l = 1
                toa = x[:, self.var_len*n]
                denorm = y[:, start]*toa 
                denorm = denorm[:,None]
                
            elif v in ["rlds","rlds_rld", "rlut"]:
                l = 1
                ts = x[:, self.var_len*n]
                sig = 5.670374419e-8
                denorm = y[:, start]*(ts**4*sig)
                denorm = denorm[:,None]
            else:
                raise ValueError("Don't know how to denormalize output variable ", v)
            if i==0:
                new_y = denorm
            else:
                new_y = torch.cat((new_y, denorm), dim=-1)
            start += l
        
        return new_y