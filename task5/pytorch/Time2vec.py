import torch
import torch.nn as nn

class Time2Vec(nn.Module):
  

   def _init_(self, in_channels = 256, out_channels = 512):
      super()._init_()
      self.out_channels = out_channels
      self.in_channels = in_channels
      self.w0 = nn.Parameter(torch.Tensor(1, in_channels))
      self.phi0 = nn.Parameter(torch.Tensor(1, in_channels))
      self.W = nn.Parameter(torch.Tensor(in_channels, out_channels-1))
      self.Phi = nn.Parameter(torch.Tensor(in_channels, out_channels-1))
      self.reset_parameters()

   def reset_parameters(self):
      nn.init.uniform_(self.w0, 0, 1)
      nn.init.uniform_(self.phi0, 0, 1)
      nn.init.uniform_(self.W, 0, 1)
      nn.init.uniform_(self.Phi, 0, 1)

   def forward(self, x):
      n_batch = x.size(0)
      original = (x*self.w0 + self.phi0).unsqueeze(-1)
      x = torch.repeat_interleave(x, repeats=self.out_channels-1, dim=0).view(n_batch,-1,self.out_channels-1)
      x = torch.sin(x * self.W + self.Phi)
      return torch.cat([original,x],-1).view(n_batch,self.out_channels,-1).contiguous()