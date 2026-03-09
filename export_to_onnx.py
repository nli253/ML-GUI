# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 17:27:30 2025

@author: sahme627
"""

import torch
import torch.nn as nn

class ResBlock2D_BN(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x):
        Fx = self.conv1(x); Fx = self.relu1(Fx); Fx = self.conv2(Fx)
        return self.relu2(Fx + x)

class Resendc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(51, 256, kernel_size=20, padding=1)
        self.bn1d = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.resblock2d = ResBlock2D_BN(1, kernel_size=3)
    def forward(self, x):                 # x: (B, T, 51)
        x = x.permute(0,2,1)              # (B,51,T)
        x = self.conv1d(x)                # (B,256,T)
        x = self.bn1d(x); x = self.relu(x)
        x = x.permute(0,2,1)              # (B,T,256)
        B,T,C = x.shape
        x = x.reshape(B*T,1,16,16)
        x = self.resblock2d(x)            # (B*T,1,16,16)
        x = x.reshape(B,T,256)            # (B,T,256)
        return x

# ---- load your checkpoint ----
ckpt_path = 'noised_training_kernal_com_80.pth'
model = Resendc()
sd = torch.load(ckpt_path, map_location='cpu')
# strip "module." if needed
clean = { (k.replace('module.','') if isinstance(k,str) else k): v for k,v in sd.items() }
model.load_state_dict(clean, strict=False)
model.eval()

# ---- export with dynamic time length (axis 1) and batch ----
dummy = torch.randn(1, 803, 51)  # T=160 as example; time axis will be dynamic
torch.onnx.export(
    model, dummy, "resendc.onnx",
    input_names=['input'], output_names=['output'],
    opset_version=13,
    dynamic_axes={'input':{0:'batch', 1:'time'}, 'output':{0:'batch', 1:'time'}}
)
print("Exported to resendc.onnx")
