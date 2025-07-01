import torch
import torch.nn as nn
import fastmri
import numpy as np
from models.modules.blocks import fft2c
from skimage.metrics import structural_similarity as ssim

def compute_metrics_real(out, label, L=0):
    psnr_ = 0
    ssim_ = 0
    out = out[0].squeeze(1).contiguous().detach().cpu().numpy()
    label = fastmri.complex_abs(label).squeeze(1).contiguous().detach().cpu().numpy()
    assert out.shape == label.shape, 'tensor size inconsistency'
    if L == 0 or L > out.shape[0]:
        num = out.shape[0]
    else:
        num = L
    for i in range(num):
        x = out[i,...]
        y = label[i,...]
        psnr_ += psnr(x, y)
        ssim_ += ssim(x, y, data_range=1.0, win_size=11)
    return psnr_ / num, ssim_ / num

def psnr(img1:torch.Tensor, img2:torch.Tensor):
    mse = ((img1-img2)**2).mean()
    if mse == 0:
        return np.inf
    else:
        return 10*np.log10(1.0/mse)


class nacs(nn.Module):
    def __init__(self, mask_size, acc):
        super(nacs, self).__init__()
        acs_mask_dict = {'320x8':[154, 166],
                        '320x4':[147, 173],
                        '256x8':[122, 132],
                        '256x4':[118, 137]}
        self.acs_mask = torch.zeros((1,1,mask_size,1), device="cuda")
        self.acs_mask[0, 0, acs_mask_dict[str(mask_size)+acc][0]:acs_mask_dict[str(mask_size)+acc][1], 0] = 1
    def forward(self, out, label):
        out = fft2c(out) * (1 - self.acs_mask)
        label = fft2c(label) * (1 - self.acs_mask)
        loss = 1e-1 * torch.norm((out - label),'fro') / torch.norm(label,'fro')
        return loss

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcr = nacs(mask_size=320, acc='x8')
    def forward(
        self, out, label
    ):
        label_real = fastmri.complex_abs(label)
        l = torch.norm((out - label_real),'fro') / torch.norm(label_real,'fro') + self.fcr(out[1], label)
        return l