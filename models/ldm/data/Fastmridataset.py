import os
import warnings
import torch
import numpy as np
import fastmri
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

def fft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def ifft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def fft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    data = data.to(torch.float64)
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data.to(torch.float32)

def ifft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    data = data.to(torch.float64)
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data.to(torch.float32)

class FastmriDataset(Dataset):
    def __init__(self, rootpath, pre_name, name, num=None, infer=False, useslice=True):
        super(FastmriDataset, self).__init__()
        self.dataset = []
        self.fdata_full = []
        datainfo = np.load(os.path.join(rootpath, (name + "/datainfo.npy")))
        self.filenames = datainfo[0]
        self.slice_num = [int(i) for i in datainfo[1]]
        pre_name = name + '/' + pre_name
        # now get filepath
        # for self.dram, get filepath in /fi/
        # for not self.dram, get filepath in /fislice/
        path = Path(os.path.join(rootpath, (pre_name+"/fislicecom/")))
        if num is not None:
            num = self.get_slice_num(num, self.slice_num)
            files = natsorted([file.name for file in path.rglob("*.npy")])[:num]
        else:
            files = natsorted([file.name for file in path.rglob("*.npy")])
        files = tqdm(files)
        print("Start reading fastmri, toltal %s number: %s" % (name, str(len(files))))
        for f in files:
            if useslice:
                ukpath = os.path.join(rootpath, pre_name)+"/ukslice/" + f.replace('fi_', 'uk_')
                fipath = os.path.join(rootpath, pre_name)+"/fislicecom/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/maskslice/" + f.replace('fi_', 'mask_')
            if infer:
                fi = torch.tensor(np.load(fipath))
                if useslice:
                    self.fdata_full.append(fi.unsqueeze(0))
            self.dataset.append([ukpath, fipath, maskpath])
            files.set_description("Reading processed data/datapath %s" % f)
    
    def get_slice_num(self, num, slice_num_list):
        s = 0
        for i in range(num):
            s += slice_num_list[i]
        return s
    
    def __getitem__(self, index):
        ukpath, fipath, maskpath = self.dataset[index]
        fi = torch.view_as_real(torch.tensor(np.load(fipath)))
        mask = torch.tensor(np.load(maskpath)).squeeze(0)
        # uki = torch.tensor(np.load(ukpath)).squeeze()
        ui = ifft2c(fft2c(fi) * mask)
        return {'image': (2 * fi) - 1, 'ui': (2 * ui) - 1, 'mask': mask}
    
    def __len__(self):
        
        return len(self.dataset)