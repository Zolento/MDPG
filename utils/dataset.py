import torch
import numpy as np
import fastmri
import os
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from pathlib import Path
from tqdm import tqdm
from utils.utils import fft2c, ifft2c

class FastmriDataset(Dataset):
    def __init__(self, rootpath, pre_name, name, acc='4x', csm=False, infer=False, mask=None, num_samples=None):
        super(FastmriDataset, self).__init__()
        self.dataset = []
        self.fdata_full = []
        self.csm = csm
        self.mask = torch.tensor(np.load(mask)) if mask != None else None # use a fixed mask or random generated mask set
        # datainfo = np.load(os.path.join(rootpath, (name + "/datainfo.npy")))
        # self.filenames = datainfo[0]
        # self.slice_num = [int(i) for i in datainfo[1]]
        pre_name = name + '/' + pre_name
        path = Path(os.path.join(rootpath, (pre_name+"/fislicecom/")))
        files = natsorted([file.name for file in path.rglob("*.npy")])
        if num_samples is not None:
            # when split a same set, e.g., only train on validation set
            # if num_samples < len(files) // 2:
            #     files = files[:num_samples]
            # else:
            #     files = files[-num_samples:]
            files = files[:num_samples]
        files = tqdm(files)
        print("Start reading fastmri, toltal %s number: %s" % (name, str(len(files))))
        for f in files:
            fipath = os.path.join(rootpath, pre_name)+"/fislicecom/" + f
            if self.mask is None:
                maskpath = os.path.join(rootpath, pre_name)+"/maskslice" + acc +'/' + f.replace('fi_', 'mask_')
            else:
                maskpath = ''
            csmpath = os.path.join(rootpath, pre_name)+"/csmslice/" + f.replace('fi_', 'csm_')
            if infer:
                fi = torch.tensor(np.load(fipath))
                self.fdata_full.append(fi.unsqueeze(0))
            self.dataset.append([fipath, maskpath, csmpath])
            files.set_description("Reading processed datapath %s" % f)
            
    def __getitem__(self, index):
        fipath, maskpath, csmpath = self.dataset[index]
        fii = torch.tensor(np.load(fipath))# (kx, ky, 2) or (C, kx, ky, 2)
        # if self.mask is None:
        #     mask = torch.tensor(np.load(maskpath))
        # else:
        #     mask = self.mask
        mask = torch.tensor(np.load(maskpath)).squeeze(0)
        uki = fft2c(fii) * mask
        fii = fastmri.complex_abs(fii) # It depends: if use k-space loss, then comment it
        if self.csm:
            csm = torch.view_as_real(torch.tensor(np.load(csmpath)))# (C, kx, ky, 2)
            return uki, fii, mask, csm
        else:
            return uki, fii, mask, torch.tensor([0])
            
    def __len__(self):
        return len(self.dataset)
    
class IXIDataset(Dataset):
    def __init__(self, rootpath, pre_name, name, acc='4x', infer=False):
        super(IXIDataset, self).__init__()
        self.dataset = []
        self.fdata_full = []
        datainfo = np.load(os.path.join(rootpath, (name + "/datainfo.npy")))
        self.filenames = datainfo[0]
        self.slice_num = [int(i) for i in datainfo[1]]
        pre_name = name + '/' + pre_name
        path = Path(os.path.join(rootpath, (pre_name+"/fislice/")))
        files = natsorted([file.name for file in path.rglob("*.npy")])
        files = tqdm(files)
        print("Start reading IXI, toltal %s number: %s" % (name, str(len(files))))
        for f in files:
            fipath = os.path.join(rootpath, pre_name)+"/fislice/" + f
            maskpath = os.path.join(rootpath, pre_name)+'/maskslice' + acc +'/' + f.replace('fi_', 'mask_')
            if infer:
                fi = torch.tensor(np.load(fipath))
                self.fdata_full.append(fi)#(kx,ky)
            self.dataset.append([fipath, maskpath])
            files.set_description("Reading processed datapath %s" % f)
            
    def __getitem__(self, index):
        fipath, maskpath = self.dataset[index]
        fi = torch.tensor(np.load(fipath)).unsqueeze(0).to(torch.float)
        mask = torch.tensor(np.load(maskpath)).unsqueeze(0)
        return fi, mask
            
    def __len__(self):
        return len(self.dataset)