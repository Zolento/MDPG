from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms as T
from natsort import natsorted
from bart import bart
import torch.nn.functional as F
import torch
import fastmri
import h5py
import numpy as np
import os

def my_espirit(ks):
    ksp_slice = np.transpose(ks, (1, 2, 0)) # to (kx, ky, C)
    sens = bart(1, 'ecalib -S', np.expand_dims(ksp_slice, axis=-2))[..., 0]
    sens = np.squeeze(sens)
    return sens

def pool_espirit(Espirit_k):
    maps = torch.zeros_like(Espirit_k)
    with Pool(8) as pool:
        results = pool.map(my_espirit, [Espirit_k[i] for i in range(Espirit_k.shape[0])])
    for i, sens in enumerate(results):
        maps[i, :, :, :] = torch.from_numpy(np.transpose(sens, (2, 0, 1)))
    return maps

def make_folders(multi_coil, part_path, preprocess_name):
    if not os.path.exists(os.path.join(part_path, preprocess_name)):
        os.makedirs(os.path.join(part_path, preprocess_name))
    if not os.path.exists(os.path.join(part_path, preprocess_name + '/fislicecom')):
        os.makedirs(os.path.join(part_path, preprocess_name + '/fislicecom'))
    if not os.path.exists(os.path.join(part_path, preprocess_name + '/maskslice')):
        os.makedirs(os.path.join(part_path, preprocess_name + '/maskslice'))
    if multi_coil:
        if not os.path.exists(os.path.join(part_path, preprocess_name + '/csmslice')):
            os.makedirs(os.path.join(part_path, preprocess_name + '/csmslice'))
    return 

def crop_or_pad(uniform_train_resolution, kdata):
    kdata = _pad_if_needed(uniform_train_resolution, kdata)
    kdata = _crop_if_needed(uniform_train_resolution, kdata)
    return kdata


def _crop_if_needed(uniform_train_resolution, image):
    w_from = h_from = 0

    if uniform_train_resolution[0] < image.shape[-2]:
        w_from = (image.shape[-2] - uniform_train_resolution[0]) // 2
        w_to = w_from + uniform_train_resolution[0]
    else:
        w_to = image.shape[-2]

    if uniform_train_resolution[1] < image.shape[-1]:
        h_from = (image.shape[-1] - uniform_train_resolution[1]) // 2
        h_to = h_from + uniform_train_resolution[1]
    else:
        h_to = image.shape[-1]

    return image[..., w_from:w_to, h_from:h_to]


def _pad_if_needed(uniform_train_resolution, image):
    pad_w = uniform_train_resolution[0] - image.shape[-2]
    pad_h = uniform_train_resolution[1] - image.shape[-1]

    if pad_w > 0:
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left
    else:
        pad_w_left = pad_w_right = 0

    if pad_h > 0:
        pad_h_left = pad_h // 2
        pad_h_right = pad_h - pad_h_left
    else:
        pad_h_left = pad_h_right = 0
        
    if isinstance(image, torch.Tensor):
        return F.pad(image, (pad_h_left, pad_h_right, pad_w_left, pad_w_right, 0, 0), 'constant', 0)
    else:
        return F.pad(torch.from_numpy(image), (pad_h_left, pad_h_right, pad_w_left, pad_w_right, 0, 0), 'constant', 0).numpy()

if __name__ == '__main__':
    center_fractions = [0.04]
    accelerations = [8]
    uniform_train_resolution = [320, 320]
    task_name = 'knee_singlecoil'
    partlist = ['train', 'val']  # , 'test']
    slicenum = [34742, 7135]
    
    if task_name.find('single') != -1:
        multi_coil = False
    else:
        multi_coil = True
        
    for part in partlist:
        num_slice_info = []
        p = 0
        fastmri_path = '/data/fastMRI/' + task_name + '/' + part
        if not os.path.exists(fastmri_path):
            continue
        preprocess_name = 'preprocessed'
        path = Path(fastmri_path)
        filenames = natsorted([file.name for file in path.rglob('*.h5')])
        files = tqdm(filenames)
        print('Convering fullsampled h5 to npy & Preprocess')
        print('Start preprocessing, toltal set number: %s' % str(len(files)))
        make_folders(multi_coil=multi_coil, part_path=fastmri_path, preprocess_name=preprocess_name)
        
        for f in files:
            data = h5py.File(os.path.join(fastmri_path, f))
            data = data['kspace'][()]
            S = data.shape[0]
            num_slice_info.append(S)
            
            slice_kspace2 = T.to_tensor(data)  # Convert from numpy array to pytorch tensor (C,kx,ky)
            slice_image = crop_or_pad(uniform_train_resolution, torch.view_as_complex(fastmri.ifft2c(slice_kspace2)))  # Apply Inverse Fourier Transform to get the complex image (C,kx,ky,2)
            slice_image = (slice_image - torch.abs(slice_image).min()) / (torch.abs(slice_image).max() - torch.abs(slice_image).min())
            slice_kspace2 = fastmri.fft2c(torch.view_as_real(slice_image))
            slice_image = torch.view_as_real(slice_image)
            
            # compute csm
            if multi_coil:
                Espirit_k = torch.view_as_complex(slice_kspace2) #(S, C, kx, ky)
                print('Espirit_k: ', Espirit_k.shape)
                maps = pool_espirit(Espirit_k)
                print('sens_maps: ', maps.shape)
                
            mask_func = RandomMaskFunc(center_fractions=center_fractions, accelerations=accelerations)
            _, mask, _ = T.apply_mask(slice_kspace2, mask_func)
                
            for i in range(S):
                np.save(os.path.join(fastmri_path, preprocess_name) + '/maskslice/' + f[:-3] + '_mask_' + str(p), mask)  # (1,1,ky,1) or (1,1,1,ky,1);
                if multi_coil:
                    np.save(os.path.join(fastmri_path, preprocess_name)+'/csmslice/'+f[:-3]+'_csm_' + str(p), maps[i])#(C,kx,ky);
                np.save(os.path.join(fastmri_path, preprocess_name)+'/fislicecom/'+f[:-3]+'_fi_' + str(p), slice_image[i])#(kx,ky,2) or (C,kx,ky,2);
                p = p + 1
            
            files.set_description('Converting %s' % f)
        np.save(fastmri_path+'/datainfo.npy', [filenames, num_slice_info])
