import torch.nn as nn
import torch
import numpy as np
import nibabel as nib
from scipy.io import loadmat
import h5py
import os
from omegaconf import OmegaConf
from models.ldm.util import instantiate_from_config
import time

SEED = 1234
d_model = 32
n_stages_vssm = 4
kwargs_vssm = {
    'input_channels': 1,
    'patch_size': 4,
    'd_model': d_model,
    'UNet_base_num_features': d_model,
    'UNet_max_num_features': 1024,
    'n_stages': n_stages_vssm,
    'kernel_sizes': 3,
    'strides': [2 if i>0 else 1 for i in range(n_stages_vssm)],
    'num_output_channels': 1,
    'conv_op': nn.Conv2d,
    'n_conv_per_stage': 2,
    'n_conv_per_stage_decoder': 2,
    'deep_supervision': True
}

d_model1 = 32
n_stages_vssm_unrolled = 5
kwargs_vssm_unrolled = {
    'input_channels': 1,
    'patch_size': 4,
    'd_model': d_model1,
    'UNet_base_num_features': d_model1,
    'UNet_max_num_features': 1024,
    'n_stages': n_stages_vssm_unrolled,
    'kernel_sizes': 3,
    'strides': [2 if i>0 else 1 for i in range(n_stages_vssm_unrolled)],
    'padding': 1,
    'num_output_channels': 1,
    'conv_op': nn.Conv2d,
    'n_conv_per_stage': 2,
    'n_conv_per_stage_decoder': 2,
    'deep_supervision': True,
}

def get_time():
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

def print_to_log_file(log_file, *args, also_print_to_console=True):
    with open(log_file, 'a+') as f:
        for a in args:
            f.write(str(a))
            f.write(" ")
        f.write("\n")
    if also_print_to_console:
        print(*args)

def adjust_learning_rate(opt, epo, max_steps, initial_lr):
    exponent = 0.9
    new_lr = initial_lr * (1 - epo / max_steps) ** exponent
    for param_group in opt.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def ifft_np(data: np.ndarray) -> np.ndarray:
    """
    for complex data
    """
    data = np.fft.ifftshift(data, axes=[-2, -1])
    data = np.fft.ifftn(data, axes=[-2, -1], norm='ortho')
    data = np.fft.fftshift(data, axes=[-2, -1])
    return data

def fft_np(data: np.ndarray) -> np.ndarray:
    """
    for complex data
    """
    data = np.fft.ifftshift(data, axes=[-2, -1])
    data = np.fft.fftn(data, axes=[-2, -1], norm='ortho')
    data = np.fft.fftshift(data, axes=[-2, -1])
    return data

def ifft(data: torch.Tensor) -> torch.Tensor:
    """
    for complex data
    """
    data = torch.fft.ifftshift(data, dim=[-2, -1])
    data = torch.fft.ifftn(data, dim=[-2, -1], norm='ortho')
    data = torch.fft.fftshift(data, dim=[-2, -1])
    return data

def fft(data: torch.Tensor) -> torch.Tensor:
    """
    for complex data
    """
    data = torch.fft.ifftshift(data, dim=[-2, -1])
    data = torch.fft.fftn(data, dim=[-2, -1], norm='ortho')
    data = torch.fft.fftshift(data, dim=[-2, -1])
    return data

def fft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data

def ifft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data

def load_model_from_config(config, ckpt):
    print(f"Loading latent diffusion model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model(ckpt_path, config_path):
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    return model

def read_file(file_path):
    """
    根据文件格式读取文件内容。

    :param file_path: 文件路径
    :return: 文件内容
    """
    # 获取文件扩展名
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.npy':
        # 读取 .npy 文件
        data = np.load(file_path)
    elif file_extension == '.nii.gz' or file_extension == '.gz':
        # 读取 .nii.gz 文件
        img = nib.load(file_path)
        data = img.get_fdata()
    elif file_extension == '.mat':
        # 读取 .mat 文件
        data = loadmat(file_path)
    elif file_extension == '.h5' or file_extension == '.hdf5':
        # 读取 .h5 或 .hdf5 文件
        with h5py.File(file_path, 'r') as f:
            data = {key: f[key][()] for key in f.keys()}
    else:
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
        except:
            raise ValueError(f"Unsupported file format: {file_extension}")

    return data

if __name__ == "__main__":
    pass