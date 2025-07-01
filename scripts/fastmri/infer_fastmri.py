import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils.utils import kwargs_vssm_unrolled as kwargs, get_model
from models.model_VSSM import VSSMUNet_unrolled
from models.ldm.models.diffusion.ddim import DDIMSampler
from utils.dataset import FastmriDataset
from torch.utils.data.dataloader import DataLoader

def rebuild(output_list, slice_num):
    p = 0
    out = []
    for i in range(len(slice_num)):
        tmps = []
        for _ in range(slice_num[i]):
            tmps.append(output_list[p])
            p += 1
        out.append(torch.cat(tmps))
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', type=int, default=1, required=False, help='iteration')
    parser.add_argument('-d_model', type=int, default=128, required=False, help='d_model')
    parser.add_argument('-dctype', type=str, default='AM', required=False, help='VN;AM')
    parser.add_argument('-bs', type=int, default=8, required=False, help='batch size')
    parser.add_argument('-eval_model', type=str, default='model_final', required=False, help='eval model name')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='from which filefolder to infer')
    parser.add_argument('-dataset_name', type=str, default='fastMRI', required=False, help='dataset name')
    parser.add_argument('-data_path', type=str, default='/data/fastMRI/knee_singlecoil/', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='preprocessed', required=False, help='preprossed folder name')
    parser.add_argument('-acc', type=str, default='8x', required=False, help='acc')
    parser.add_argument('-ldmcfg_path', type=str, default='models/saved/pretrained/configs/fastmri-ldm-kl-16.yaml', required=False, help='pretrained LDM cfg_path')
    parser.add_argument('-ldmckpt_path', type=str, default='models/saved/pretrained/fastmri-kl-16-8x.ckpt', required=False, help='pretrained LDM ckpt_path')
    parser.add_argument('-save_path', type=str, default='models/saved', required=False, help='model save path')
    parser.add_argument('-output_path', type=str, default='models/output', required=False, help='output path')
    args = parser.parse_args()
    # In this part we inference the whole raw data and evaluate the average PSNR and SSIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fastmri_path = args.data_path
    Model_path = args.save_path
    # Checking paths
    args.eval_model = 'model_latest'
    args.model_folder = 'x8latentonly'
    assert args.model_folder is not None, 'A specific model-folder must be given to infer! eg: 2020-01-01-00:00:00***'
    Cur_model_path = os.path.join(os.path.join(Model_path, args.dataset_name), args.model_folder)
    output_path = os.path.join(os.path.join(os.path.join(args.output_path, args.dataset_name), args.model_folder), args.eval_model)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    infer_dataset = FastmriDataset(rootpath=Fastmri_path, 
                pre_name=args.pre_name,
                name='val',
                acc=args.acc,
                infer=True,
                )
    infer_dataloader = DataLoader(dataset=infer_dataset, 
                            batch_size=args.bs, 
                            shuffle=False, 
                            drop_last=False,
                            num_workers=0,
                            pin_memory=True)
    torch.cuda.empty_cache()
    print("Creating model")
    if kwargs['d_model'] != args.d_model:   
        kwargs['d_model'] = args.d_model
        kwargs['UNet_base_num_features'] = args.d_model
    if kwargs['input_channels'] != 2:
        kwargs['input_channels'] = 2
    for k,v in kwargs.items():
        print(k,v)
    if args.ldmcfg_path != None and args.ldmckpt_path != None:
        ldm_model = get_model(args.ldmckpt_path, args.ldmcfg_path)
        sampler = DDIMSampler(ldm_model)
    else:
        sampler = None
        ldm_model = None
    model = VSSMUNet_unrolled(iter=args.iter, DC_type=args.dctype, posterior_sampler=sampler, kwargs=kwargs)
    model = model.to(device)
    print("Start evaluation...")
    checkpoint = torch.load(os.path.join(Cur_model_path, args.eval_model+'.pth'), map_location='cuda:0')['model_state_dict']
    name = next(iter(checkpoint))
    if name[:6] == 'module':
        new_state_dict = {}
        for k,v in checkpoint.items():
            new_state_dict[k[7:]] = v
        checkpoint = new_state_dict
    model.load_state_dict(checkpoint)
    model.eval()
    label_full = infer_dataset.fdata_full
    slice_num = infer_dataset.slice_num
    files = infer_dataset.filenames
    slice_num_iterpoints = []
    s = 0
    for i in slice_num:
        s += i
        slice_num_iterpoints.append(s)
    output_list = []
    volumn_idx = 0
    with torch.no_grad():
        with tqdm(total=len(infer_dataloader)) as pbar:
            for idx, [input, label, mask, csm] in enumerate(infer_dataloader):
                torch.cuda.empty_cache()
                out = model(input.to(device), mask.to(device), csm.to(device), args.dorefine).to('cpu').squeeze()
                p = 0
                for i in out:
                    output_list.append(i.unsqueeze(0))
                    if idx*args.bs + p == slice_num_iterpoints[0] - 1:
                        slice_num_iterpoints.pop(0)
                        np.save(output_path+'/'+files[volumn_idx].split('.')[0], rebuild(output_list, [slice_num[volumn_idx]])[0])
                        for i in range(slice_num[volumn_idx]):
                            output_list.pop(0)
                        volumn_idx += 1
                    p += 1
                pbar.update(1)
    print("Inference done.")
