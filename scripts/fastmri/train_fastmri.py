import os
import torch
import argparse
import numpy as np
from utils.utils import SEED, get_time, adjust_learning_rate, print_to_log_file, get_model
from utils.utils import kwargs_vssm_unrolled as kwargs
from models.ldm.models.diffusion.ddim import DDIMSampler
from models.model_VSSM import VSSMUNet_unrolled
from tqdm import tqdm
from utils.dataset import FastmriDataset
from utils.Loss_fastmri import Loss, compute_metrics_real
from torch.utils.data.dataloader import DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', type=int, default=100, required=False, help='number of epochs')
    parser.add_argument('-lr', type=float, default=4e-3, required=False, help='initial learning rate')
    parser.add_argument('-bs', type=int, default=8, required=False, help='batch size')
    parser.add_argument('-iter', type=int, default=1, required=False, help='iteration')
    parser.add_argument('-d_model', type=int, default=128, required=False, help='d_model')
    parser.add_argument('-amp', type=str, default='fp32', required=False, help='fp16;fp32')
    parser.add_argument('-dctype', type=str, default='AM', required=False, help='VN;AM')
    parser.add_argument('-expname', type=str, default='', required=False)
    parser.add_argument('-eval_continue_model', type=str, default='model_latest', required=False, help='eval model name')
    parser.add_argument('-dataset_name', type=str, default='fastMRI', required=False, help='dataset name')
    parser.add_argument('-read_path', type=str, default='/data/fastMRI/knee_singlecoil', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='preprocessed', required=False, help='preprossed folder name')
    parser.add_argument('-acc', type=str, default='8x', required=False, help='acc')
    parser.add_argument('-save_path', type=str, default='models/saved', required=False, help='model save path')
    parser.add_argument('-ldmcfg_path', type=str, default='models/saved/pretrained/configs/fastmri-ldm-kl-16.yaml', required=False, help='pretrained LDM cfg_path')
    parser.add_argument('-ldmckpt_path', type=str, default='models/saved/pretrained/fastmri-kl-16-8x.ckpt', required=False, help='pretrained LDM ckpt_path')
    parser.add_argument('--c', action='store_true', required=False, help='[OPTIONAL] continue training from latest checkpoint')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='[INTEGRAL IF --c] from which filefolder to continue')
    parser.add_argument('--ddp', action='store_true', required=False, help='[OPTIONAL] if ddp')
    parser.add_argument('--eval', action='store_true', default=False, required=False, help='[OPTIONAL] if eval')
    args = parser.parse_args()
    # set default device and random pick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fastmri_path = args.read_path
    Model_path = args.save_path
    load_done = False
    np.random.seed(SEED)
    # Checking paths
    Cur_model_path = os.path.join(Model_path, args.dataset_name)
    if not os.path.exists(Cur_model_path):
        os.makedirs(Cur_model_path)
    dataloader_params = {'batch_size': args.bs,
                    'shuffle':True,
                    'num_workers': 4,
                    'pin_memory': True, 
                    'drop_last' : True}
    dataloader_params_ = {'batch_size': 4,
            'shuffle':True,
            'num_workers': 4,
            'pin_memory': True, 
            'drop_last' : True}
    if not args.eval:
        train_dataset = FastmriDataset(rootpath=Fastmri_path, 
                    pre_name=args.pre_name,
                    acc=args.acc,
                    name='train'
                    )
        train_dataloader = DataLoader(dataset=train_dataset, **dataloader_params)
    test_dataset = FastmriDataset(rootpath=Fastmri_path,
                    pre_name=args.pre_name,
                    acc=args.acc,
                    name='val'
                    )
    test_dataloader = DataLoader(dataset=test_dataset, **dataloader_params)
    torch.cuda.empty_cache()
    print("Creating model")
    if kwargs['d_model'] != args.d_model:   
        kwargs['d_model'] = args.d_model
        kwargs['UNet_base_num_features'] = args.d_model
    if kwargs['input_channels'] != 2:
        kwargs['input_channels'] = 2
    if args.ldmcfg_path != '' and args.ldmckpt_path != '':
        ldm_model = get_model(args.ldmckpt_path, args.ldmcfg_path)
        sampler = DDIMSampler(ldm_model)
    else:
        print('No LDPG')
        sampler = None
        ldm_model = None
    model = VSSMUNet_unrolled(iter=args.iter, DC_type=args.dctype, posterior_sampler=sampler, kwargs=kwargs)
    # to device
    print(f"Total num_gpu = {torch.cuda.device_count()}, DDP = {args.ddp}")
    if args.ddp and torch.cuda.device_count()>=2:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
        device_id = rank % torch.cuda.device_count()
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id],find_unused_parameters=True)
    else:
        device_id = device
        model = model.to(device_id)
        rank = 0
    loss = Loss().to(device_id)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    if args.amp == 'fp16':
        scaler = GradScaler()
    elif args.amp == 'fp32':
        scaler = None
    # If continute training
    if args.c or args.eval:
        Cur_model_path = os.path.join(Cur_model_path, args.model_folder)
        assert args.model_folder is not None, 'A specific model-folder must be given to continue or evaluate! eg: 2020-01-01-00:00:00'
        checkpoint = torch.load(os.path.join(Cur_model_path, args.eval_continue_model+'.pth'))
        name = next(iter(checkpoint['model_state_dict']))
        if name[:6] == 'module' and not args.ddp:
            new_state_dict = {}
            for k,v in checkpoint['model_state_dict'].items():
                new_state_dict[k[7:]] = v
            checkpoint['model_state_dict'] = new_state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_ep = checkpoint['epoch'] + 1
        load_done = True
    else:
        start_ep = 0
        Cur_model_path = os.path.join(Cur_model_path, get_time()+args.pre_name+'_VSSM_unrolled'+args.dctype + args.expname)
        os.makedirs(Cur_model_path, exist_ok=True)
        config_file = os.path.join(Cur_model_path,"kwargs.txt")
        with open(config_file, 'w') as f:
            for arg in vars(args):
                f.write(arg+':'+str(getattr(args, arg))+'\n')
            f.write('{\n')
            for k,v in kwargs.items():
                f.write(str(k))
                f.write(':')
                f.write(str(v))
                f.write(',\n')
            f.write('}\n')
    assert start_ep < args.ep, 'countinue training={}, start_ep={}, max_ep={}'.format(args.c, start_ep, args.ep)
    ep = tqdm(range(start_ep, args.ep))
    log_file = os.path.join(Cur_model_path, get_time()+'.txt')
    if rank == 0:
        for k,v in kwargs.items():
            print(k,v)
    if args.eval:
        # notice that validate is on croppped dataset
        # if need to validate on original data, go for the inference
        # ===================offline validate========================
        print("Start evaluation...")
        assert load_done, 'Please follow the steps of continue training first'
        model.eval()
        p = 0
        s = 0
        with torch.no_grad():
            for batch_idx, (input, label, mask) in enumerate(test_dataloader):
                output = model(input.to(device_id), mask.to(device_id))
                PSNR, SSIM = compute_metrics_real(output, label.to(device_id))
                p = p + PSNR
                s = s + SSIM
            print("test_psnr: %.4f test_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
        print("Evaluation done.")
    else:
        # ===================train========================
        print_to_log_file(log_file, "Start training...") if not args.c else print_to_log_file(log_file, "Continue training...")
        best_psnr = 0
        for epoch in ep:
            torch.cuda.empty_cache()
            model.zero_grad()
            cur_lr = adjust_learning_rate(optimizer, epoch, args.ep, args.lr)
            model.train()
            p = 0
            s = 0
            loss_log = 0
            for batch_idx, [input, label, mask, csm] in enumerate(train_dataloader):
                if scaler is not None:
                    with autocast(device_type='cuda'):
                        output = model(input.to(device_id), mask.to(device_id), csm.to(device_id))
                        optimizer.zero_grad()
                        l = loss(output, label.to(device_id))
                        loss_log = loss_log + l.data.item()
                    scaler.scale(l).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(input.to(device_id), mask.to(device_id), csm.to(device_id))
                    optimizer.zero_grad()
                    l = loss(output, label.to(device_id))
                    loss_log = loss_log + l.data.item()
                    l.backward()
                    optimizer.step()
                PSNR, SSIM = compute_metrics_real(output, label.to(device_id))
                if rank == 0:
                    print('['+str(batch_idx)+'/'+str(len(train_dataloader))+']'+' '+str(PSNR)+' '+str(SSIM) +' '+str(l.data.item()))
                p = p + PSNR
                s = s + SSIM
            if rank == 0:
                print_to_log_file(log_file, "%s Epoch %d lr %.6f train_loss: %.12f " % (get_time(), epoch, cur_lr,  float(loss_log/(batch_idx+1))))
                print_to_log_file(log_file, "train_psnr: %.4f train_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
            if (epoch + 1) % 10 == 0:
                torch.save({
                            'kwargs': kwargs,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(Cur_model_path, 'model{}{}.pth'.\
                            format(epoch+1, get_time())))
            elif (epoch + 1) == args.ep:
                torch.save({
                            'kwargs': kwargs,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(Cur_model_path, 'model_final.pth'))
            else:
                torch.save({
                'kwargs': kwargs,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(Cur_model_path, 'model_latest.pth'))                 
            # ===================online validate========================
            p = 0
            s = 0
            with torch.no_grad():
                for batch_idx, [input, label, mask, csm]in enumerate(test_dataloader):
                    output = model(input.to(device_id), mask.to(device_id), csm.to(device_id))
                    PSNR, SSIM = compute_metrics_real(output, label.to(device_id))
                    p = p + PSNR
                    s = s + SSIM
                if rank == 0:
                    print_to_log_file(log_file, "test_psnr: %.4f test_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
            if float(p/(batch_idx+1)) > best_psnr:
                best_psnr = float(p/(batch_idx+1))
                torch.save({
                'kwargs': kwargs,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(Cur_model_path, 'model_best_psnr.pth'))     
            ep.set_description("Trained epoch %s" % epoch)
        print_to_log_file(log_file, "Training done.")
        
    
