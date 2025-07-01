# (MICCAI2025) MDPG: Multi-domain Diffusion Prior Guidance for MRI Reconstruction
## Environment preparation
We use the Pytorch 2.2.0 under CUDA 12.1. VMamba is necessary for run the model, please follow https://github.com/MzeroMiko/VMamba for compilation.

[optional] bart toolbox is necessary if you want to use multicoil dataset.
## Dataset preparation
Download dataset at https://fastmri.med.nyu.edu/ and https://brain-development.org/ixi-dataset/. Use utils/convert_fastmri.py to preprocess the fastmri dataset.

    Example tree structure:
    ```
    /data/fastmri/knee_singlecoil/
    ├── datainfo.npy
    ├── maskslice
    │   ├── ...
    ├── fislicecom
    │   ├── ...
    └── csmslice(available when multicoil)
        ├── ...
    ```

change the `center_fractions` and `accelerations` to control the mask.

## Train Stage I
To train the a LDM-16 model in the paper, please follow the https://github.com/CompVis/latent-diffusion. You can find our pretrained LDM-16 model and corresponding config files here: https://drive.google.com/drive/folders/1Rcy8DRG___1hmPqk12FD4roRKzAmGBPL?usp=sharing.

## Train Stage II
Run:

    python scripts/fastmri/train_fastmri.py 


## Evaluation

    python scripts/fastmri/infer_fastmri.py

[notice] final results are combined as volumns.