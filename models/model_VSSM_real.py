import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from typing import Union, Type, List, Tuple
from models.modules.vmamba import *
from models.modules.blocks import *
from models.humus_net import SensitivityModel
from timm.models.layers import trunc_normal_
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from utils.utils import kwargs_plainunet as plainunet_kwargs
import fastmri

def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
    stride = patch_size // 2
    kernel_size = stride + 1
    padding = 1
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
        (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
        (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    )

def _make_patch_embed_v23D(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
    # if channel first, then Norm and Output are both channel_first
    stride = patch_size // 2
    kernel_size = stride + 1
    padding = 1
    return nn.Sequential(
        nn.Conv3d(in_chans, embed_dim // 2, kernel_size=(1, kernel_size, kernel_size), stride=(1,stride,stride), padding=(0,padding,padding)),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 4, 1)),
        (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 4, 1, 2, 3)),
        nn.GELU(),
        nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(1, kernel_size, kernel_size), stride=(1,stride,stride), padding=(0,padding,padding)),
        (nn.Identity() if channel_first else Permute(0, 2, 3, 4, 1)),
        (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    )
   
def _make_patch_expand(in_chans, out_chans, patch_size):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size),
    )
    
def _make_patch_expand3D(in_chans, out_chans, patch_size):
    return nn.Sequential(
        nn.ConvTranspose3d(in_chans, out_chans, kernel_size=(1,patch_size//2,patch_size//2), stride=(1,patch_size//2,patch_size//2)),
        LayerNorm(out_chans, data_format="channels_first"),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(out_chans, out_chans, kernel_size=(1,patch_size//2,patch_size//2), stride=(1,patch_size//2,patch_size//2)),
    )

class VSSM_Decoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 ouput_channels: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision: bool = True
                 ):

        super().__init__()
        stages = []
        transpconvs = []
        seg_layers = []
        if encoder.conv_op == Conv2d_channel_last:
            transpconv_op = Conv2dTran_channel_last
        elif encoder.conv_op == nn.Conv2d:
            transpconv_op = nn.ConvTranspose2d
        elif encoder.conv_op == nn.Conv3d:
            transpconv_op = nn.ConvTranspose3d
        else:
            transpconv_op = Conv3dTran_channel_last
        conv_op = encoder.conv_op
        n_stages_encoder = len(encoder.output_channels)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                in_channels=input_features_below, 
                out_channels=input_features_skip, 
                kernel_size=stride_for_transpconv, 
                stride=stride_for_transpconv,
            )
            )
            stage_modules = []
            stage_modules.append(
                StackedConvBlocks(
                    num_convs=n_conv_per_stage[s-1], 
                    conv_op=conv_op,
                    input_channels=input_features_skip*2, 
                    output_channels=input_features_skip, 
                    kernel_size=3,
                    initial_stride=1,
                    conv_bias=True,
                    norm_op=Instance2d_channel_last,
                    norm_op_kwargs={'eps': 1e-5, 'affine': True},
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nn.LeakyReLU,
                    nonlin_kwargs={'inplace': True},
                    nonlin_first=False))
            stages.append(nn.Sequential(*stage_modules))
            
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.deep_supervision = deep_supervision
    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), -1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(x)
            elif s == (len(self.stages) - 1):
                seg_outputs.append(x)
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

class VSSM_Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_size:int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 padding: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 return_skips: bool = False
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(padding, int):
            padding = [padding] * n_stages
            
        stages = []
        self.return_skips = return_skips
        self.output_channels = features_per_stage
        self.strides = strides
        self.padding = padding
        if conv_op == nn.Conv2d:
            conv_op = Conv2d_channel_last
        elif conv_op == nn.Conv3d:
            conv_op = Conv3d_channel_last
        
        self.conv_op = conv_op
        if conv_op == nn.Conv2d or conv_op == Conv2d_channel_last:
            stages.append(_make_patch_embed_v2(
                            in_chans=input_channels,
                            embed_dim=features_per_stage[0],
                            patch_size=patch_size))
        else:
            stages.append(_make_patch_embed_v23D(
                            in_chans=input_channels,
                            embed_dim=features_per_stage[0],
                            patch_size=patch_size))
            
        self.latent_guided_attns = nn.ModuleList([LatentGuidedAttention(x_dim=features_per_stage[-3], 
                                                                        heads=8,
                                                                        dim_head=features_per_stage[-3]//8),
                                                LatentGuidedAttention(x_dim=features_per_stage[-2], 
                                                                        heads=16,
                                                                        dim_head=features_per_stage[-2]//16),
                                                LatentGuidedAttention(x_dim=features_per_stage[-1], 
                                                                        heads=32,
                                                                        dim_head=features_per_stage[-1]//32)])
        # q: x; k,v: z
        self.latent_pool = nn.ModuleList([nn.Identity(), 
                                          nn.AdaptiveAvgPool2d((8,8)),
                                          nn.AdaptiveAvgPool2d((4,4))])
        for s in range(n_stages):
            stage_modules = []
            conv_stride = strides[s]
            conv_padding = padding[s]
            if s > 0:
                stage_modules.append(
                    conv_op(
                    in_channels=input_channels,
                    out_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=conv_stride,
                    padding=conv_padding
                    )
                    )
            for _ in range(n_conv_per_stage[s]):
                stage_modules.append(
                VSSBlock(
                    hidden_dim=features_per_stage[s],
                    ssm_init="v2",
                    forward_type="v2_noz"
                )
                )
            input_channels = features_per_stage[s]
            stages.append(nn.Sequential(*stage_modules))
        self.stages = nn.ModuleList(stages)
        
    def latent_guided_attn(self, x, z, idx):
        b, h, w, c = x.shape
        # xg = F.adaptive_avg_pool2d(x.permute(0,3,1,2), (1, 1)).flatten(1)
        x = x.permute(0,3,1,2).flatten(2).transpose(1, 2)
        z = z.flatten(2).transpose(1, 2)
        x = self.latent_guided_attns[idx](x, z).reshape(b, h, w, c)
        return x
        
    def forward(self, x, post=None):
        ret = []
        for i, s in enumerate(self.stages):
            if i in [3, 4, 5] and post != None:
                x = self.latent_guided_attn(s[0](x), self.latent_pool[i-3](post), i-3)
                x = s[1:](x)
            else:
                x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

class VSSMUNet(nn.Module):
    def __init__(self,
                input_channels: int,
                patch_size: int,
                d_model: int,
                n_stages: int,
                conv_op: Type[_ConvNd],
                kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                strides: Union[int, List[int], Tuple[int, ...]],
                padding: Union[int, List[int], Tuple[int, ...]],
                n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                num_output_channels: int,
                UNet_base_num_features: int,
                UNet_max_num_features: int,
                n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                deep_supervision: bool = True,
                out_put: bool = False,
                DC_type: str = 'AM',
                num_coils: int = 1
                ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        assert DC_type in ('VN', 'AM', '')
        self.DC_type = DC_type
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        features_per_stage = [min(UNet_base_num_features * 2 ** i,
                                UNet_max_num_features) for i in range(n_stages)]
        self.input_channels = input_channels
        num_output_channels = input_channels
        self.d_model = d_model
        if conv_op==nn.Conv2d:
            self.patch_embed = _make_patch_embed_v2(in_chans=input_channels,
                                                        embed_dim=d_model,
                                                        patch_size=patch_size)
            self.patch_expand = _make_patch_expand(in_chans=d_model,
                                                        out_chans=input_channels,
                                                        patch_size=patch_size)
        else:
            self.patch_embed = _make_patch_embed_v23D(in_chans=input_channels,
                                                        embed_dim=d_model,
                                                        patch_size=patch_size)
            self.patch_expand = _make_patch_expand3D(in_chans=UNet_base_num_features,
                                                        out_chans=input_channels,
                                                        patch_size=patch_size)
        if DC_type != '':
            self.DC = DC_layer(DC_type=DC_type, soft=True)
        else:
            print('Hard DC')
            self.DC = DC_layer(DC_type=DC_type, soft=False)
        self.out_put = out_put
        self.encoder = VSSM_Encoder(input_channels=input_channels,
                                    patch_size=patch_size,
                                    n_stages=n_stages, 
                                    n_conv_per_stage=n_conv_per_stage,
                                    features_per_stage=features_per_stage, 
                                    conv_op=conv_op,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    padding=padding,
                                    return_skips=True)
        self.decoder = VSSM_Decoder(self.encoder, 
                                    num_output_channels, 
                                    n_conv_per_stage_decoder, 
                                    deep_supervision)
        
        self.DualDomainFusion = DualDomainFusion(in_channel=1)
        
    def forward_(self, x, mask, uk, res=None):
        B, T, C, h, w = x.size()
        x = x.reshape(B, T*C, h, w)
        skips = self.encoder(x)
        out = self.decoder(skips)
        out_put = out[0].permute(0,3,1,2)#(B, T*C, h, w)
        if res is not None:
            return self.DC(self.patch_expand(out_put).reshape(B, T, C, h, w) + res, mask, uk)
        else:
            return self.DC(self.patch_expand(out_put).reshape(B, T, C, h, w), mask, uk)

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std
    
    def unnorm(self,
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    @staticmethod
    def sens_expand(x: torch.Tensor, sens_map: torch.Tensor) -> torch.Tensor:
        x = fastmri.complex_mul(x.unsqueeze(-1).transpose(1, -1), sens_map)
        x = fft2c(x, dim=(-2, -1))
        
        return x
    
    @staticmethod
    def sens_reduce(x: torch.Tensor, sens_map: torch.Tensor, dim=0) -> torch.Tensor:
        x = ifft2c(x, dim=(-2, -1))
        x = fastmri.complex_mul(x, fastmri.complex_conj(sens_map)).sum(dim) # (B, H, W, 2)
        
        return x.permute(0, 3, 1, 2)
    
    def apply_model(self, input, sens, post, iz_rec=None):
        if sens != None:
            input = self.sens_reduce(input, sens, dim=1)
        else:
            input = torch.real(ifft(input))
        if iz_rec != None:
            input = self.DualDomainFusion(input, iz_rec)
        input, mean, std = self.norm(input)
        
        skips = self.encoder(input, post)
        out_put = self.decoder(skips)[0]
        out_put = self.patch_expand(out_put.permute(0,3,1,2))

        out_put = self.unnorm(out_put, mean, std)
        
        if sens != None:
            out_put = self.sens_expand(out_put, sens)
        else:
            out_put = fft(out_put)
        return out_put
    
    def forward(self, uk, mask, uk0, sens, post, iz_rec2=None):
        Gk = self.apply_model(uk, sens, post, iz_rec2)
        
        if self.DC_type == 'VN':
            k_out = self.DC(uk, mask, uk0) - Gk
        elif self.DC_type == 'AM':
            k_out = self.DC(uk + Gk, mask, uk0)
        else:
            k_out = self.DC(Gk, mask, uk0)
            
        if self.out_put:
            i_rec = torch.real(ifft(k_out))
            if sens == None:
                return [i_rec]
            else:
                i_rec = fastmri.rss_complex(i_rec, dim=1)
                return [i_rec]
        else:
            return k_out

class VSSMUNet_unrolled(nn.Module):
    def __init__(self, iter=6, DC_type='VN', num_coils=1, learned_csm=False, posterior_sampler=None, kwargs=None):
        super().__init__()
        self.layers = []
        self.learned_csm = learned_csm
        self.ddim_steps = 20
        self.latent_shape = [16, 16, 16]
        for _ in range(iter-1):
            self.layers.append(VSSMUNet(**kwargs, DC_type=DC_type))
        self.layers.append(VSSMUNet(**kwargs, DC_type=DC_type, out_put=True))
        self.layers = nn.ModuleList(self.layers)
        if num_coils > 1 and learned_csm:
            self.sens_net = SensitivityModel(
                chans=num_coils,
                num_pools=4,
                mask_center=True,
            )
        else:
            self.sens_net = None
        self.posterior_sampler = posterior_sampler
        if self.posterior_sampler != None:
            self.z_norm = LayerNorm(self.latent_shape[0], data_format="channels_first")
        self.apply(self._init_weights)
        
    def forward(self, x, mask, csm):
        posterior = None
        if csm == None:
            pass
        elif csm.max() == 0:
            csm = None
        ui = torch.real(ifft(x))
        uk0 = x.clone()
        sens = self.sens_net(uk0, mask) if self.sens_net != None and self.learned_csm else csm
        if self.posterior_sampler != None:
            with torch.no_grad():
                with self.posterior_sampler.model.ema_scope():
                    self.posterior_sampler.model.eval()
                    posterior = self.sample_posterior(bs=x.shape[0],\
                    cond = self.posterior_sampler.model.get_learned_conditioning(self.normalize(ui)))
                    iz_rec = self.unnormalize(self.posterior_sampler.model.decode_first_stage(posterior))
        for layer in self.layers:
            x = layer(x, mask, uk0, sens, posterior, iz_rec)
        return x[0]
    
    def normalize(self, x):
        return 2 * x - 1

    def unnormalize(self, x):
        return (x + 1) / 2
    
    def sample_posterior(self, bs, cond):
        samples_ddim, _ = self.posterior_sampler.sample(S=self.ddim_steps,
                                conditioning=cond,
                                batch_size=bs,
                                shape=self.latent_shape,
                                verbose=False,
                                unconditional_guidance_scale=1.0,
                                unconditional_conditioning=None, 
                                eta=0.0)
        return samples_ddim
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=1e-2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, Instance2d_channel_last)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d, Conv2d_channel_last, Conv2dTran_channel_last)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            trunc_normal_(m.weight, std=1e-2)

class DC_layer(nn.Module):
    def __init__(self, DC_type, soft=False):
        super(DC_layer, self).__init__()
        self.soft = soft
        if self.soft:
            if DC_type != '':
                self.dc_weight = nn.Parameter(torch.Tensor([1]))
            else:
                self.dc_weight = None
    def forward(self, k_rec, mask, k_ref):
        if len(mask.shape) < len(k_rec.shape):
            mask = mask.unsqueeze(-1)
        masknot = 1 - mask
        if self.soft:
            k_out = masknot * k_rec + mask * k_rec * (1 - self.dc_weight) + mask * k_ref * self.dc_weight
        else:
            k_out = masknot * k_rec + mask * k_ref
        return k_out