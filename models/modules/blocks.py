import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
# from mamba_ssm import Mamba
class DualDomainFusion(nn.Module):
    def __init__(self, in_channel, inner_dim=128, num_resblocks=4):
        super().__init__()
        res_blocks = []
        res_blocks.append(BasicResBlock(conv_op=nn.Conv2d, 
                                        input_channels=in_channel, 
                                        output_channels=inner_dim,
                                        norm_op=nn.BatchNorm2d,
                                        norm_op_kwargs={},
                                        use_1x1conv=True))
        for _ in range(num_resblocks-2):
            res_blocks.append(BasicResBlock(conv_op=nn.Conv2d, 
                                                input_channels=inner_dim, 
                                                output_channels=inner_dim,
                                                norm_op=nn.BatchNorm2d,
                                                norm_op_kwargs={},
                                                use_1x1conv=False))
        res_blocks.append(BasicResBlock(conv_op=nn.Conv2d, 
                                        input_channels=inner_dim, 
                                        output_channels=in_channel,
                                        norm_op=nn.BatchNorm2d,
                                        norm_op_kwargs={},
                                        use_1x1conv=True))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.conv1x1_1 = nn.Conv2d(2, inner_dim, 1, 1, 0)
        self.conv1x1_2 = nn.Conv2d(2, inner_dim, 1, 1, 0)
        
        self.silu = nn.SiLU()
        self.to_z1 = nn.Conv2d(inner_dim, inner_dim, 1, 1, 0)
        self.to_z2 = nn.Conv2d(inner_dim, inner_dim, 1, 1, 0)
        # self.to_z = nn.Conv2d(inner_dim, inner_dim, 1, 1, 0)
        self.conv1x1_fout = nn.Conv2d(inner_dim, 2, 1, 1, 0)
        
        self.conv1x1_pout = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        
        self.for_real_img = in_channel == 1
            
    def pixel_diff_fusion(self, x, y):
        
        return self.res_blocks(x + y)
    
    def freq_fusion(self, x, y):
        # x,y (B,C,H,W)
        xf = self.conv1x1_1(fft2c(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        yf = self.conv1x1_2(fft2c(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        z1 = self.silu(self.to_z1(F.adaptive_avg_pool2d(xf + yf, (1,1))))
        z2 = self.silu(self.to_z2(F.adaptive_avg_pool2d(xf + yf, (1,1))))
        # z = F.sigmoid(self.to_z(F.adaptive_avg_pool2d(xf + yf, (1,1))))
        
        xf = xf * z1
        yf = yf * z2
        
        return ifft2c(self.conv1x1_fout(xf + yf).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    
    def freq_fusion_realimg(self, x, y):
        # x,y (B,C,H,W)
        xf = self.conv1x1_1(torch.view_as_real(fft(x)).squeeze(1).permute(0, 3, 1, 2))
        yf = self.conv1x1_2(torch.view_as_real(fft(y)).squeeze(1).permute(0, 3, 1, 2))
        
        z1 = self.silu(self.to_z1(F.adaptive_avg_pool2d(xf + yf, (1,1))))
        z2 = self.silu(self.to_z2(F.adaptive_avg_pool2d(xf + yf, (1,1))))
        # z = F.sigmoid(self.to_z(F.adaptive_avg_pool2d(xf + yf, (1,1))))
        
        xf = xf * z1
        yf = yf * z2
        
        return ifft2c(self.conv1x1_fout(xf + yf).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)[:,:1,:,:]
    
    def forward(self, x, y):
        if not self.for_real_img:
            f = self.pixel_diff_fusion(x, y) + self.freq_fusion(x, y)
        else:
            f = self.pixel_diff_fusion(x, y) + self.freq_fusion_realimg(x, y)
        return self.conv1x1_pout(f)

class LatentGuidedAttention(nn.Module):
    def __init__(self, x_dim, heads=8, dim_head=64, z_dim=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.ln_x = nn.LayerNorm(x_dim)
        self.ln_z = nn.LayerNorm(z_dim)
        
        self.to_q = nn.Linear(x_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(x_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(x_dim, inner_dim, bias=False)
        self.to_z = nn.Linear(z_dim, x_dim * 2, bias=False)
        
        self.silu = nn.SiLU()
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, x_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        h = self.heads
        z = self.silu(self.to_z(self.ln_z(z)))
        z1, z2 = torch.split(z, z.shape[-1]//2, dim=-1)
        x_norm = self.ln_x(x)
        q = self.to_q(x_norm * z1 + z2)
        k = self.to_k(x_norm * z1 + z2)
        v = self.to_v(x_norm * z1 + z2)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if len(x.shape) == 4:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x
            elif len(x.shape) == 5:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
                return x

class Conv2d_channel_last(nn.Conv2d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return F.conv2d(x.permute(0,3,1,2), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,1)

class Conv3d_channel_last(nn.Conv3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return F.conv3d(x.permute(0,4,1,2,3), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,4,1)

class Conv2dTran_channel_last(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return F.conv_transpose2d(x.permute(0,3,1,2), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,1)
    
class Conv3dTran_channel_last(nn.ConvTranspose3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return F.conv_transpose3d(x.permute(0,4,1,2,3), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,4,1)

class Instance2d_channel_last(nn.InstanceNorm3d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return self._apply_instance_norm(x.permute(0,3,1,2)).permute(0,2,3,1)
    
class Instance3d_channel_last(nn.InstanceNorm3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return self._apply_instance_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
    
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

def complex2c_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    b, h, w, two = x.shape
    assert two == 2
    return x.permute(0,3,1,2).contiguous()

def chan2_to_complex2c(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    assert c == 2
    return x.permute(0,2,3,1).contiguous()