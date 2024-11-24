import torch
from torch import nn
from . import common_freq as common
import torch.nn.functional as F
import torch.distributed as dist

from .utils import shift_dim, adopt_weight, comp_getattr, hinge_d_loss, vanilla_d_loss
from .utils import AMPLoss, PhaLoss
from .lpips import LPIPS
# from vq_gan_3d.model.codebook import Codebook
import numpy as np
# from .discriminator import NLayerDiscriminator, NLayerDiscriminator3D
import pytorch_lightning as pl


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)
    
    
class ModelBackbone(nn.Module):
    def __init__(self, num_features, act, base_num_every_group, num_channels, temb_ch):
        super(ModelBackbone, self).__init__()
        
        self.num_features = num_features
        self.act = act
        self.num_channels = num_channels
        self.temb_ch = temb_ch
        
        # self.args = args
        num_every_group = base_num_every_group
        
        self.init_T2_frq_branch()
        self.init_T2_spa_branch(num_every_group)
        self.init_T2_fre_spa_fusion()

        self.init_T1_frq_branch()
        self.init_T1_spa_branch(num_every_group)

        self.init_modality_fre_fusion()
        self.init_modality_spa_fusion()
    
    def init_T2_frq_branch(self):
        ### T2frequency branch
        modules_head_fre = [common.ConvBNReLU2D(1, out_channels=self.num_features,
                                            kernel_size=3, padding=1, act=self.act, temb_ch=self.temb_ch)]
        self.head_fre = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [common.DownSample(self.num_features, False, False),
                            common.FreBlock9(self.num_features)
                        ]

        self.down1_fre = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo = nn.Sequential(common.FreBlock9(self.num_features))

        modules_down2_fre = [common.DownSample(self.num_features, False, False),
                        common.FreBlock9(self.num_features)
                        ]
        self.down2_fre = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo = nn.Sequential(common.FreBlock9(self.num_features))

        modules_down3_fre = [common.DownSample(self.num_features, False, False),
                        common.FreBlock9(self.num_features)
                        ]
        self.down3_fre = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo = nn.Sequential(common.FreBlock9(self.num_features))

        modules_neck_fre = [common.FreBlock9(self.num_features)
                        ]
        self.neck_fre = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo = nn.Sequential(common.FreBlock9(self.num_features))

        modules_up1_fre = [common.UpSampler(2, self.num_features),
                        common.FreBlock9(self.num_features)
                        ]
        self.up1_fre = nn.Sequential(*modules_up1_fre)
        self.up1_fre_mo = nn.Sequential(common.FreBlock9(self.num_features))

        modules_up2_fre = [common.UpSampler(2, self.num_features),
                    common.FreBlock9(self.num_features)
                        ]
        self.up2_fre = nn.Sequential(*modules_up2_fre)
        self.up2_fre_mo = nn.Sequential(common.FreBlock9(self.num_features))

        modules_up3_fre = [common.UpSampler(2, self.num_features),
                    common.FreBlock9(self.num_features)
                        ]
        self.up3_fre = nn.Sequential(*modules_up3_fre)
        self.up3_fre_mo = nn.Sequential(common.FreBlock9(self.num_features))

        # define tail module
        modules_tail_fre = [
            common.ConvBNReLU2D(self.num_features, out_channels=self.num_channels, kernel_size=3, padding=1,
                        act=self.act, temb_ch=self.temb_ch)]
        self.tail_fre = nn.Sequential(*modules_tail_fre)

    def init_T2_spa_branch(self, num_every_group):
        ### spatial branch
        modules_head = [common.ConvBNReLU2D(1, out_channels=self.num_features,
                                            kernel_size=3, padding=1, act=self.act, temb_ch=self.temb_ch)]
        self.head = nn.Sequential(*modules_head)

        modules_down1 = [common.DownSample(self.num_features, False, False),
                         common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act,
                             n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.down1 = nn.Sequential(*modules_down1)


        self.down1_mo = nn.Sequential(common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_down2 = [common.DownSample(self.num_features, False, False),
                         common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.down2 = nn.Sequential(*modules_down2)

        self.down2_mo = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_down3 = [common.DownSample(self.num_features, False, False),
                         common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.down3 = nn.Sequential(*modules_down3)
        self.down3_mo = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_neck = [common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.neck = nn.Sequential(*modules_neck)

        self.neck_mo = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_up1 = [common.UpSampler(2, self.num_features),
                       common.ResidualGroup(
                           self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.up1 = nn.Sequential(*modules_up1)

        self.up1_mo = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_up2 = [common.UpSampler(2, self.num_features),
                       common.ResidualGroup(
                           self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.up2 = nn.Sequential(*modules_up2)
        self.up2_mo = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))


        modules_up3 = [common.UpSampler(2, self.num_features),
                       common.ResidualGroup(
                           self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.up3 = nn.Sequential(*modules_up3)
        self.up3_mo = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        # define tail module
        modules_tail = [
            common.ConvBNReLU2D(self.num_features, out_channels=self.num_channels, kernel_size=3, padding=1,
                         act=self.act, temb_ch=self.temb_ch)]

        self.tail = nn.Sequential(*modules_tail)

    def init_T2_fre_spa_fusion(self):
        ### T2 frq & spa fusion part
        conv_fuse = []
        for i in range(14):
            conv_fuse.append(common.FuseBlock7(self.num_features))
        self.conv_fuse = nn.Sequential(*conv_fuse)

    def init_T1_frq_branch(self):
        ### T2frequency branch
        modules_head_fre = [common.ConvBNReLU2D(1, out_channels=self.num_features,
                                            kernel_size=3, padding=1, act=self.act, temb_ch=self.temb_ch)]
        self.head_fre_T1 = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [common.DownSample(self.num_features, False, False),
                            common.FreBlock9(self.num_features)
                        ]

        self.down1_fre_T1 = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo_T1 = nn.Sequential(common.FreBlock9(self.num_features))

        modules_down2_fre = [common.DownSample(self.num_features, False, False),
                        common.FreBlock9(self.num_features)
                        ]
        self.down2_fre_T1 = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo_T1 = nn.Sequential(common.FreBlock9(self.num_features))

        modules_down3_fre = [common.DownSample(self.num_features, False, False),
                        common.FreBlock9(self.num_features)
                        ]
        self.down3_fre_T1 = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo_T1 = nn.Sequential(common.FreBlock9(self.num_features))

        modules_neck_fre = [common.FreBlock9(self.num_features)
                        ]
        self.neck_fre_T1 = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo_T1 = nn.Sequential(common.FreBlock9(self.num_features))

    def init_T1_spa_branch(self, num_every_group):
        ### spatial branch
        modules_head = [common.ConvBNReLU2D(1, out_channels=self.num_features,
                                            kernel_size=3, padding=1, act=self.act, temb_ch=self.temb_ch)]
        self.head_T1 = nn.Sequential(*modules_head)

        modules_down1 = [common.DownSample(self.num_features, False, False),
                         common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.down1_T1 = nn.Sequential(*modules_down1)


        self.down1_mo_T1 = nn.Sequential(common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_down2 = [common.DownSample(self.num_features, False, False),
                         common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.down2_T1 = nn.Sequential(*modules_down2)

        self.down2_mo_T1 = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_down3 = [common.DownSample(self.num_features, False, False),
                         common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.down3_T1 = nn.Sequential(*modules_down3)
        self.down3_mo_T1 = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

        modules_neck = [common.ResidualGroup(
                             self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch)
                         ]
        self.neck_T1 = nn.Sequential(*modules_neck)

        self.neck_mo_T1 = nn.Sequential(common.ResidualGroup(
            self.num_features, 3, 4, act=self.act, n_resblocks=num_every_group, norm=None, temb_ch=self.temb_ch))

    def init_modality_fre_fusion(self):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(common.Modality_FuseBlock6(self.num_features))
        self.conv_fuse_fre = nn.Sequential(*conv_fuse)

    def init_modality_spa_fusion(self):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(common.Modality_FuseBlock6(self.num_features))
        self.conv_fuse_spa = nn.Sequential(*conv_fuse)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


        
class TwoBranchModel(pl.LightningModule):
    def __init__(self, image_channels, 
                 disc_channels, disc_layers, disc_loss_type, 
                 gan_feat_weight, image_gan_weight,
                 discriminator_iter_start,
                 perceptual_weight, l1_weight, 
                 num_features, act, base_num_every_group, num_channels):
        super(TwoBranchModel, self).__init__()

        num_group = 4
        # lr = self.lr
        self.use_fre_mix = False
        self.use_res = True

        self.temb_ch = num_channels * 4
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(num_channels,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])



        self.model = ModelBackbone(num_features, act,
                                   base_num_every_group, num_channels,
                                   temb_ch=self.temb_ch)

        
        self.amploss = AMPLoss() #.to(self.device, non_blocking=True)
        self.phaloss = PhaLoss() # .to(self.device, non_blocking=True)

        if disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.discriminator_iter_start = discriminator_iter_start

        self.gan_feat_weight = gan_feat_weight
        self.image_gan_weight = image_gan_weight


        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        self.save_hyperparameters()
        


    def forward(self, main, aux, t):
        if self.use_fre_mix:
            x_fre = common.frequency_transform(main)


        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)



        #### T1 fre encoder  # T1
        t1_fre = self.model.head_fre_T1(aux) # 128

        down1_fre_t1 = self.model.down1_fre_T1(t1_fre)# 64
        down1_fre_mo_t1 = self.model.down1_fre_mo_T1(down1_fre_t1)

        down2_fre_t1 = self.model.down2_fre_T1(down1_fre_mo_t1) # 32
        down2_fre_mo_t1 = self.model.down2_fre_mo_T1(down2_fre_t1)

        down3_fre_t1 = self.model.down3_fre_T1(down2_fre_mo_t1) # 16
        down3_fre_mo_t1 = self.model.down3_fre_mo_T1(down3_fre_t1)

        neck_fre_t1 = self.model.neck_fre_T1(down3_fre_mo_t1) # 16
        neck_fre_mo_t1 = self.model.neck_fre_mo_T1(neck_fre_t1)

        #### T2 fre encoder and T1 & T2 fre fusion
        x_fre = self.model.head_fre(main, temb) # 128
        x_fre_fuse = self.model.conv_fuse_fre[0](t1_fre, x_fre, temb)

        down1_fre = self.model.down1_fre(x_fre_fuse, temb)# 64
        down1_fre_mo = self.model.down1_fre_mo(down1_fre, temb)
        down1_fre_mo_fuse = self.model.conv_fuse_fre[1](down1_fre_mo_t1, down1_fre_mo, temb)

        down2_fre = self.model.down2_fre(down1_fre_mo_fuse, temb) # 32
        down2_fre_mo = self.model.down2_fre_mo(down2_fre, temb)
        down2_fre_mo_fuse = self.model.conv_fuse_fre[2](down2_fre_mo_t1, down2_fre_mo, temb)

        down3_fre = self.model.down3_fre(down2_fre_mo_fuse, temb) # 16
        down3_fre_mo = self.model.down3_fre_mo(down3_fre, temb)
        down3_fre_mo_fuse = self.model.conv_fuse_fre[3](down3_fre_mo_t1, down3_fre_mo, temb)

        neck_fre = self.model.neck_fre(down3_fre_mo_fuse, temb) # 16
        neck_fre_mo = self.model.neck_fre_mo(neck_fre, temb)
        neck_fre_mo_fuse = self.model.conv_fuse_fre[4](neck_fre_mo_t1, neck_fre_mo, temb)


        #### T2 fre decoder
        neck_fre_mo = neck_fre_mo_fuse + down3_fre_mo_fuse

        up1_fre = self.model.up1_fre(neck_fre_mo, temb) # 32
        up1_fre_mo = self.model.up1_fre_mo(up1_fre, temb)
        up1_fre_mo = up1_fre_mo + down2_fre_mo_fuse

        up2_fre = self.model.up2_fre(up1_fre_mo, temb) # 64
        up2_fre_mo = self.model.up2_fre_mo(up2_fre, temb)
        up2_fre_mo = up2_fre_mo + down1_fre_mo_fuse

        up3_fre = self.model.up3_fre(up2_fre_mo, temb) # 128
        up3_fre_mo = self.model.up3_fre_mo(up3_fre, temb)
        up3_fre_mo = up3_fre_mo + x_fre_fuse

        res_fre = self.model.tail_fre(up3_fre_mo, temb)

        #### T1 spa encoder
        x_t1 = self.model.head_T1(aux, temb)  # 128

        down1_t1 = self.model.down1_T1(x_t1, temb) # 64
        down1_mo_t1 = self.model.down1_mo_T1(down1_t1, temb)

        down2_t1 = self.model.down2_T1(down1_mo_t1, temb) # 32
        down2_mo_t1 = self.model.down2_mo_T1(down2_t1, temb)  # 32

        down3_t1 = self.model.down3_T1(down2_mo_t1, temb) # 16
        down3_mo_t1 = self.model.down3_mo_T1(down3_t1, temb)  # 16

        neck_t1 = self.model.neck_T1(down3_mo_t1, temb) # 16
        neck_mo_t1 = self.model.neck_mo_T1(neck_t1, temb)

        #### T2 spa encoder and fusion
        x = self.model.head(main, temb)  # 128
        
        x_fuse = self.model.conv_fuse_spa[0](x_t1, x, temb)
        down1 = self.model.down1(x_fuse, temb) # 64
        down1_fuse = self.model.conv_fuse[0](down1_fre, down1, temb)
        down1_mo = self.model.down1_mo(down1_fuse, temb)
        down1_fuse_mo = self.model.conv_fuse[1](down1_fre_mo_fuse, down1_mo, temb)

        down1_fuse_mo_fuse = self.model.conv_fuse_spa[1](down1_mo_t1, down1_fuse_mo, temb)
        down2 = self.model.down2(down1_fuse_mo_fuse, temb) # 32
        down2_fuse = self.model.conv_fuse[2](down2_fre, down2, temb)
        down2_mo = self.model.down2_mo(down2_fuse, temb)  # 32
        down2_fuse_mo = self.model.conv_fuse[3](down2_fre_mo, down2_mo, temb)

        down2_fuse_mo_fuse = self.model.conv_fuse_spa[2](down2_mo_t1, down2_fuse_mo, temb)
        down3 = self.model.down3(down2_fuse_mo_fuse, temb) # 16
        down3_fuse = self.model.conv_fuse[4](down3_fre, down3, temb)
        down3_mo = self.model.down3_mo(down3_fuse, temb)  # 16
        down3_fuse_mo = self.model.conv_fuse[5](down3_fre_mo, down3_mo, temb)

        down3_fuse_mo_fuse = self.model.conv_fuse_spa[3](down3_mo_t1, down3_fuse_mo, temb)
        neck = self.model.neck(down3_fuse_mo_fuse, temb) # 16
        neck_fuse = self.model.conv_fuse[6](neck_fre, neck, temb)
        neck_mo = self.model.neck_mo(neck_fuse, temb)
        neck_mo = neck_mo + down3_mo
        neck_fuse_mo = self.model.conv_fuse[7](neck_fre_mo, neck_mo, temb)

        neck_fuse_mo_fuse = self.model.conv_fuse_spa[4](neck_mo_t1, neck_fuse_mo, temb)
        #### T2 spa decoder
        up1 = self.model.up1(neck_fuse_mo_fuse, temb) # 32
        up1_fuse = self.model.conv_fuse[8](up1_fre, up1, temb)
        up1_mo = self.model.up1_mo(up1_fuse, temb)
        up1_mo = up1_mo + down2_mo
        up1_fuse_mo = self.model.conv_fuse[9](up1_fre_mo, up1_mo, temb)

        up2 = self.model.up2(up1_fuse_mo, temb) # 64
        up2_fuse = self.model.conv_fuse[10](up2_fre, up2, temb)
        up2_mo = self.model.up2_mo(up2_fuse, temb)
        up2_mo = up2_mo + down1_mo
        up2_fuse_mo = self.model.conv_fuse[11](up2_fre_mo, up2_mo, temb)

        up3 = self.model.up3(up2_fuse_mo, temb) # 128

        up3_fuse = self.model.conv_fuse[12](up3_fre, up3, temb)
        up3_mo = self.model.up3_mo(up3_fuse, temb)

        up3_mo = up3_mo + x
        up3_fuse_mo = self.model.conv_fuse[13](up3_fre_mo, up3_mo, temb)

        res = self.model.tail(up3_fuse_mo, temb)
        if self.use_res:
            res = res + main
            res_fre = res_fre + main

        return res,  res_fre
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.model.train()
        
        x   = batch['image']
        aux = batch['aux']
        
        x = x.squeeze(1)
        aux = aux.squeeze(1)
        
        # print(x.shape)
        # print(aux.shape)
    
        # torch.Size([8, 96, 96, 1])
        # torch.Size([16, 1, 96, 96, 96])

        x   = x.permute(0,   -1, -3, -2)#.detach()      # [B, C, H, W]   
        aux = aux.permute(0, -1, -3, -2)#.detach()      # [B, C, H, W]   
        
        out = self.forward(x, aux)
        recon_out = out['recon_out']   
        recon_fre = out['recon_fre']
            
        if optimizer_idx == 0:
            fft_weight = 0.01
            use_dis = False
            recon_out_loss = self.get_recon_loss(recon_out, x, tag="recon_out", use_dis=use_dis)
            recon_fre_loss = self.get_recon_loss(recon_fre, x, tag="recon_fre", use_dis=use_dis)
            # amp = self.amploss(recon_fre, x)
            # pha = self.phaloss(recon_fre, x)
            loss = recon_out_loss + recon_fre_loss #+ fft_weight * ( amp + pha )
            
        elif optimizer_idx == 1:
            loss = self.get_dis_loss(recon_out, x, tag="dis")
            
        # print("loss = ", loss)
        
        return loss
    

    def get_dis_loss(self, recon, target, tag="dis"):
        B, C, H, W = recon.shape
        # Selects one random 2D image from each 3D Image
            
        logits_image_real, _ = self.image_discriminator(target.detach())
        logits_image_fake, _ = self.image_discriminator(recon.detach())
        
        print("logits_image_real = ", torch.mean(logits_image_real))
        print("logits_image_fake = ", torch.mean(logits_image_fake))
        
        d_image_loss = self.disc_loss(logits_image_real , logits_image_fake)
        # print("d_image_loss = ", d_image_loss)
        
        # d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
        disc_factor = adopt_weight(
            self.global_step, threshold=self.discriminator_iter_start)
        discloss = disc_factor * (self.image_gan_weight * d_image_loss )

        self.log(f"train/{tag}/logits_image_real", logits_image_real.mean().detach(),
                    logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/logits_image_fake", logits_image_fake.mean().detach(),
                    logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/d_image_loss", d_image_loss,
                    logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/disc_loss", discloss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)
        return discloss

    
    def get_recon_loss(self, recon, target, tag="recon_out", use_dis=True):
        recon_loss = F.l1_loss(recon, target) * self.l1_weight
        # recon_loss = ((recon - target)**2).mean() * self.l1_weight
        
        # Perceptual loss
        perceptual_loss = 0
        aeloss = 0
        image_gan_feat_loss = 0
        g_image_loss = 0
        
        # Slice it into T, H, W random slices
        if self.perceptual_weight > 0:
            B, C, H, W = recon.shape
            # Selects one random 2D image from each 3D Image
            
            perceptual_loss = self.perceptual_model(recon, target).mean() * self.perceptual_weight
            recon_loss += perceptual_loss 
            
            
            if use_dis:
                # Discriminator loss (turned on after a certain epoch)
                logits_image_fake, pred_image_fake = self.image_discriminator(recon)
                # logits_video_fake, pred_video_fake = self.video_discriminator(recon)
                g_image_loss = -torch.mean(logits_image_fake)
                # g_video_loss = -torch.mean(logits_video_fake)
                g_loss = self.image_gan_weight * g_image_loss 
                
                disc_factor = adopt_weight(
                    self.global_step, threshold=self.discriminator_iter_start)
                aeloss = disc_factor * g_loss

                # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
                image_gan_feat_loss = 0
                feat_weights = 4.0 / (3 + 1)
                if self.image_gan_weight > 0:
                    logits_image_real, pred_image_real = self.image_discriminator( recon)
                    for i in range(len(pred_image_fake)-1):
                        image_gan_feat_loss += feat_weights * \
                            F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
                            )) * (self.image_gan_weight > 0)
            
                gan_feat_loss = disc_factor * self.gan_feat_weight *  (image_gan_feat_loss)
                recon_loss += gan_feat_loss + aeloss
            
                self.log(f"train/{tag}/g_image_loss", g_image_loss,
                            logger=True, on_step=True, on_epoch=True)
                self.log(f"train/{tag}/image_gan_feat_loss", image_gan_feat_loss,
                            logger=True, on_step=True, on_epoch=True)
                self.log(f"train/{tag}/aeloss", aeloss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)
                
        self.log(f"train/{tag}/perceptual_loss", perceptual_loss,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/recon_loss", recon_loss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)
            
        return recon_loss
    

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


