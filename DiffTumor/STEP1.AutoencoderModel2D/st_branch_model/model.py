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
    

class TwoBranchModel(pl.LightningModule):
    def __init__(self, args):
        super(TwoBranchModel, self).__init__()

        num_group = 4
        num_every_group = args.model.base_num_every_group
        self.args = args

        self.init_T2_frq_branch(args)
        self.init_T2_spa_branch(args, num_every_group)
        self.init_T2_fre_spa_fusion(args)

        self.init_T1_frq_branch(args)
        self.init_T1_spa_branch(args, num_every_group)

        self.init_modality_fre_fusion(args)
        self.init_modality_spa_fusion(args)
        
        self.image_discriminator = NLayerDiscriminator(
            args.dataset.image_channels, args.model.disc_channels, args.model.disc_layers, norm_layer=nn.BatchNorm2d)
        self.video_discriminator = NLayerDiscriminator3D(
            args.dataset.image_channels, args.model.disc_channels, args.model.disc_layers, norm_layer=nn.BatchNorm3d)

        self.amploss = AMPLoss() #.to(self.device, non_blocking=True)
        self.phaloss = PhaLoss() # .to(self.device, non_blocking=True)

        if args.model.disc_loss_type == 'vanilla':
                self.disc_loss = vanilla_d_loss
        elif args.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.gan_feat_weight = args.model.gan_feat_weight
        self.image_gan_weight = args.model.image_gan_weight
        self.video_gan_weight = args.model.video_gan_weight

        self.perceptual_weight = args.model.perceptual_weight

        self.l1_weight = args.model.l1_weight
        self.save_hyperparameters()
        

    def init_T2_frq_branch(self, args):
        ### T2frequency branch
        modules_head_fre = [common.ConvBNReLU2D(1, out_channels=args.model.num_features,
                                            kernel_size=3, padding=1, act=args.model.act)]
        self.head_fre = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [common.DownSample(args.model.num_features, False, False),
                            common.FreBlock9(args.model.num_features, args)
                        ]

        self.down1_fre = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_down2_fre = [common.DownSample(args.model.num_features, False, False),
                        common.FreBlock9(args.model.num_features, args)
                        ]
        self.down2_fre = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_down3_fre = [common.DownSample(args.model.num_features, False, False),
                        common.FreBlock9(args.model.num_features, args)
                        ]
        self.down3_fre = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_neck_fre = [common.FreBlock9(args.model.num_features, args)
                        ]
        self.neck_fre = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_up1_fre = [common.UpSampler(2, args.model.num_features),
                        common.FreBlock9(args.model.num_features, args)
                        ]
        self.up1_fre = nn.Sequential(*modules_up1_fre)
        self.up1_fre_mo = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_up2_fre = [common.UpSampler(2, args.model.num_features),
                    common.FreBlock9(args.model.num_features, args)
                        ]
        self.up2_fre = nn.Sequential(*modules_up2_fre)
        self.up2_fre_mo = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_up3_fre = [common.UpSampler(2, args.model.num_features),
                    common.FreBlock9(args.model.num_features, args)
                        ]
        self.up3_fre = nn.Sequential(*modules_up3_fre)
        self.up3_fre_mo = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        # define tail module
        modules_tail_fre = [
            common.ConvBNReLU2D(args.model.num_features, out_channels=args.model.num_channels, kernel_size=3, padding=1,
                        act=args.model.act)]
        self.tail_fre = nn.Sequential(*modules_tail_fre)

    def init_T2_spa_branch(self, args, num_every_group):
        ### spatial branch
        modules_head = [common.ConvBNReLU2D(1, out_channels=args.model.num_features,
                                            kernel_size=3, padding=1, act=args.model.act)]
        self.head = nn.Sequential(*modules_head)

        modules_down1 = [common.DownSample(args.model.num_features, False, False),
                         common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down1 = nn.Sequential(*modules_down1)


        self.down1_mo = nn.Sequential(common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_down2 = [common.DownSample(args.model.num_features, False, False),
                         common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down2 = nn.Sequential(*modules_down2)

        self.down2_mo = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_down3 = [common.DownSample(args.model.num_features, False, False),
                         common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down3 = nn.Sequential(*modules_down3)
        self.down3_mo = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_neck = [common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.neck = nn.Sequential(*modules_neck)

        self.neck_mo = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_up1 = [common.UpSampler(2, args.model.num_features),
                       common.ResidualGroup(
                           args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up1 = nn.Sequential(*modules_up1)

        self.up1_mo = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_up2 = [common.UpSampler(2, args.model.num_features),
                       common.ResidualGroup(
                           args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up2 = nn.Sequential(*modules_up2)
        self.up2_mo = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))


        modules_up3 = [common.UpSampler(2, args.model.num_features),
                       common.ResidualGroup(
                           args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up3 = nn.Sequential(*modules_up3)
        self.up3_mo = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        # define tail module
        modules_tail = [
            common.ConvBNReLU2D(args.model.num_features, out_channels=args.model.num_channels, kernel_size=3, padding=1,
                         act=args.model.act)]

        self.tail = nn.Sequential(*modules_tail)

    def init_T2_fre_spa_fusion(self, args):
        ### T2 frq & spa fusion part
        conv_fuse = []
        for i in range(14):
            conv_fuse.append(common.FuseBlock7(args.model.num_features))
        self.conv_fuse = nn.Sequential(*conv_fuse)

    def init_T1_frq_branch(self, args):
        ### T2frequency branch
        modules_head_fre = [common.ConvBNReLU2D(1, out_channels=args.model.num_features,
                                            kernel_size=3, padding=1, act=args.model.act)]
        self.head_fre_T1 = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [common.DownSample(args.model.num_features, False, False),
                            common.FreBlock9(args.model.num_features, args)
                        ]

        self.down1_fre_T1 = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_down2_fre = [common.DownSample(args.model.num_features, False, False),
                        common.FreBlock9(args.model.num_features, args)
                        ]
        self.down2_fre_T1 = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_down3_fre = [common.DownSample(args.model.num_features, False, False),
                        common.FreBlock9(args.model.num_features, args)
                        ]
        self.down3_fre_T1 = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.model.num_features, args))

        modules_neck_fre = [common.FreBlock9(args.model.num_features, args)
                        ]
        self.neck_fre_T1 = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.model.num_features, args))

    def init_T1_spa_branch(self, args, num_every_group):
        ### spatial branch
        modules_head = [common.ConvBNReLU2D(1, out_channels=args.model.num_features,
                                            kernel_size=3, padding=1, act=args.model.act)]
        self.head_T1 = nn.Sequential(*modules_head)

        modules_down1 = [common.DownSample(args.model.num_features, False, False),
                         common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down1_T1 = nn.Sequential(*modules_down1)


        self.down1_mo_T1 = nn.Sequential(common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_down2 = [common.DownSample(args.model.num_features, False, False),
                         common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down2_T1 = nn.Sequential(*modules_down2)

        self.down2_mo_T1 = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_down3 = [common.DownSample(args.model.num_features, False, False),
                         common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down3_T1 = nn.Sequential(*modules_down3)
        self.down3_mo_T1 = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

        modules_neck = [common.ResidualGroup(
                             args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.neck_T1 = nn.Sequential(*modules_neck)

        self.neck_mo_T1 = nn.Sequential(common.ResidualGroup(
            args.model.num_features, 3, 4, act=args.model.act, n_resblocks=num_every_group, norm=None))

    
    def configure_optimizers(self):
        lr = self.args.model.lr
        
        net_module_set = [self.head_fre, self.down1_fre, self.down1_fre_mo, self.down2_fre, self.down2_fre_mo, self.down3_fre, self.down3_fre_mo, self.neck_fre, self.neck_fre_mo, self.up1_fre,
                            self.up1_fre_mo,
                            self.up2_fre, self.up2_fre_mo,
                            self.up3_fre, self.up3_fre_mo, self.tail_fre,

                            self.head, self.down1, self.down2, self.down3,
                            self.down1_mo, self.down2_mo, self.down3_mo, self.neck, self.neck_mo,
                            self.up1, self.up2, self.up3, 
                            self.up1_mo, self.up2_mo, self.up3_mo, 
                            self.tail, 

                            self.conv_fuse,

                            self.head_fre_T1, self.down1_fre_T1, self.down1_fre_mo_T1, self.down2_fre_T1, self.down2_fre_mo_T1, self.down3_fre_T1, self.down3_fre_mo_T1, self.neck_fre_T1, self.neck_fre_mo_T1, 


                            self.head_T1, self.down1_T1, self.down2_T1, self.down3_T1,
                            self.down1_mo_T1, self.down2_mo_T1, self.down3_mo_T1, self.neck_T1, self.neck_mo_T1]
        params = []
        for module in net_module_set:
            params += list(module.parameters())
        
        opt_ae = torch.optim.Adam(params,
                                  lr=lr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
                                    list(self.video_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        
        scheduler_ae = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(opt_ae, milestones=[40000,50000,60000], gamma=0.1),
                'interval': 'step',
                'frequency': 1
            }
        scheduler_disc = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=[40000,50000,60000], gamma=0.1),
                'interval': 'step',
                'frequency': 1
            }
        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]
    
    
    def init_modality_fre_fusion(self, args):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(common.Modality_FuseBlock6(args.model.num_features))
        self.conv_fuse_fre = nn.Sequential(*conv_fuse)

    def init_modality_spa_fusion(self, args):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(common.Modality_FuseBlock6(args.model.num_features))
        self.conv_fuse_spa = nn.Sequential(*conv_fuse)

    def forward(self, main, aux):
        #### T1 fre encoder  # T1
        t1_fre = self.head_fre_T1(aux) # 128

        down1_fre_t1 = self.down1_fre_T1(t1_fre)# 64
        down1_fre_mo_t1 = self.down1_fre_mo_T1(down1_fre_t1)

        down2_fre_t1 = self.down2_fre_T1(down1_fre_mo_t1) # 32
        down2_fre_mo_t1 = self.down2_fre_mo_T1(down2_fre_t1)

        down3_fre_t1 = self.down3_fre_T1(down2_fre_mo_t1) # 16
        down3_fre_mo_t1 = self.down3_fre_mo_T1(down3_fre_t1)

        neck_fre_t1 = self.neck_fre_T1(down3_fre_mo_t1) # 16
        neck_fre_mo_t1 = self.neck_fre_mo_T1(neck_fre_t1)


        #### T2 fre encoder and T1 & T2 fre fusion
        x_fre = self.head_fre(main) # 128
        x_fre_fuse = self.conv_fuse_fre[0](t1_fre, x_fre)

        down1_fre = self.down1_fre(x_fre_fuse)# 64
        down1_fre_mo = self.down1_fre_mo(down1_fre)
        down1_fre_mo_fuse = self.conv_fuse_fre[1](down1_fre_mo_t1, down1_fre_mo)

        down2_fre = self.down2_fre(down1_fre_mo_fuse) # 32
        down2_fre_mo = self.down2_fre_mo(down2_fre)
        down2_fre_mo_fuse = self.conv_fuse_fre[2](down2_fre_mo_t1, down2_fre_mo)

        down3_fre = self.down3_fre(down2_fre_mo_fuse) # 16
        down3_fre_mo = self.down3_fre_mo(down3_fre)
        down3_fre_mo_fuse = self.conv_fuse_fre[3](down3_fre_mo_t1, down3_fre_mo)

        neck_fre = self.neck_fre(down3_fre_mo_fuse) # 16
        neck_fre_mo = self.neck_fre_mo(neck_fre)
        neck_fre_mo_fuse = self.conv_fuse_fre[4](neck_fre_mo_t1, neck_fre_mo)


        #### T2 fre decoder
        neck_fre_mo = neck_fre_mo_fuse + down3_fre_mo_fuse

        up1_fre = self.up1_fre(neck_fre_mo) # 32
        up1_fre_mo = self.up1_fre_mo(up1_fre)
        up1_fre_mo = up1_fre_mo + down2_fre_mo_fuse

        up2_fre = self.up2_fre(up1_fre_mo) # 64
        up2_fre_mo = self.up2_fre_mo(up2_fre)
        up2_fre_mo = up2_fre_mo + down1_fre_mo_fuse

        up3_fre = self.up3_fre(up2_fre_mo) # 128
        up3_fre_mo = self.up3_fre_mo(up3_fre)
        up3_fre_mo = up3_fre_mo + x_fre_fuse

        res_fre = self.tail_fre(up3_fre_mo)

        #### T1 spa encoder
        x_t1 = self.head_T1(aux)  # 128

        down1_t1 = self.down1_T1(x_t1) # 64
        down1_mo_t1 = self.down1_mo_T1(down1_t1)

        down2_t1 = self.down2_T1(down1_mo_t1) # 32
        down2_mo_t1 = self.down2_mo_T1(down2_t1)  # 32

        down3_t1 = self.down3_T1(down2_mo_t1) # 16
        down3_mo_t1 = self.down3_mo_T1(down3_t1)  # 16

        neck_t1 = self.neck_T1(down3_mo_t1) # 16
        neck_mo_t1 = self.neck_mo_T1(neck_t1)

        #### T2 spa encoder and fusion
        x = self.head(main)  # 128
        
        x_fuse = self.conv_fuse_spa[0](x_t1, x)
        down1 = self.down1(x_fuse) # 64
        down1_fuse = self.conv_fuse[0](down1_fre, down1)
        down1_mo = self.down1_mo(down1_fuse)
        down1_fuse_mo = self.conv_fuse[1](down1_fre_mo_fuse, down1_mo)

        down1_fuse_mo_fuse = self.conv_fuse_spa[1](down1_mo_t1, down1_fuse_mo)
        down2 = self.down2(down1_fuse_mo_fuse) # 32
        down2_fuse = self.conv_fuse[2](down2_fre, down2)
        down2_mo = self.down2_mo(down2_fuse)  # 32
        down2_fuse_mo = self.conv_fuse[3](down2_fre_mo, down2_mo)

        down2_fuse_mo_fuse = self.conv_fuse_spa[2](down2_mo_t1, down2_fuse_mo)
        down3 = self.down3(down2_fuse_mo_fuse) # 16
        down3_fuse = self.conv_fuse[4](down3_fre, down3)
        down3_mo = self.down3_mo(down3_fuse)  # 16
        down3_fuse_mo = self.conv_fuse[5](down3_fre_mo, down3_mo)

        down3_fuse_mo_fuse = self.conv_fuse_spa[3](down3_mo_t1, down3_fuse_mo)
        neck = self.neck(down3_fuse_mo_fuse) # 16
        neck_fuse = self.conv_fuse[6](neck_fre, neck)
        neck_mo = self.neck_mo(neck_fuse)
        neck_mo = neck_mo + down3_mo
        neck_fuse_mo = self.conv_fuse[7](neck_fre_mo, neck_mo)

        neck_fuse_mo_fuse = self.conv_fuse_spa[4](neck_mo_t1, neck_fuse_mo)
        #### T2 spa decoder
        up1 = self.up1(neck_fuse_mo_fuse) # 32
        up1_fuse = self.conv_fuse[8](up1_fre, up1)
        up1_mo = self.up1_mo(up1_fuse)
        up1_mo = up1_mo + down2_mo
        up1_fuse_mo = self.conv_fuse[9](up1_fre_mo, up1_mo)

        up2 = self.up2(up1_fuse_mo) # 64
        up2_fuse = self.conv_fuse[10](up2_fre, up2)
        up2_mo = self.up2_mo(up2_fuse)
        up2_mo = up2_mo + down1_mo
        up2_fuse_mo = self.conv_fuse[11](up2_fre_mo, up2_mo)

        up3 = self.up3(up2_fuse_mo) # 128

        up3_fuse = self.conv_fuse[12](up3_fre, up3)
        up3_mo = self.up3_mo(up3_fuse)

        up3_mo = up3_mo + x
        up3_fuse_mo = self.conv_fuse[13](up3_fre_mo, up3_mo)

        res = self.tail(up3_fuse_mo)

        return {'recon_out': res + main, 'recon_fre': res_fre + main}
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x   = batch['image']
        aux = batch['aux']
        
        x = x.squeeze(1)
        aux = aux.squeeze(1)
        
        # print(x.shape)
        # print(aux.shape)
        

        # torch.Size([8, 96, 96, 1])
        # torch.Size([16, 1, 96, 96, 96])
        # torch.Size([16, 1, 96, 96, 96])

        x   = x.permute(0,   -1, -3, -2).detach()      # [B, C, H, W]    ?
        aux = aux.permute(0, -1, -3, -2).detach()      # [B, C, H, W]    ?
        
        out = self.forward(x, aux)
        recon_out = out['recon_out']    # x_recon?
        recon_fre = out['recon_fre']
        
        if optimizer_idx == 0:
            fft_weight = 0.01
            recon_out_loss = self.get_recon_loss(recon_out, x, tag="recon_out")
            recon_fre_loss = self.get_recon_loss(recon_fre, x, tag="recon_fre")
            amp = self.amploss(recon_fre, x)
            pha = self.phaloss(recon_fre, x)
            loss = recon_out_loss + recon_fre_loss + fft_weight * ( amp + pha )
            
        elif optimizer_idx == 1:
            loss = self.get_dis_loss(recon_out, x, tag="dis")
    
        return loss
    

    def get_dis_loss(self, recon, target, tag="dis"):
        B, C, H, W = recon.shape
        # Selects one random 2D image from each 3D Image
            
        logits_image_real, _ = self.image_discriminator(target.detach())
        # logits_video_real, _ = self.video_discriminator(target.detach())

        logits_image_fake, _ = self.image_discriminator(
            recon.detach())
        # logits_video_fake, _ = self.video_discriminator(recon.detach())

        d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
        # d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
        disc_factor = adopt_weight(
            self.global_step, threshold=self.args.model.discriminator_iter_start)
        discloss = disc_factor * \
            (self.image_gan_weight*d_image_loss )

        self.log(f"train/{tag}/logits_image_real", logits_image_real.mean().detach(),
                    logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/logits_image_fake", logits_image_fake.mean().detach(),
                    logger=True, on_step=True, on_epoch=True)
        # self.log(f"train/{tag}/logits_video_real", logits_video_real.mean().detach(),
                    # logger=True, on_step=True, on_epoch=True)
        # self.log(f"train/{tag}/logits_video_fake", logits_video_fake.mean().detach(),
                    # logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/d_image_loss", d_image_loss,
                    logger=True, on_step=True, on_epoch=True)
        # self.log(f"train/{tag}/d_video_loss", d_video_loss,
                    # logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/discloss", discloss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)
        return discloss

    
    def get_recon_loss(self, recon, target, tag="recon_out"):
        recon_loss = F.l1_loss(recon, target) * self.l1_weight
        
        # Perceptual loss
        perceptual_loss = 0
        # Slice it into T, H, W random slices
        if self.perceptual_weight > 0:
            B, C, H, W = recon.shape
            # Selects one random 2D image from each 3D Image
            
            perceptual_loss = self.perceptual_model(recon, target).mean() * self.perceptual_weight

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(recon)
            # logits_video_fake, pred_video_fake = self.video_discriminator(recon)
            
            
            g_image_loss = -torch.mean(logits_image_fake)
            # g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight * g_image_loss 
            
            disc_factor = adopt_weight(
                self.global_step, threshold=self.args.model.discriminator_iter_start)
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

            recon_loss += aeloss + perceptual_loss + gan_feat_loss # commitment_loss +
            
        self.log(f"train/{tag}/g_image_loss", g_image_loss,
                    logger=True, on_step=True, on_epoch=True)
        # self.log(f"train/{tag}/g_video_loss", g_video_loss,
        #             logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/image_gan_feat_loss", image_gan_feat_loss,
                    logger=True, on_step=True, on_epoch=True)
        # self.log(f"train/{tag}/video_gan_feat_loss", video_gan_feat_loss,
        #             logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/perceptual_loss", perceptual_loss,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/recon_loss", recon_loss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{tag}/aeloss", aeloss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)
            
        return recon_loss
    

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
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
