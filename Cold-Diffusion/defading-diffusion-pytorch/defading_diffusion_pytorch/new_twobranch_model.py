import math
import torch
import torch.nn as nn

from .st_branch_model.utils import AMPLoss, PhaLoss


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


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        return self.transformer(x)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.attention(query, key, value)
        return attn_output, attn_weights




class FreBlock(nn.Module):
    def __init__(self, channels, embed_dim = 256):
        super(FreBlock, self).__init__()

        num_heads = 8

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(
            TransformerBlock(embed_dim, num_heads, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            TransformerBlock(embed_dim, num_heads, embed_dim),
            nn.ReLU())

        self.pha_fuse = nn.Sequential(
            TransformerBlock(embed_dim, num_heads, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            TransformerBlock(embed_dim, num_heads, embed_dim)
        )

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)


        # self.transformer = TransformerBlock(embed_dim, num_heads, feedforward_dim)
        self.cross_attention = CrossAttention(embed_dim, num_heads)
        self.cross_attention_2 = CrossAttention(embed_dim, num_heads)


    def forward(self, x, k=None):
        _, _, H, W = x.shape
        # k shape, msF_component_fuse shape torch.Size([24, 1, 128, 128]) torch.Size([24, 256, 16, 9])

        # rfft2 输出的形状 (半频谱): (rows, cols//2 + 1)
        half_W = W // 2 + 1
        # down-scale
        k = torch.nn.functional.interpolate(k, size=(H, W), mode='bilinear',
                                            align_corners=False).cuda()
        k = k[...,:half_W]


        fpre = self.fpre(x)
        msF = torch.fft. rfft2(fpre + 1e-8, norm='ortho')
        msF = torch.fft.fftshift(msF, dim=[2, 3])

        msF_ori= msF.clone() # * k


        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        batch_size, channels, height, width = msF_amp.shape
        msF_amp_flatten = msF_amp.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, H*W, channels)
        msF_pha_flatten = msF_pha.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, H*W, channels)
        # print("msF_amp_flatten shape", msF_amp_flatten.shape)

        # channels = msF_amp.shape[1]
        msF_amp_flatten, _ = self.cross_attention( msF_amp_flatten, msF_pha_flatten, msF_pha_flatten)
        msF_pha_flatten, _ = self.cross_attention_2(msF_pha_flatten, msF_amp_flatten, msF_amp_flatten)

        amplitude_features = self.amp_fuse(msF_amp_flatten) # + msF_component
        angle_features = self.pha_fuse(msF_pha_flatten) # + msF_component

        # cross attention
        amp_fuse = amplitude_features.permute(0, 2, 1).view(batch_size, channels, height, width)
        pha_fuse = angle_features.permute(0, 2, 1).view(batch_size, channels, height, width)

        amp_fuse = nn.ReLU()(amp_fuse)
        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8

        out = torch.complex(real, imag) + 1e-8
        out = out  + msF_ori    # * (1 - k)

        out = torch.fft.ifftshift(out, dim=[2, 3])
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='ortho'))
        out = self.post(out)



        out = torch.nan_to_num(out, nan=1e-5, posinf=1, neginf=-1)

        return out


class Branch(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4


        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels * 2,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            fre  = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
                    fre.append(FreBlock(channels=block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            down.fre = fre
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.mid_fre = FreBlock(channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            fre  = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
                    fre.append(FreBlock(channels=block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            up.fre = fre
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)



class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4


        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        self.spatial = Branch(ch=ch, out_ch=out_ch, ch_mult=ch_mult,
                                     num_res_blocks=num_res_blocks,
                                     attn_resolutions=attn_resolutions,
                                     dropout=dropout, resamp_with_conv=resamp_with_conv,
                                     in_channels=in_channels, resolution=resolution)

        self.amploss = AMPLoss()  # .to(self.device, non_blocking=True)
        self.phaloss = PhaLoss()  # .to(self.device, non_blocking=True)

        self.use_front_fre = False
        self.use_after_fre = False
        print("=== use front fre", self.use_front_fre)   # NAN
        print("=== use after fre", self.use_after_fre)   # use_after_fre_ BUG NAN

    def forward(self, x, aux, k, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # k = k.to(x.device)

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x_in = torch.cat((x, aux), dim=1)

        # spatial downsampling
        hs = [self.spatial.conv_in(x_in)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.spatial.down[i_level].block[i_block](hs[-1], temb)
                if len(self.spatial.down[i_level].attn) > 0:
                    if self.use_front_fre:
                        h = self.spatial.down[i_level].fre[i_block](h, k)
                    h = self.spatial.down[i_level].attn[i_block](h)

                    if self.use_after_fre:
                        h = self.spatial.down[i_level].fre[i_block](h, k) + h

                hs.append(h)

            if i_level != self.num_resolutions-1:
                hs.append(self.spatial.down[i_level].downsample(hs[-1]))

        # spatial middle
        h = hs[-1]
        h = self.spatial.mid.block_1(h, temb)
        h = self.spatial.mid.attn_1(h)
        h = self.spatial.mid.block_2(h, temb)

        # if self.use_front_fre or self.use_after_fre:
        h = self.spatial.mid_fre(h, k) # + h  # NAN??

        # spatial upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.spatial.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.spatial.up[i_level].attn) > 0:
                    if self.use_front_fre:
                        h = self.spatial.up[i_level].fre[i_block](h, k)
                    h = self.spatial.up[i_level].attn[i_block](h)
                    if self.use_after_fre:
                        h = self.spatial.up[i_level].fre[i_block](h, k) + h

                # TODO residual
                # h += hs.pop()

            if i_level != 0:
                h = self.spatial.up[i_level].upsample(h)

        # spatial end
        h = self.spatial.norm_out(h)
        h = nonlinearity(h)
        h = self.spatial.conv_out(h)

        return h
