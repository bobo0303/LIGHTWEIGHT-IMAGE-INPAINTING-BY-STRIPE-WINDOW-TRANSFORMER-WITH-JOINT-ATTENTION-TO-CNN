import logging
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class CSwinAttention_block(nn.Module):
    def __init__(self, args, dim, split_size=4, dim_out=None, num_heads=2, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.cswinln = nn.LayerNorm(args['n_embd'], eps=1e-4)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.sp = split_size

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # lepe conv from set_lepe_conv

        # self.proj = nn.LazyConv2d(dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def set_lepe_conv(self, x: torch.Tensor) -> nn.Conv2d:
        """input_size : [B, C, H', W']"""
        dim = x.size(1)
        # init lepe conv
        # self.lepe_conv = nn.LazyConv2d(dim, kernel_size=3, stride=1, padding=1, groups=dim).to(x.device)
        self.lepe_conv = nn.Conv2d(dim,dim, kernel_size=3, stride=1, padding=1, groups=dim).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input_size : [B, C, H, W]  or [B, L, C]"""

        # create a list for later concat
        attened_x = []
        attened_att = []

        if len(x.shape) == 3:  # [B, L, C]
            B, L, C = x.shape
            H = W = int(np.sqrt(L))

        elif len(x.shape) == 4:  # [B, C, H, W]
            B, C, H, W = x.shape

        assert H % self.sp == 0 and W % self.sp == 0,\
            f'H={H} or W={W} cannot be divided by split_size={self.sp} '

        condition = (H == self.sp and W == self.sp)  # feature size == split size, one attn operation
                                                     # feature size  > split size, two attn operations

        if condition:
            h = w = 1
            hsp = wsp = self.sp  # full feature
            param = [(h, hsp, w, wsp)]

        else:
            h1, hsp_1, w_1, wsp_1 = 1, H, W // self.sp, self.sp  # vertical
            h2, hsp_2, w_2, wsp_2 = H // self.sp, self.sp, 1, W  # horizontal
            param = [(h1, hsp_1, w_1, wsp_1), (h2, hsp_2, w_2, wsp_2)]

        if len(x.shape) == 3:  # already in patch-form [B, (H*W), C]
            x_patch = x
        if len(x.shape) == 4:  # [B, C, H, W] to [B, (H*W), C]
            x_patch = rearrange(x, 'b c h w -> b (h w) c')

        x_patch = self.cswinln(x_patch)
        qkv = self.to_qkv(x_patch).chunk(3, dim=-1)


        if condition:
           qkv = [qkv]

        else:
            # split channel [:C // 2] , [C // 2:] (h w) = l
            qkv = map(lambda t: rearrange(t, 'b l (split c)  -> split b l c', split=2), qkv)
            (q1, q2), (k1, k2), (v1, v2) = qkv
            qkv = [(q1, k1, v1), (q2, k2, v2)]

        for index, (x, (h, hsp, w, wsp)) in enumerate(zip(qkv, param)):

            # print(h, hsp, w, wsp)
            # cswin format
            q, k, v = map(lambda t: rearrange(t, 'b (h hsp w wsp) (c head)  -> (b h w) head (hsp wsp) c',
                                              head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp), x)

            # print(f'{q.shape=},{k.shape=}, {v.shape=} ')
            """
            from
            [B, (H*W)        , C       ]
            [b, (h hsp w wsp), (c head)]
            to
            [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
            [(b h w)            , head, (hsp wsp)  , c     ]
            Note:
            c = C / self.num_heads, head = self.mun_heads
            h = H / self.sp      , hsp  = self.sp
            w = W / self.sp      , wsp  = self.sp
            """

            # from [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head] to [(B * H/hsp * W/wsp), C, hsp, wsp]
            lepe = rearrange(v, '(b h w) head (hsp wsp) c -> (b h w) (c head) hsp wsp',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

            # set lepe_conv
            self.set_lepe_conv(lepe)  ###

            # lepe_conv
            lepe = self.lepe_conv(lepe)

            # back to [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
            lepe = rearrange(lepe, '(b h w) (c head) hsp wsp -> (b h w) head (hsp wsp) c',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)
            # print(f'{lepe.shape=}')

            # attention
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)

            x = (attn @ v) + lepe

            # [(B * H / hsp * W / wsp), head, (hsp * wsp), C / head] to[(B , C, H, W]
            x = rearrange(x, '(b h w) head (hsp wsp) c -> b (c head) (h hsp) (w wsp)',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

            attened_x.append(x)
            attened_att.append(attn)

        x = self.proj(torch.cat(attened_x, dim=1))


        return x, attened_att

class SelfAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        assert args['n_embd'] % args['n_head'] == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(args['n_embd'], args['n_embd'])
        self.query = nn.Linear(args['n_embd'], args['n_embd'])
        self.value = nn.Linear(args['n_embd'], args['n_embd'])
        # regularization
        self.attn_drop = nn.Dropout(args['attn_pdrop'])
        self.resid_drop = nn.Dropout(args['attn_pdrop'])
        # output projection
        self.proj = nn.Linear(args['n_embd'], args['n_embd'])

        self.register_buffer("mask", torch.tril(torch.ones(args['block_size'], args['block_size']))
                             .view(1, 1, args['block_size'], args['block_size']))
        self.n_head = args['n_head']

        self.args = args

    def forward(self, x):

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs) 8 8 32*32 256/8
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # att = (q.transpose(-2, -1) @ k) * (1.0 / math.sqrt(k.size(-1)))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # y = v @ att  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class cswin_Transformer(CSwinAttention_block):

    def __init__(self, args):
        super().__init__(args, dim=128, split_size=4, dim_out=None, num_heads=2, attn_drop=0., proj_drop=0., qk_scale=None)

        self.n_embd = args['n_embd']
        # self.num_heads = args['n_head']
        self.dim = args['dim']
        self.num_layers = args['num_layers']
        # self.split_size = args['split_size']
        self.cswinln = nn.LayerNorm(args['n_embd'], eps=1e-4)
        self.selfln = nn.LayerNorm(args['n_embd'])
        self.ln_mlp1 = nn.LayerNorm(args['n_embd'])
        self.ln_mlp2 = nn.LayerNorm(args['n_embd'], eps=1e-4)
        self.CSwinAttention_block = nn.ModuleList([CSwinAttention_block(args, self.dim, split_size=i, dim_out=None, num_heads=j, attn_drop=0., proj_drop=0., qk_scale=None) for i,j in zip(args['split_size'],args['head'])])
        self.atten = SelfAttention(args)     # self
        self.loop_time = args['loop_time']

        self.mlp1 = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            GELU2(),    #sigmoid
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(args['resid_pdrop']),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            GELU(),    #特別的gelu
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(args['resid_pdrop']),
        )

    def forward(self, x):
        [b, c, h, w] = x.shape
        x0 = x.clone()

        for n in range(self.num_layers):
            xo = x
            # print('layer')
            for _ in range(self.loop_time[n]):
                # print('loop')
                x_cs_att, attn = self.CSwinAttention_block[n](x)
                x_self = rearrange(x, 'b c h w -> b (h w) c')
                x_self_att, att = self.atten(self.selfln(x_self))
                x_self_att = x_self_att.permute(0, 2, 1).reshape(b, c, h, w)
                x = xo + x_cs_att
                x1 = rearrange(x, 'b c h w -> b (h w) c')
                x = x + self.mlp1(self.ln_mlp1(x1)).reshape(b, h, w, c).permute(0, 3, 1, 2) + x_self_att      # mlp1 layernorm
                x2 = rearrange(x, 'b c h w -> b (h w) c')
                x = x + self.mlp2(self.ln_mlp2(x2)).reshape(b, h, w, c).permute(0, 3, 1, 2)   # mlp2 layernorm
                x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
                x = x.contiguous()

            if n == 1:
                att2 = attn
            if n == 3:
                att4 = attn
        return x, att2, att4
