import logging
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

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

class RowColAttention(nn.Module):

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


    def forward(self, x, mask=None, rel_pos=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs) 8 8 32*32 256/8
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))     # 8 8 32 32
        if rel_pos is not None:
            att += rel_pos
        if mask is not None:  # maybe we don't need mask in axial-transformer
            # mask:[B,1,L(1),L]
            att = att.masked_fill(mask == 1, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)   # default 0
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y

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

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q.transpose(-2, -1) @ k) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = v @ att  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Mid_Transformer_Golobal_layer(nn.Module):
    """ Transformer block with original GELU2 """

    def __init__(self,args , n_embd, n_head, H, W, add_rel_pos=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)     #LayerNorm
        self.ln2 = nn.LayerNorm(n_embd)     #LayerNorm
        self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)  # row layerNorm eps:default 1e-5
        self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)  # col layerNorm
        self.ln2_rc = nn.LayerNorm(n_embd, eps=1e-4)  # base layerNorm
        self.attn_r = RowColAttention(args)     # row
        self.attn_c = RowColAttention(args)     # col
        self.attn = SelfAttention(args)     # self
        self.mlp1 = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU2(),#sigmoid
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(args['resid_pdrop']),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),    #特別的gelu
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(args['resid_pdrop']),
        )
        self.add_rel_pos = add_rel_pos  # default True
        self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        if row:
            rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        rel_pos = rel_pos.contiguous()
        return rel_pos

    # def oblique_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
    #     # hidden_states:[B,L,D], [1,L]
    #     position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
    #     # [1,1,L]-[1,L,1]-->[1,L,L]
    #     rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
    #     rel_pos_mat -= torch.min(rel_pos_mat)
    #     # [1,L,L]->[1,L,L,D]
    #     rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
    #     # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
    #     if row:
    #         rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
    #     else:
    #         rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
    #
    #     rel_pos = rel_pos.contiguous()
    #     return rel_pos

    def forward(self, x):
        [b, c, h, w] = x.shape
        mask_row = None
        mask_col = None

        x0 = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        # ROW ATTENTION
        x_row = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        if self.add_rel_pos:
            row_rel_pos = self._cal_1d_pos_emb(x_row, rel_pos_onehot_size=h, row=True)
        else:
            row_rel_pos = None

        # COL ATTENTION
        x_col = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
        if self.add_rel_pos:
            col_rel_pos = self._cal_1d_pos_emb(x_col, rel_pos_onehot_size=w, row=False)
        else:
            col_rel_pos = None

        x_row = self.attn_r(self.rln1(x_row), mask_row, rel_pos=row_rel_pos)
        x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)
        x_col = self.attn_c(self.cln1(x_col), mask_col, rel_pos=col_rel_pos)
        x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)
        x_self = self.attn(self.ln1(x0))

        x = x0 + x_row + x_col
        x = x + self.mlp1(self.ln2_rc(x))
        # x_self = self.attn(self.ln1(x))
        x = x + x_self
        x = x + self.mlp2(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x.contiguous()
        return x

class Mid_Golobal_layer(Mid_Transformer_Golobal_layer):

    def __init__(self, args):
        super().__init__(args, args['n_embd'], args['n_head'], 32, 32, add_rel_pos=True)
