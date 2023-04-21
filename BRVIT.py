import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


class BRViT(nn.Module):
    def __init__(self, patch_size, din, dmodel, dff, nheads, nlayers, dout, out_activation, dropout=0.1):
        super(BRViT, self).__init__()

        self.patch_size = patch_size
        n, c, h, w = din
        npatches = h * w // patch_size ** 2

        self.input_layer = nn.Linear(c * (patch_size ** 2), dmodel)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dmodel, nheads, dff, activation='gelu', dropout=dropout, batch_first=True,
                                       norm_first=True), nlayers)
        self.brtransformer = BlockRecurrentTransformerCell(dmodel, dff, nheads, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(dmodel), nn.Flatten(), nn.Linear(npatches * dmodel, dout))
        self.out = out_activation
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, npatches, dmodel))

    def forward(self, x, h):
        x = self.patch(x)
        x = self.input_layer(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x, h = self.brtransformer(x, h)
        x = self.head(x)
        if self.out:
            x = self.out(x)
        return x, h

    def patch(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1, 2)
        x = x.flatten(2, 4)
        return x


class BlockRecurrentTransformerCell(nn.Module):
    def __init__(self, dmodel, dff, nheads, dropout=0.1):
        super(BlockRecurrentTransformerCell, self).__init__()
        self.nhead = nheads

        self.Qvx = nn.Linear(dmodel, dmodel, bias=False)
        self.Qhx = nn.Linear(dmodel, dmodel, bias=False)
        self.Qvh = nn.Linear(dmodel, dmodel, bias=False)
        self.Qhh = nn.Linear(dmodel, dmodel, bias=False)

        self.Kx = nn.Linear(dmodel, dmodel, bias=False)
        self.Kh = nn.Linear(dmodel, dmodel, bias=False)

        self.Vx = nn.Linear(dmodel, dmodel, bias=False)
        self.Vh = nn.Linear(dmodel, dmodel, bias=False)

        self.linear_projv = nn.Linear(dmodel * 2, dmodel)
        self.linear_projh = nn.Linear(dmodel * 2, dmodel)

        self.drop_v = nn.Dropout(dropout)
        self.drop_h = nn.Dropout(dropout)

        self.mlp_v = nn.Sequential(nn.Linear(dmodel, dff), nn.GELU(), nn.Linear(dff, dmodel), nn.Dropout(dropout))

        self.layer_normx = nn.LayerNorm(dmodel)
        self.layer_normh = nn.LayerNorm(dmodel)
        self.layer_norm_ver = nn.LayerNorm(dmodel)

        self.gate = nn.Sigmoid()
        self.gate_bias = nn.Parameter(torch.randn(dmodel))

    def scaleddotproduct(self, q, k, v):
        batch_size, head, length, d_k = k.size()

        k_t = k.transpose(2, 3)
        attn = F.softmax(torch.matmul(q, k_t) / math.sqrt(d_k))
        v = torch.matmul(attn, v)

        return v, attn

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.nhead
        tensor = tensor.view(batch_size, length, self.nhead, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

    def forward(self, x, h):
        inp_x = self.layer_normx(x)
        inp_h = self.layer_normh(h)

        # Query matrices
        qvx = self.Qvx(inp_x)  # self attention input query for vertical
        qvh = self.Qvh(inp_x)  # cross attention input query for vertical
        qhh = self.Qhh(inp_h)  # self attention hidden query for horizontal
        qhx = self.Qhx(inp_h)  # cross attention hidden query for horizontal

        qvx = self.split(qvx)  # Split for each head
        qvh = self.split(qvh)  # Split for each head
        qhh = self.split(qhh)  # Split for each head
        qhx = self.split(qhx)  # Split for each head

        # Input Key, Value
        kx = self.Kx(inp_x)
        vx = self.Vx(inp_x)

        kx = self.split(kx)  # Split for each head
        vx = self.split(vx)  # Split for each head

        # Hidden Key, Value
        kh = self.Kh(inp_h)
        vh = self.Vh(inp_h)

        kh = self.split(kh)  # Split for each head
        vh = self.split(vh)  # Split for each head

        ############################################
        # Calculate vertical direction
        ############################################

        # Compute self attention
        self_attn_v, attn = self.scaleddotproduct(qvx, kx, vx)

        # Compute cross attention
        cross_attn_v, attn = self.scaleddotproduct(qvh, kh, vh)

        # Concatenate attention from each head
        self_attn_v = self.concat(self_attn_v)
        cross_attn_v = self.concat(cross_attn_v)

        # Concatenate the attention matrices
        combined_attn_v = torch.concat((cross_attn_v, self_attn_v), dim=-1)

        # Project attn
        proj_v = self.linear_projv(combined_attn_v)

        proj_v = self.drop_v(proj_v)

        # Residual connection
        v_x = proj_v + inp_x

        # Feed forward
        ff_x = self.layer_norm_ver(v_x)
        ff_x = self.mlp_v(x)
        v_out = v_x + ff_x

        #############################################
        # Calculate horizontal direction
        #############################################

        # Compute self attention
        self_attn_h, attn = self.scaleddotproduct(qhh, kh, vh)

        # Compute cross attention
        cross_attn_h, attn = self.scaleddotproduct(qhx, kx, vx)

        # Concatenate attention from each head
        self_attn_h = self.concat(self_attn_h)
        cross_attn_h = self.concat(cross_attn_h)

        # Concatenate the attention matrices
        combined_attn_h = torch.concat((cross_attn_h, self_attn_h), dim=-1)

        # Project attn
        proj_h = self.linear_projh(combined_attn_h)

        proj_h = self.drop_h(proj_h)

        # Skip gate - fixed
        g = self.gate(self.gate_bias)
        h_out = (proj_h * (1 - g)) + (inp_h * g)

        return v_out, h_out

