import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x:torch.Tensor, causal_mask=False):
        #x: batch_size, seq_len, dim

        input_shape = x.shape 
        batch_size, seq_len, dim = input_shape 
        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        #--> batch_size, seq_len, dim*3 --> 3 tensors of shape batch_size, seq_len, dim
        q, k, v= self.in_proj(x).chunk(3, dim=-1)

        #batch_size, seq_len, dim --> batch_size, seq_len, n_head, dim/H --> batch_size, n_head, seq_len, dim/H
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        # batch_size, n_heead, seq_len, seq_len
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        #batch_size, n_heead, seq_len, seq_len @ batch_size, n_head, seq_len, dim/H --> batch_size, n_head, seq_len, dim/H
        output = weight @ v

        # batch_size, seq_len, n_head, dim/H
        output = output.transpose(1,2)
        # batch_size, seq_len, dim
        output = output.view(input_shape)

        output = self.proj_out(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads 

    def forward(self, x, y):
        #x: (latent): batch, seq_len_q, dim_q
        #y: (context): batch, seq_len_kv, dim_kv = batch, 77, 768

        input_shape = x.shape 
        batch_size, seq_len, d_embed = x.shape 

        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2) 
        v = v.view(interm_shape).transpose(1,2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # no mask here 

        output = weight @ v
        output = output.transpose(1,2).contiguous()     

        output = output.view(input_shape)

        return self.out_proj(output)
      