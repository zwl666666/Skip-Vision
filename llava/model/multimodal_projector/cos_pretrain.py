
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

import math
import numpy as np
from collections import OrderedDict

from timm.models.layers import trunc_normal_

from .cos_configs import COS_CONFIGS

class CrossAttnV3(nn.Module):
    def __init__(self, query_dim, context_dim, n_head, inner_dim=None):
        super().__init__()
        self.n_head = n_head
        inner_dim = query_dim if inner_dim is None else inner_dim
        self.scale = (inner_dim//n_head)**-0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim*2, bias=False)
        self.proj = nn.Linear(inner_dim, query_dim)

        self.query_norm = nn.LayerNorm(query_dim)
        self.cont_norm = nn.LayerNorm(context_dim)

    def forward(self, q, context):
        q = self.query_norm(q)
        kv = self.cont_norm(context)
        b, win_size, c = kv.shape

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h c) -> (b h) n c", h=self.n_head), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = rearrange((attn @ v), "(b h) n c -> b n (h c)", h=self.n_head) #* (win_size**-0.5)
        out = self.proj(out)
        return out

class RandomBipartiteSoftMatching(nn.Module):
    def __init__(self,num_tokens):
        super(RandomBipartiteSoftMatching, self).__init__()
        self.num_tokens = num_tokens

    def forward(self, metric):
        r = self.num_tokens
        if r==0:
            return metric
        else:
            #r = self.num_tokens
            B, N, _ = metric.shape
            x = metric
            with torch.no_grad():
                metric = metric / metric.norm(dim=-1, keepdim=True)
                sim = torch.matmul(metric, metric.transpose(-1, -2))
                diag_elements = torch.diagonal(sim, dim1=-2, dim2=-1)
                diag = torch.diag_embed(diag_elements)
                sim = (sim - diag).sum(dim=-1)
                idx = sim.argsort(dim=-1, descending=True)[..., None]
                a_idx = idx[:, :r, :]
                b_idx = idx[:, r:, :]
                C = metric.shape[-1]
                a = metric.gather(dim=1, index=a_idx.expand(B, r, C))
                b = metric.gather(dim=1, index=b_idx.expand(B, N - r, C))
                scores = torch.matmul(a, b.transpose(-1, -2))
                _, dst_idx = scores.max(dim=-1)
                dst_idx = dst_idx[..., None]
            C = x.shape[-1]
            src = x.gather(dim=1, index=a_idx.expand(B, r, C))
            dst = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            C = src.shape[-1]
            dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce="mean")
            return dst

class CosResampler(nn.Module):
    def __init__(self, cos_config_version):
        super().__init__()

        cos_config = COS_CONFIGS[cos_config_version]

        tokens_per_side = cos_config.get("cos_tokens_per_side")
        window_sizes    = cos_config.get("cos_window_sizes")
        n_query         = cos_config.get("cos_n_query")
        feat_levels     = cos_config.get("cos_feat_levels")
        seq_feat_levels = cos_config.get("cos_seq_feat_levels")
        dim             = cos_config.get("cos_out_dim")
        context_dim     = cos_config.get("cos_context_dim")
        llm_dim         = cos_config.get("llm_dim")

        n_layers        = 1
        n_head          = dim // 64
        dropout         = 0.0

        assert context_dim is not None


        self.window_sizes           = window_sizes
        self.n_query                = n_query
        self.feat_levels            = feat_levels
        self.seq_feat_levels        = seq_feat_levels
        self.n_tokens_per_window    = []
        self.ada = nn.Sequential(nn.LayerNorm(dim),
                                    nn.Linear(dim, 1))
        for qid, nq in enumerate(self.n_query):
            assert nq >= ((tokens_per_side//self.window_sizes[qid])**2), f"Required query num for window size {self.window_sizes[qid]} is at minimum {((tokens_per_side//self.window_sizes[qid])**2)}, got {nq}"
            print(f"Setting cls_token_w{self.window_sizes[qid]}_q{self.n_query[qid]}...")
            setattr(self, f"cls_token_w{self.window_sizes[qid]}_q{self.n_query[qid]}", nn.Parameter(torch.zeros(1, nq, dim)))
            trunc_normal_(getattr(self, f"cls_token_w{self.window_sizes[qid]}_q{self.n_query[qid]}"), std=0.02)
            self.n_tokens_per_window.append(nq//((tokens_per_side//self.window_sizes[qid])**2))
        self.n_layers = n_layers

        for i in range(n_layers):
            for feat_level in self.seq_feat_levels:
                for wid, window_size in enumerate(self.window_sizes):
                    setattr(self, f"attn_{i}_w{window_size}_l{feat_level}_q{self.n_query[wid]}", CrossAttnV3(dim, context_dim, n_head))
                    setattr(self, f"mlp_{i}_w{window_size}_l{feat_level}_q{self.n_query[wid]}", nn.Sequential(OrderedDict([
                        ("fc1", nn.Linear(dim, dim*4)),
                        ("gelu", nn.GELU()),
                        ("fc2", nn.Linear(dim*4, dim)),
                        ("dropout", nn.Dropout(dropout))
                    ])))
                    setattr(self, f"ln_{i}_w{window_size}_l{feat_level}_q{self.n_query[wid]}", nn.LayerNorm(dim))

        self.ln_out = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, llm_dim)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[RANK: {rank}] Initializing weights for WindowedPoolerMultiResMultiScaleV3Init.")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            trunc_normal_(module.weight.data, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):

        arbitrary_existing_feat_name = list(x.keys())[0]

        b = x[arbitrary_existing_feat_name].shape[0]

        cls_tokens = []
        windows = []
        for levelid, feat_level in enumerate(self.seq_feat_levels):
            windows_level = []
            for wid, window_size in enumerate(self.window_sizes):
                cls_token = getattr(self, f"cls_token_w{self.window_sizes[wid]}_q{self.n_query[wid]}")
                cls_token, window, _ = self.window_partition(cls_token.repeat(b, 1, 1), x[f'block_{feat_level}'], self.window_sizes[wid], self.n_tokens_per_window[wid])
                if levelid == 0:
                    cls_tokens.append(cls_token)
                windows_level.append(window)
            windows.append(windows_level)

        for i in range(self.n_layers):
            for levelid, feat_level in enumerate(self.seq_feat_levels):
                for wid, window_size in enumerate(self.window_sizes):
                    cls_tokens[wid] = cls_tokens[wid] + getattr(self, f"attn_{i}_w{window_size}_l{feat_level}_q{self.n_query[wid]}")(cls_tokens[wid], windows[levelid][wid])
                    cls_tokens[wid] = cls_tokens[wid] + getattr(self, f"mlp_{i}_w{window_size}_l{feat_level}_q{self.n_query[wid]}")(getattr(self, f"ln_{i}_w{window_size}_l{feat_level}_q{self.n_query[wid]}")(cls_tokens[wid]))

        for cid in range(len(cls_tokens)):
            cls_tokens[cid] = self.window_resume(cls_tokens[cid], b)
            if cid>0:
                q = torch.matmul(torch.softmax(self.ada(cls_tokens[cid]),dim=1).transpose(1,2) ,cls_tokens[cid])
                cls_tokens[cid] = torch.cat((cls_tokens[cid],q), dim=1)

        return self.out_proj(self.ln_out(torch.cat(cls_tokens, dim=1)))

        # out_q = F.normalize(torch.cat(cls_tokens, dim=1),dim=2)
        # return self.out_proj(self.ln_out(out_q))

    def window_partition(self, cls_token, x, window_size, n_tokens_per_window):

        if int(math.sqrt(x.shape[1]-1)) == int(math.sqrt(x.shape[1])):
            x_cls, x_feat = x[:, 0:1], x[:, 1:]
        else:
            x_cls = None
            x_feat = x
        b, hw, c = x_feat.shape
        c_cls = cls_token.shape[-1]
        h = w = int(math.sqrt(hw))

        x_feat = x_feat.view(b, h, w, c)
        windows = x_feat.reshape(b, h//window_size, window_size, w//window_size, window_size, c)
        windows = windows.permute(0,1,3,2,4,5).contiguous().reshape(-1, window_size*window_size, c)

        cls_token = cls_token.view(b, -1, n_tokens_per_window, c_cls).reshape(-1, n_tokens_per_window, c_cls)

        return cls_token, windows, x_cls

    def window_resume(self, cls_token, b):
        n_tokens_per_window, c_cls = cls_token.size()[-2:]

        cls_token = cls_token.view(b, -1, n_tokens_per_window, c_cls).reshape(b, -1, c_cls)
        return cls_token


if __name__ == "__main__":
    module = CosResampler("v1")
    x = {
        "block_0": torch.randn(4,257,1024),
        "block_15": torch.randn(4,257,1024),
        "block_22": torch.randn(4,257,1024)
    }
    print(module(x).shape)
    print("ok")
