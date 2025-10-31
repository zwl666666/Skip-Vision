from collections import OrderedDict
from itertools import repeat
import collections.abc
import math
import logging

import torch
import torch.nn.functional as F
from torch import nn

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from typing import Optional

from einops import rearrange

from transformers.deepspeed import is_deepspeed_zero3_enabled

logger = logging.getLogger(__name__)

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    model.apply(_convert_weights_to_fp16)

def convert_weights_to_bf16(model: nn.Module):
    """Convert applicable model parameters to bf16"""

    def _convert_weights_to_bf16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(torch.bfloat16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.bfloat16)

    model.apply(_convert_weights_to_bf16)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_grad_checkpointing=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_grad_checkpointing=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, use_grad_checkpointing and i>12) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

    def forward_with_mid_feats(self, x: torch.Tensor):
        out = {}
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)
            out[f'block_{i}'] = x.permute(1, 0, 2)
        return out

    def forward_layer_i(self, x:torch.Tensor, i:int):
        return self.resblocks[i](x.permute(1,0,2)).permute(1,0,2)



class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, use_grad_checkpointing: bool, process_resolution: Optional[int] = None,):
        super().__init__()
        self.embed_dim = width
        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        self.num_patches = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_grad_checkpointing=use_grad_checkpointing)


        if process_resolution is None:
            self.process_resolution = input_resolution
        else:
            self.process_resolution = process_resolution

#         self.ln_final = LayerNorm(width)

    def forward(self, x: torch.Tensor, without_mid_feat=False):
        if without_mid_feat:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            return x
        else:
            return self.forward_with_mid_feats(x)

    def forward_stem(self, x:torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        return x

    def forward_layer_i(self, x:torch.Tensor, i:int):
        return self.transformer.forward_layer_i(x, i)

    def forward_with_mid_feats(self, x: torch.Tensor):
        out = {}

        b, c, h, w = x.shape

        if self.input_resolution == self.process_resolution:
            if h != self.process_resolution or w != self.process_resolution:
                n_h = h // self.process_resolution
                n_w = w // self.process_resolution
                x = x.reshape(b, c, n_h, self.process_resolution, n_w, self.process_resolution)
                x = x.permute(0,2,4,1,3,5).contiguous().reshape(-1, c, self.process_resolution, self.process_resolution)
            else:
                n_h = 1
                n_w = 1


            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding
            x = self.ln_pre(x)
        else:
            n_h = h // self.process_resolution
            n_w = w // self.process_resolution
            x = self.conv1(x)
            grid_h, grid_w = x.shape[2:]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x + self.positional_embedding[1:]
            x = x.reshape(x.shape[0], x.shape[1], n_h, grid_h//n_h, n_w, grid_w//n_w)
            x = x.permute(0,2,4,3,5,1).reshape(-1,grid_h//n_h*grid_w//n_w,x.shape[1])
            x = torch.cat(
                [
                    ((self.class_embedding + self.positional_embedding[0]) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)),
                    x
                ],
                dim=1
            )
            x = self.ln_pre(x)


        out["patch_embed"] = x.detach()

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.forward_with_mid_feats(x)
        for k, v in x.items():
            if n_h > 1 or n_w > 1:
                # remove cls_token
                v = x[k][:,1:,:]
                grid = int(math.sqrt(v.shape[1]))
                out[k] = rearrange(x[k][:,1:,:], "(b n m) (h w) c -> b (n h m w) c", n=n_h, m=n_w, h=grid, w=grid)
            else:
                out[k] = v

        return out

    def get_num_layer(self, var_name=""):
        if var_name in ("class_embedding", "positional_embedding"):
            print(f"{var_name}: 0")
            return 0
        elif any([var_name.startswith("conv1"), var_name.startswith("ln_pre")]):
            print(f"{var_name}: 0")
            return 0
        elif var_name.startswith("transformer.resblocks"):
            layer_id = int(var_name.split('.')[2])
            print(f"{var_name}: {layer_id + 1}")
            return layer_id + 1
        else:
            print(f"{var_name}: {len(self.transformer.resblocks)}")
            return len(self.transformer.resblocks)

    def forward_features(self, x, **kwargs):
        return self.forward(x)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def interpolate_pos_embed(model, state_dict, interpolation: str = 'bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('positional_embedding', None)

    grid_size = round((model.positional_embedding.shape[0] - 1) ** 0.5)
    if old_pos_embed is None:
        return
    grid_size = to_2tuple(grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed

    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    print('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed

def _load_state_dict_into_model(model_to_load, state_dict, start_prefix, assign_to_params_buffers=False):
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            logger.warning(PARAM_RENAME_WARNING.format("gamma", "weight"))
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            logger.warning(PARAM_RENAME_WARNING.format("beta", "bias"))
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix="", assign_to_params_buffers=False):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".", assign_to_params_buffers)

    load(model_to_load, state_dict, prefix=start_prefix, assign_to_params_buffers=assign_to_params_buffers)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


def create_clip_vit_L(img_size=224,use_checkpoint=False,precision="fp16", full=False, freeze=True, process_resolution=None):
    model = VisionTransformer(
            input_resolution=img_size,
            patch_size=14,
            width=1024,
            layers=24 if full else 23,
            heads=16,
            use_grad_checkpointing=use_checkpoint,
            process_resolution=process_resolution
        )
    if img_size == 336:
        cached_file = "/gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/clip/clip_vit_L_336_full.pth"
    elif img_size in [224, 364, 392, 448, 672, 896]:
        if full:
            cached_file = "/gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/clip/clip_vit_L_full.pth"
        else:
            cached_file = "/gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/clip/clip_vit_L.pth"

    state_dict = torch.load(cached_file, map_location="cpu")
    interpolate_pos_embed(model,state_dict)
    model.hidden_size = 1024
    if torch.distributed.torch.distributed.is_initialized():
        print(f"[RANK {torch.distributed.get_rank()}] Loading state dict to CLIP VIT.")
    msg = _load_state_dict_into_model(model, state_dict, "")
    print(msg)
    # print(f"Loading visual checkpoint from {cached_file}")
    # if is_deepspeed_zero3_enabled():
    #     import deepspeed

    #     state_dict = torch.load(cached_file, map_location="cpu")
    #     interpolate_pos_embed(model,state_dict)

    #     named_parameters = dict(model.named_parameters(prefix="", recurse=False))
    #     params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
    #     # params_to_gather = []
    #     # for name, param in model.named_parameters():
    #     #     params_to_gather.append(param)
    #     with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
    #         print(f"[RANK {torch.distributed.get_rank()}] Gathering parameter")
    #         if torch.distributed.get_rank() == 0:

    #             print("Initializing CLIP_ViT in rank 0")
    #             mismatch = model.load_state_dict(state_dict, strict=False)
    #             print("Keys in model not matched: {}".format(mismatch[0]))
    #             print("Keys in checkpoint not matched: {}".format(mismatch[1]))
    #         else:
    #             print(f"[RANK {torch.distributed.get_rank()}] Skipping CLIP ViT initialization")
    # else:
    #     state_dict = torch.load(cached_file, map_location="cpu")
    #     interpolate_pos_embed(model,state_dict)
    #     mismatch = model.load_state_dict(state_dict, strict=False)

    #     print("Keys in model not matched: {}".format(mismatch[0]))
    #     print("Keys in checkpoint not matched: {}".format(mismatch[1]))

    if precision == "fp16" and freeze:
        convert_weights_to_fp16(model)
    elif precision == "bf16" and freeze:
        convert_weights_to_bf16(model)
    return model

def extract_visual_state_dict(state_dict):
    out_state_dict = {}
    for k, v in state_dict.items():
        if 'visual' in k:
            out_state_dict[k.replace("visual.", "")] = v
    return out_state_dict

def create_clip_vit_b(cfg, img_size, precision="fp32"):
    model = VisionTransformer(
            input_resolution=img_size,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            use_grad_checkpointing=False,
        )
    sd_dir = "/gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/clip/ViT-B-16.pth"
    state_dict = extract_visual_state_dict(torch.load(sd_dir, map_location="cpu"))
    interpolate_pos_embed(model,state_dict)

    print(f"Loading state dict from {sd_dir}")

    mismatch = model.load_state_dict(state_dict, strict=False)

    print("Keys in model not matched: {}".format(mismatch[0]))
    print("Keys in checkpoint not matched: {}".format(mismatch[1]))

    if precision == "fp16":
        convert_weights_to_fp16(model)
    return model

if __name__ == "__main__":
    clip_model = create_clip_vit_L()
    print(clip_model)
