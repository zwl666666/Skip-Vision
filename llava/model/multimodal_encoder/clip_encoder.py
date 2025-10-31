import torch
import torch.nn as nn

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import CLIPImageProcessor, CLIPVisionConfig
from .modeling_clip import CLIPVisionModel

from llava.model.multimodal_projector.cos_configs import COS_CONFIGS
from .visual_processors import _processors

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if hasattr(args, "use_zip") and args.use_zip:
            self.use_zip = True
        else:
            self.use_zip = False
        if hasattr(args, "use_hd") and args.use_hd:
            self.use_hd = True
        else:
            self.use_hd = False

        if args.mm_projector_type == "chain-of-sight":
            cos_configs = COS_CONFIGS[args.cos_config_version]
            self.select_layer = cos_configs['cos_seq_feat_levels']

        self.pretrained_resolution = CLIPVisionConfig.from_pretrained(self.vision_tower_name).image_size
        if hasattr(args, "backbone_resolution"):
            self.backbone_resolution = args.backbone_resolution
        else:
            self.backbone_resolution = 336

        if self.use_hd:
            self.backbone_resolution = 336

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        if self.use_hd:
            image_processor = _processors['hd_resize_processor'].from_config()
            self.image_processor = image_processor
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        if self.backbone_resolution > 0:
            print(f"Initializing CLIP backbone with image size: {self.backbone_resolution}")
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                image_size=self.backbone_resolution,
                device_map=device_map
            )
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def setup_for_training(self):
        self.vision_tower.requires_grad_(True)

    def feature_select(self, image_forward_outs, dtype):
        if type(self.select_layer) is list:
            image_features = {}
            for layer_id in self.select_layer:
                feat = image_forward_outs.hidden_states[layer_id]
                if self.select_feature == 'patch':
                    image_features[f"block_{layer_id}"] = feat[:, 1:].to(dtype)
                elif self.select_feature == 'cls_patch':
                    image_features[f"block_{layer_id}"] = feat.to(dtype)
                else:
                    raise ValueError(f'Unexpected select feature: {self.select_feature}')
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.use_zip:
                attn_weights  = image_forward_outs.attentions[self.select_layer]
                hidden_states = image_forward_outs.hidden_states[self.select_layer]
                dominant_num =  100 #128  #256
                contextual_num = 576-100 #576-128  #576-128

                ## Dominant Visual Tokens
                cls_idx = 0
                cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]
                cls_attention_sum = cls_attention.sum(dim=1)
                topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
                all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=cls_attention.device), topk_indices], dim=1)

                mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=cls_attention.device).scatter_(1, all_indices, False)
                dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
                contextual_tokens = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])
                image_features = torch.cat([dominant_tokens, contextual_tokens], dim=1)
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:].to(dtype)
            elif self.select_feature == 'cls_patch':
                image_features = image_features.to(dtype)
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_outs = self.vision_tower(image.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out, image.dtype)
                image_features.append(image_feature)
        else:
            if self.use_zip:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs, images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
