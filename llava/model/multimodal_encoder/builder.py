import os
from .clip_encoder import CLIPVisionTower

from .clip_vit import create_clip_vit_L
from .visual_processors import _processors

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

        if hasattr(vision_tower, "vision_tower"):
            vision_tower.patch_size = vision_tower.vision_tower.vision_model.embeddings.patch_size

        return vision_tower
    elif vision_tower.startswith("own_implementation_clip"):
        vision_tower = create_clip_vit_L(
            img_size=vision_tower_cfg.backbone_resolution if vision_tower_cfg.backbone_resolution > 0 else 224,
            precision="fp16",
            full=False,
        )
        visual_processor_name = getattr(vision_tower_cfg, "processor_name", "none")
        if visual_processor_name == "none":
            print(f"Visual processor name not specified. Default to only_resize_processor.")
            visual_processor_name = "only_resize_processor"
        print(f"Building visual processor at the resolution of {vision_tower_cfg.backbone_resolution if vision_tower_cfg.backbone_resolution > 0 else 224}")
        image_processor = _processors[visual_processor_name].from_config(
            image_size=vision_tower_cfg.backbone_resolution if vision_tower_cfg.backbone_resolution > 0 else 224,
        )
        vision_tower.image_processor = image_processor

        return vision_tower

    raise ValueError(f'Unknown vision tower: {vision_tower}')
