import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
@registry.register_processor("blip2_dynamic_image_internvl")
class Blip2DynamicImageInternVLProcessor(CLIPImageProcessor):
    def __init__(
        self,
        patch_size=336,
        hd_num=4,
        use_visual_prompt=False,
        max_ratio=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hd_num = hd_num
        self.use_visual_prompt = use_visual_prompt
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                transforms.Resize(
                    (patch_size, patch_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        print(
            "init image process <blip2_dynamic_image_internvl>, patch_size:{}, hd_num:{}, use_visual_prompt:{}".format(
                patch_size, hd_num, use_visual_prompt
            )
        )

    def __call__(
        self,
        image,
        masked_regions=None,
        image_path=None,
        eliminate_rnd_padding=False,
        hd_num=None,
        patch_size=None,
        oss_reader=None,
    ):
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
        """
        if patch_size is None:
            patch_size = self.patch_size
        if hd_num is None:
            hd_num = self.hd_num
        if image is None:
            return None

        if isinstance(image, str):
            if image.startswith("collectcore/"):
                image = oss_reader(image)
            else:
                image = Image.open(image).convert("RGB")
        if self.use_visual_prompt and masked_regions is not None:
            xs, ys = collect_coords(masked_regions)
            if len(xs) > 0:
                img_arr = np.array(image)
                img_arr[ys, xs, :] = 0
                image = Image.fromarray(img_arr)
        images = self.dynamic_preprocess(
            image, image_size=patch_size, use_thumbnail=True, max_num=hd_num
        )
        pixel_values = [self.transform(image) for image in images]
        # pixel_values = torch.stack(pixel_values)
        # return pixel_values
        resize_ratio = None
        padding = []
        shape = []
        return pixel_values, resize_ratio, padding, hd_num, shape

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        zz = random.random()
        if zz < 0.01:
            print(
                "dynamic_preprocess:({},{}) resize to ({},{}), aspect_ratio:{}, target_aspect_ratio:{}".format(
                    orig_width,
                    orig_height,
                    target_width,
                    target_height,
                    aspect_ratio,
                    target_aspect_ratio,
                )
            )
        resized_img = image.resize((target_width, target_height))
        # image.save("/gruntdata/datacube_nas/workspace/lxy222919/tmp_file/show/interV/debug.png")
        processed_images = []
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == (blocks+1)
        return processed_images

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        patch_size = cfg.get("patch_size", 224)
        hd_num = cfg.get("hd_num", 4)
        max_ratio = cfg.get("max_ratio", False)
        use_visual_prompt = cfg.get("use_visual_prompt", False)

        return cls(
            patch_size=patch_size,
            hd_num=hd_num,
            use_visual_prompt=use_visual_prompt,
            max_ratio=max_ratio,
        )
