import dataclasses
from dataclasses import dataclass
import torch.nn.functional as F
from typing import Any, List, Optional, Union
import torch
import numpy as np
from collections.abc import Sequence
import torchvision.transforms as T

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("__name__")


@dataclass
class Observation:
    """模型输入"""

    # [batch_size, height, width, channel]
    images: dict[str, Any]
    # [batch_size]
    image_masks: dict[str, Optional[torch.Tensor]]
    # [batch_size, state_dim]
    state: List[Optional[torch.Tensor]]
    # [batch_size, seq_len]
    tokenized_prompt: Optional[torch.Tensor]
    # [batch_size, seq_len]
    tokenized_prompt_mask: Optional[torch.Tensor]

    # for FAST AR model,选择性的
    # [batch_size, seq_len]
    token_ar_mask: Optional[torch.Tensor]
    # [batch_size, seq_len]
    token_loss_mask: Optional[torch.Tensor]

    @classmethod
    def get_from_dict(cls, data):
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must be provided together."
            )
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = (
                    data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                )

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self):
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Action

Actions: Optional[torch.Tensor] = None

# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_observation(
    random_seed: int,
    observation: Observation,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f"images dict missing keys: expected {image_keys}, got {list(observation.images)}"
        )

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        print("第一步：", image.shape)
        print(image.shape[1:3] != image_resolution)
        if image.shape[1:3] != image_resolution:
            logger.info(
                f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}"
            )
            if image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
                print(f"原始大小：{image.shape}")
                image = resize_with_pad(image, *image_resolution)
                image = image.permute(0, 2, 3, 1)  # BCHW -> BHWC
                print(f"调整大小后：{image.shape}")
            else:
                image = resize_with_pad(image, *image_resolution)
            print(image.shape)
        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5
            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                crop_factor = 0.9  # 降低裁剪因子，从0.95改为0.9
                crop_size = (int(height * crop_factor), int(width * crop_factor))

                # 安全检查，确保裁剪尺寸小于图像尺寸
                if crop_size[0] >= height or crop_size[1] >= width:
                    crop_size = (max(1, int(height * 0.8)), max(1, int(width * 0.8)))

                transforms.extend(
                    [
                        T.RandomCrop(crop_size),
                        T.Resize((height, width)),
                        T.RandomRotation(degrees=(-5, 5)),
                    ]
                )
            transforms.append(
                T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            )
            transform_chain = T.Compose(transforms)
            transformed_images = []
            for i in range(image.shape[0]):
                # 设置随机种子以确保可重复性(如果需要)
                # torch.manual_seed(seeds[i])
                print(image[i].shape)
                img = image[i].permute(2, 0, 1)
                transformed = transform_chain(img)
                transformed = transformed.permute(1, 2, 0)
                transformed_images.append(transformed)

            image = torch.stack(transformed_images)
            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 默认不使用掩码
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool)
        else:
            out_masks[key] = torch.tensor(
                observation.image_masks[key], dtype=torch.bool
            )

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def resize_with_pad(
    images: Union[torch.Tensor, torch.LongTensor],
    height: int,
    width: int,
    method: str = "bilinear",
) -> torch.Tensor:
    """将图像调整为目标高度和宽度，通过黑色填充来避免失真。
    如果图像是float32类型，它必须在[-1, 1]范围内。
    """
    has_batch_dim = images.dim() == 4
    if not has_batch_dim:
        images = images.unsqueeze(0)  # 添加批次维度

    cur_height, cur_width = images.shape[2:4]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    print(
        f"当前大小：{cur_height, cur_width}，调整大小：{resized_height, resized_width}"
    )
    # 重新调整大小保持宽高比
    resized_images = F.interpolate(
        images.float(),
        size=(resized_height, resized_width),
        mode=method,
        align_corners=False if method in ["bilinear", "bicubic"] else None,
    )

    # 根据原始数据类型处理
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"不支持的图像数据类型: {images.dtype}")

    # 计算填充
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # 填充值取决于数据类型
    pad_value = 0 if images.dtype == torch.uint8 else -1.0

    # 使用F.pad填充
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # PyTorch填充顺序: (左,右,上,下)
        value=pad_value,
    )

    if not has_batch_dim:
        padded_images = padded_images.squeeze(0)

    return padded_images
