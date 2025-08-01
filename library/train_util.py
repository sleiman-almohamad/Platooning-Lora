# common functions for training

import argparse
import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import datetime
import importlib
import json
import logging
import pathlib
import re
import shutil
import time
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState
import glob
import math
import os
import random
import hashlib
import subprocess
from io import BytesIO
import toml

# from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from packaging.version import Version

import torch
from library.device_utils import init_ipex, clean_memory_on_device
from library.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, TextEncodingStrategy

init_ipex()

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL,
)
from library import custom_train_functions, sd3_utils
from library.original_unet import UNet2DConditionModel
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import imagesize
import cv2
import safetensors.torch
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
import library.model_util as model_util
import library.huggingface_util as huggingface_util
import library.sai_model_spec as sai_model_spec
import library.deepspeed_utils as deepspeed_utils
from library.utils import setup_logging, resize_image, validate_interpolation_fn

setup_logging()
import logging

logger = logging.getLogger(__name__)
# from library.attention_processors import FlashAttnProcessor
# from library.hypernetwork import replace_attentions_for_hypernetwork
from library.original_unet import UNet2DConditionModel

HIGH_VRAM = False

# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

# region dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# JPEG-XL on Linux and Windows
try:
    import pillow_jxl
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"
TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3 = "_sd3_te.npz"


def split_train_val(
    paths: List[str],
    sizes: List[Optional[Tuple[int, int]]],
    is_training_dataset: bool,
    validation_split: float,
    validation_seed: int | None,
) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
    """
    Split the dataset into train and validation

    Shuffle the dataset based on the validation_seed or the current random seed.
    For example if the split of 0.2 of 100 images.
    [0:80] = 80 training images
    [80:] = 20 validation images
    """
    dataset = list(zip(paths, sizes))
    if validation_seed is not None:
        logging.info(f"Using validation seed: {validation_seed}")
        prevstate = random.getstate()
        random.seed(validation_seed)
        random.shuffle(dataset)
        random.setstate(prevstate)
    else:
        random.shuffle(dataset)

    paths, sizes = zip(*dataset)
    paths = list(paths)
    sizes = list(sizes)
    # Split the dataset between training and validation
    if is_training_dataset:
        # Training dataset we split to the first part
        split = math.ceil(len(paths) * (1 - validation_split))
        return paths[0:split], sizes[0:split]
    else:
        # Validation dataset we split to the second part
        split = len(paths) - round(len(paths) * validation_split)
        return paths[split:], sizes[split:]


class ImageInfo:
    def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: Optional[torch.Tensor] = None
        self.latents_flipped: Optional[torch.Tensor] = None
        self.latents_npz: Optional[str] = None  # set in cache_latents
        self.latents_original_size: Optional[Tuple[int, int]] = None  # original image size, not latents size
        self.latents_crop_ltrb: Optional[Tuple[int, int]] = (
            None  # crop left top right bottom in original pixel size, not latents size
        )
        self.cond_img_path: Optional[str] = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        self.text_encoder_outputs_npz: Optional[str] = None  # set in cache_text_encoder_outputs

        # new
        self.text_encoder_outputs: Optional[List[torch.Tensor]] = None
        # old
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None

        self.alpha_mask: Optional[torch.Tensor] = None  # alpha mask can be flipped in runtime
        self.resize_interpolation: Optional[str] = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        # Convert any lists to tuples to ensure they are hashable for the set
        self.predefined_resos = [tuple(reso) if isinstance(reso, list) else reso for reso in resos]
        self.predefined_resos_set = set(self.predefined_resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in self.predefined_resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # logger.info(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # logger.info(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucketのほうが横長→縦を合わせる
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentationsへの依存をなくしたがとりあえず同じinterfaceを持たせる

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        # self.color_aug_method = albu.OneOf(
        #     [
        #         albu.HueSaturationValue(8, 0, 0, p=0.5),
        #         albu.RandomGamma((95, 105), p=0.5),
        #     ],
        #     p=0.33,
        # )
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(self, use_color_aug: bool):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        alpha_mask: Optional[bool],
        num_repeats: int,
        shuffle_caption: bool,
        caption_separator: str,
        keep_tokens: int,
        keep_tokens_separator: str,
        secondary_separator: Optional[str],
        enable_wildcard: bool,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        caption_prefix: Optional[str],
        caption_suffix: Optional[str],
        token_warmup_min: int,
        token_warmup_step: Union[float, int],
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.alpha_mask = alpha_mask if alpha_mask is not None else False
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.caption_separator = caption_separator
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.secondary_separator = secondary_separator
        self.enable_wildcard = enable_wildcard
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix

        self.token_warmup_min = token_warmup_min  # step=0におけるタグの数
        self.token_warmup_step = token_warmup_step  # N（N<1ならN*max_train_steps）ステップ目でタグの数が最大になる

        self.custom_attributes = custom_attributes if custom_attributes is not None else {}

        self.img_count = 0

        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.resize_interpolation = resize_interpolation


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        cache_info: bool,
        alpha_mask: bool,
        num_repeats,
        shuffle_caption,
        caption_separator: str,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class FineTuningSubset(BaseSubset):
    def __init__(
        self,
        image_dir,
        metadata_file: str,
        alpha_mask: bool,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert metadata_file is not None, "metadata_file must be specified / metadata_fileは指定が必須です"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.metadata_file = metadata_file

    def __eq__(self, other) -> bool:
        if not isinstance(other, FineTuningSubset):
            return NotImplemented
        return self.metadata_file == other.metadata_file


class ControlNetSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        conditioning_data_dir: str,
        caption_extension: str,
        cache_info: bool,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            False,  # alpha_mask
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.conditioning_data_dir = conditioning_data_dir
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, ControlNetSubset):
            return NotImplemented
        return self.image_dir == other.image_dir and self.conditioning_data_dir == other.conditioning_data_dir


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        debug_dataset: bool,
        resize_interpolation: Optional[str] = None
    ) -> None:
        super().__init__()

        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.current_epoch: int = 0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        if resize_interpolation is not None:
            assert validate_interpolation_fn(resize_interpolation), f"Resize interpolation \"{resize_interpolation}\" is not a valid interpolation"
        self.resize_interpolation = resize_interpolation

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

        self.tokenize_strategy = None
        self.text_encoder_output_caching_strategy = None
        self.latents_caching_strategy = None

    def set_current_strategies(self):
        self.tokenize_strategy = TokenizeStrategy.get_strategy()
        self.text_encoder_output_caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        self.latents_caching_strategy = LatentsCachingStrategy.get_strategy()

    def adjust_min_max_bucket_reso_by_steps(
        self, resolution: Tuple[int, int], min_bucket_reso: int, max_bucket_reso: int, bucket_reso_steps: int
    ) -> Tuple[int, int]:
        # make min/max bucket reso to be multiple of bucket_reso_steps
        if min_bucket_reso % bucket_reso_steps != 0:
            adjusted_min_bucket_reso = min_bucket_reso - min_bucket_reso % bucket_reso_steps
            logger.warning(
                f"min_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / min_bucket_resoがbucket_reso_stepsの倍数になるように調整されました: {min_bucket_reso} -> {adjusted_min_bucket_reso}"
            )
            min_bucket_reso = adjusted_min_bucket_reso
        if max_bucket_reso % bucket_reso_steps != 0:
            adjusted_max_bucket_reso = max_bucket_reso + bucket_reso_steps - max_bucket_reso % bucket_reso_steps
            logger.warning(
                f"max_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / max_bucket_resoがbucket_reso_stepsの倍数になるように調整されました: {max_bucket_reso} -> {adjusted_max_bucket_reso}"
            )
            max_bucket_reso = adjusted_max_bucket_reso

        assert (
            min(resolution) >= min_bucket_reso
        ), f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
        assert (
            max(resolution) <= max_bucket_reso
        ), f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"

        return min_bucket_reso, max_bucket_reso

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # epochが切り替わったらバケツをシャッフルする
            if epoch > self.current_epoch:
                logger.info("epoch is incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                num_epochs = epoch - self.current_epoch
                for _ in range(num_epochs):
                    self.current_epoch += 1
                    self.shuffle_buckets()
                # self.current_epoch seem to be set to 0 again in the next epoch. it may be caused by skipped_dataloader?
            else:
                logger.warning("epoch is not incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        # caption に prefix/suffix を付ける
        if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix

        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            # process wildcards
            if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'
                # escape the curly braces like {{ or }}
                replacer1 = "⦅"
                replacer2 = "⦆"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "⦅"
                    replacer2 += "⦆"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                # replace the wildcard
                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                # unescape the curly braces
                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                # if caption is multiline, use the first line
                caption = caption.split("\n")[0]

            if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
                fixed_tokens = []
                flex_tokens = []
                fixed_suffix_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(subset.keep_tokens_separator, 1)
                    if subset.keep_tokens_separator in flex_part:
                        flex_part, fixed_suffix_part = flex_part.split(subset.keep_tokens_separator, 1)
                        fixed_suffix_tokens = [t.strip() for t in fixed_suffix_part.split(subset.caption_separator) if t.strip()]

                    fixed_tokens = [t.strip() for t in fixed_part.split(subset.caption_separator) if t.strip()]
                    flex_tokens = [t.strip() for t in flex_part.split(subset.caption_separator) if t.strip()]
                else:
                    tokens = [t.strip() for t in caption.strip().split(subset.caption_separator)]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:  # 初回に上書きする
                    subset.token_warmup_step = math.floor(subset.token_warmup_step * self.max_train_steps)
                if subset.token_warmup_step and self.current_step < subset.token_warmup_step:
                    tokens_len = (
                        math.floor(
                            (self.current_step) * ((len(flex_tokens) - subset.token_warmup_min) / (subset.token_warmup_step))
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    l = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            l.append(token)
                    return l

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            # process secondary separator
            if subset.secondary_separator:
                caption = caption.replace(subset.secondary_separator, subset.caption_separator)

            # textual inversion対応
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if type(str_to) == list:
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        input_ids = tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt"
        ).input_ids

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        # # run in parallel
        # max_workers = min(os.cpu_count(), len(self.image_data))  # TODO consider multi-gpu (processes)
        # with ThreadPoolExecutor(max_workers) as executor:
        #     futures = []
        #     for info in tqdm(self.image_data.values(), desc="loading image sizes"):
        #         if info.image_size is None:
        #             def get_and_set_image_size(info):
        #                 info.image_size = self.get_image_size(info.absolute_path)
        #             futures.append(executor.submit(get_and_set_image_size, info))
        #             # consume futures to reduce memory usage and prevent Ctrl-C hang
        #             if len(futures) >= max_workers:
        #                 for future in futures:
        #                     future.result()
        #                 futures = []
        #     for future in futures:
        #         future.result()

        if self.enable_bucket:
            logger.info("make buckets")
        else:
            logger.info("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
        if self.enable_bucket:
            if self.bucket_manager is None:  # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets()
                else:
                    logger.warning(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                    image_width, image_height
                )

                # logger.info(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            logger.info("number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）")
            for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
                    logger.info(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            img_ar_errors = np.array(img_ar_errors)
            mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            logger.info(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def verify_bucket_reso_steps(self, min_steps: int):
        assert self.bucket_reso_steps is None or self.bucket_reso_steps % min_steps == 0, (
            f"bucket_reso_steps is {self.bucket_reso_steps}. it must be divisible by {min_steps}.\n"
            + f"bucket_reso_stepsが{self.bucket_reso_steps}です。{min_steps}で割り切れる必要があります"
        )

    def is_latent_cacheable(self):
        return all([not subset.color_aug and not subset.random_crop for subset in self.subsets])

    def is_text_encoder_output_cacheable(self):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        r"""
        a brand new method to cache latents. This method caches latents with caching strategy.
        normal cache_latents method is used by default, but this method is used when caching strategy is specified.
        """
        logger.info("caching latents with caching strategy.")
        caching_strategy = LatentsCachingStrategy.get_strategy()
        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batch: List[ImageInfo] = []
        current_condition = None

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        # define a function to submit a batch to cache
        def submit_batch(batch, cond):
            for info in batch:
                if info.image is not None and isinstance(info.image, Future):
                    info.image = info.image.result()  # future to image
            caching_strategy.cache_batch_latents(model, batch, cond.flip_aug, cond.alpha_mask, cond.random_crop)

            # remove image from memory
            for info in batch:
                info.image = None

        # define ThreadPoolExecutor to load images in parallel
        max_workers = min(os.cpu_count(), len(image_infos))
        max_workers = max(1, max_workers // num_processes)  # consider multi-gpu
        max_workers = min(max_workers, caching_strategy.batch_size)  # max_workers should be less than batch_size
        executor = ThreadPoolExecutor(max_workers)

        try:
            # iterate images
            logger.info("caching latents...")
            for i, info in enumerate(tqdm(image_infos)):
                subset = self.image_to_subset[info.image_key]

                if info.latents_npz is not None:  # fine tuning dataset
                    continue

                # check disk cache exists and size of latents
                if caching_strategy.cache_to_disk:
                    # info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                    info.latents_npz = caching_strategy.get_latents_npz_path(info.absolute_path, info.image_size)

                    # if the modulo of num_processes is not equal to process_index, skip caching
                    # this makes each process cache different latents
                    if i % num_processes != process_index:
                        continue

                    # print(f"{process_index}/{num_processes} {i}/{len(image_infos)} {info.latents_npz}")

                    cache_available = caching_strategy.is_disk_cached_latents_expected(
                        info.bucket_reso, info.latents_npz, subset.flip_aug, subset.alpha_mask
                    )
                    if cache_available:  # do not add to batch
                        continue

                # if batch is not empty and condition is changed, flush the batch. Note that current_condition is not None if batch is not empty
                condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
                if len(batch) > 0 and current_condition != condition:
                    submit_batch(batch, current_condition)
                    batch = []

                if info.image is None:
                    # load image in parallel
                    info.image = executor.submit(load_image, info.absolute_path, condition.alpha_mask)

                batch.append(info)
                current_condition = condition

                # if number of data in batch is enough, flush the batch
                if len(batch) >= caching_strategy.batch_size:
                    submit_batch(batch, current_condition)
                    batch = []
                    current_condition = None

            if len(batch) > 0:
                submit_batch(batch, current_condition)

        finally:
            executor.shutdown()

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        # マルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
        logger.info("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batches: List[Tuple[Condition, List[ImageInfo]]] = []
        batch: List[ImageInfo] = []
        current_condition = None

        logger.info("checking cache validity...")
        for info in tqdm(image_infos):
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:  # fine tuning dataset
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                if not is_main_process:  # store to info only
                    continue

                cache_available = is_disk_cached_latents_is_expected(
                    info.bucket_reso, info.latents_npz, subset.flip_aug, subset.alpha_mask
                )

                if cache_available:  # do not add to batch
                    continue

            # if batch is not empty and condition is changed, flush the batch. Note that current_condition is not None if batch is not empty
            condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
            if len(batch) > 0 and current_condition != condition:
                batches.append((current_condition, batch))
                batch = []

            batch.append(info)
            current_condition = condition

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append((current_condition, batch))
                batch = []
                current_condition = None

        if len(batch) > 0:
            batches.append((current_condition, batch))

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
        logger.info("caching latents...")
        for condition, batch in tqdm(batches, smoothing=1, total=len(batches)):
            cache_batch_latents(vae, cache_to_disk, batch, condition.flip_aug, condition.alpha_mask, condition.random_crop)

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        r"""
        a brand new method to cache text encoder outputs. This method caches text encoder outputs with caching strategy.
        """
        tokenize_strategy = TokenizeStrategy.get_strategy()
        text_encoding_strategy = TextEncodingStrategy.get_strategy()
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        batch_size = caching_strategy.batch_size or self.batch_size

        logger.info("caching Text Encoder outputs with caching strategy.")
        image_infos = list(self.image_data.values())

        # split by resolution
        batches = []
        batch = []

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        logger.info("checking cache validity...")
        for i, info in enumerate(tqdm(image_infos)):
            # check disk cache exists and size of text encoder outputs
            if caching_strategy.cache_to_disk:
                te_out_npz = caching_strategy.get_outputs_npz_path(info.absolute_path)
                info.text_encoder_outputs_npz = te_out_npz  # set npz filename regardless of cache availability

                # if the modulo of num_processes is not equal to process_index, skip caching
                # this makes each process cache different text encoder outputs
                if i % num_processes != process_index:
                    continue

                cache_available = caching_strategy.is_disk_cached_outputs_expected(te_out_npz)
                if cache_available:  # do not add to batch
                    continue

            batch.append(info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if len(batches) == 0:
            logger.info("no Text Encoder outputs to cache")
            return

        # iterate batches
        logger.info("caching Text Encoder outputs...")
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            # cache_batch_latents(vae, cache_to_disk, batch, subset.flip_aug, subset.alpha_mask, subset.random_crop)
            caching_strategy.cache_batch_outputs(tokenize_strategy, models, text_encoding_strategy, batch)

    # if weight_dtype is specified, Text Encoder itself and output will be converted to the dtype
    # this method is only for SDXL, but it should be implemented here because it needs to be a method of dataset
    # to support SD1/2, it needs a flag for v2, but it is postponed
    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, output_dtype, cache_to_disk=False, is_main_process=True
    ):
        assert len(tokenizers) == 2, "only support SDXL"
        return self.cache_text_encoder_outputs_common(
            tokenizers, text_encoders, [device, device], output_dtype, [output_dtype], cache_to_disk, is_main_process
        )

    # same as above, but for SD3
    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, devices, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        return self.cache_text_encoder_outputs_common(
            [tokenizer],
            text_encoders,
            devices,
            output_dtype,
            te_dtypes,
            cache_to_disk,
            is_main_process,
            TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
            batch_size,
        )

    def cache_text_encoder_outputs_common(
        self,
        tokenizers,
        text_encoders,
        devices,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
        batch_size=None,
    ):
        # latentsのキャッシュと同様に、ディスクへのキャッシュに対応する
        # またマルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
        logger.info("caching text encoder outputs.")

        tokenize_strategy = TokenizeStrategy.get_strategy()

        if batch_size is None:
            batch_size = self.batch_size

        image_infos = list(self.image_data.values())

        logger.info("checking cache existence...")
        image_infos_to_cache = []
        for info in tqdm(image_infos):
            # subset = self.image_to_subset[info.image_key]
            if cache_to_disk:
                te_out_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                info.text_encoder_outputs_npz = te_out_npz

                if not is_main_process:  # store to info only
                    continue

                if os.path.exists(te_out_npz):
                    # TODO check varidity of cache here
                    continue

            image_infos_to_cache.append(info)

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # prepare tokenizers and text encoders
        for text_encoder, device, te_dtype in zip(text_encoders, devices, te_dtypes):
            text_encoder.to(device)
            if te_dtype is not None:
                text_encoder.to(dtype=te_dtype)

        # create batch
        is_sd3 = len(tokenizers) == 1
        batch = []
        batches = []
        for info in image_infos_to_cache:
            if not is_sd3:
                input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
                input_ids2 = self.get_input_ids(info.caption, tokenizers[1])
                batch.append((info, input_ids1, input_ids2))
            else:
                l_tokens, g_tokens, t5_tokens = tokenize_strategy.tokenize(info.caption)
                batch.append((info, l_tokens, g_tokens, t5_tokens))

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # iterate batches: call text encoder and cache outputs for memory or disk
        logger.info("caching text encoder outputs...")
        if not is_sd3:
            for batch in tqdm(batches):
                infos, input_ids1, input_ids2 = zip(*batch)
                input_ids1 = torch.stack(input_ids1, dim=0)
                input_ids2 = torch.stack(input_ids2, dim=0)
                cache_batch_text_encoder_outputs(
                    infos, tokenizers, text_encoders, self.max_token_length, cache_to_disk, input_ids1, input_ids2, output_dtype
                )
        else:
            for batch in tqdm(batches):
                infos, l_tokens, g_tokens, t5_tokens = zip(*batch)

                # stack tokens
                # l_tokens = [tokens[0] for tokens in l_tokens]
                # g_tokens = [tokens[0] for tokens in g_tokens]
                # t5_tokens = [tokens[0] for tokens in t5_tokens]

                cache_batch_text_encoder_outputs_sd3(
                    infos,
                    tokenizers[0],
                    text_encoders,
                    self.max_token_length,
                    cache_to_disk,
                    (l_tokens, g_tokens, t5_tokens),
                    output_dtype,
                )

    def get_image_size(self, image_path):
        if image_path.endswith(".jxl") or image_path.endswith(".JXL"):
            return get_jxl_size(image_path)
        # return imagesize.get(image_path)
        image_size = imagesize.get(image_path)
        if image_size[0] <= 0:
            # imagesize doesn't work for some images, so use PIL as a fallback
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as e:
                logger.warning(f"failed to get image size: {image_path}, error: {e}")
                image_size = (0, 0)
        return image_size

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str, alpha_mask=False):
        img = load_image(image_path, alpha_mask)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # 短いほう
        min_scale = max(self.height / height, self.width / width)  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1])))  # 指定した顔最小サイズ
        max_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0])))  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = resize_image(image, width, height, nw, nh, subset.resize_interpolation)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if subset.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                range = max(length - face_p, face_p)  # 画像の端から顔中心までの距離の長いほう
                p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        if self.caching_mode is not None:  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs_list = []
        custom_attributes = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            flipped = subset.flip_aug and random.random() < 0.5  # not flipped or flipped with 50% chance

            # image/latentsを処理する
            if image_info.latents is not None:  # cache_latents=Trueの場合
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb  # calc values later if flipped
                if not flipped:
                    latents = image_info.latents
                    alpha_mask = image_info.alpha_mask
                else:
                    latents = image_info.latents_flipped
                    alpha_mask = None if image_info.alpha_mask is None else torch.flip(image_info.alpha_mask, [1])

                image = None
            elif image_info.latents_npz is not None:  # FineTuningDatasetまたはcache_latents_to_disk=Trueの場合
                latents, original_size, crop_ltrb, flipped_latents, alpha_mask = (
                    self.latents_caching_strategy.load_latents_from_disk(image_info.latents_npz, image_info.bucket_reso)
                )
                if flipped:
                    latents = flipped_latents
                    alpha_mask = None if alpha_mask is None else alpha_mask[:, ::-1].copy()  # copy to avoid negative stride problem
                    del flipped_latents
                latents = torch.FloatTensor(latents)
                if alpha_mask is not None:
                    alpha_mask = torch.FloatTensor(alpha_mask)

                image = None
            else:
                # 画像を読み込み、必要ならcropする
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                    subset, image_info.absolute_path, subset.alpha_mask
                )
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        subset.random_crop, img, image_info.bucket_reso, image_info.resized_size, resize_interpolation=image_info.resize_interpolation
                    )
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / 画像サイズが大きいのでface_crop_aug_rangeかrandom_crop、またはbucketを有効にしてください: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    # augment RGB channels only
                    img_rgb = img[:, :, :3]
                    img_rgb = aug(image=img_rgb)["image"]
                    img[:, :, :3] = img_rgb

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                if subset.alpha_mask:
                    if img.shape[2] == 4:
                        alpha_mask = img[:, :, 3]  # [H,W]
                        alpha_mask = alpha_mask.astype(np.float32) / 255.0  # 0.0~1.0
                        alpha_mask = torch.FloatTensor(alpha_mask)
                    else:
                        alpha_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.float32)
                else:
                    alpha_mask = None

                img = img[:, :, :3]  # remove alpha channel

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる
                del img

            images.append(image)
            latents_list.append(latents)
            alpha_mask_list.append(alpha_mask)

            target_size = (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # captionとtext encoder outputを処理する
            caption = image_info.caption  # default

            tokenization_required = (
                self.text_encoder_output_caching_strategy is None or self.text_encoder_output_caching_strategy.is_partial
            )
            text_encoder_outputs = None
            input_ids = None

            if image_info.text_encoder_outputs is not None:
                # cached
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                # on disk
                text_encoder_outputs = self.text_encoder_output_caching_strategy.load_outputs_npz(
                    image_info.text_encoder_outputs_npz
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension
                # if self.XTI_layers:
                #     caption_layer = []
                #     for layer in self.XTI_layers:
                #         token_strings_from = " ".join(self.token_strings)
                #         token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                #         caption_ = caption.replace(token_strings_from, token_strings_to)
                #         caption_layer.append(caption_)
                #     captions.append(caption_layer)
                # else:
                #     captions.append(caption)

                # if not self.token_padding_disabled:  # this option might be omitted in future
                #     # TODO get_input_ids must support SD3
                #     if self.XTI_layers:
                #         token_caption = self.get_input_ids(caption_layer, self.tokenizers[0])
                #     else:
                #         token_caption = self.get_input_ids(caption, self.tokenizers[0])
                #     input_ids_list.append(token_caption)

                #     if len(self.tokenizers) > 1:
                #         if self.XTI_layers:
                #             token_caption2 = self.get_input_ids(caption_layer, self.tokenizers[1])
                #         else:
                #             token_caption2 = self.get_input_ids(caption, self.tokenizers[1])
                #         input_ids2_list.append(token_caption2)

            input_ids_list.append(input_ids)
            captions.append(caption)

        def none_or_stack_elements(tensors_list, converter):
            # [[clip_l, clip_g, t5xxl], [clip_l, clip_g, t5xxl], ...] -> [torch.stack(clip_l), torch.stack(clip_g), torch.stack(t5xxl)]
            if len(tensors_list) == 0 or tensors_list[0] == None or len(tensors_list[0]) == 0 or tensors_list[0][0] is None:
                return None
            return [torch.stack([converter(x[i]) for x in tensors_list]) for i in range(len(tensors_list[0]))]

        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

        # if one of alpha_masks is not None, we need to replace None with ones
        none_or_not = [x is None for x in alpha_mask_list]
        if all(none_or_not):
            example["alpha_masks"] = None
        elif any(none_or_not):
            for i in range(len(alpha_mask_list)):
                if alpha_mask_list[i] is None:
                    if images[i] is not None:
                        alpha_mask_list[i] = torch.ones((images[i].shape[1], images[i].shape[2]), dtype=torch.float32)
                    else:
                        alpha_mask_list[i] = torch.ones(
                            (latents_list[i].shape[1] * 8, latents_list[i].shape[2] * 8), dtype=torch.float32
                        )
            example["alpha_masks"] = torch.stack(alpha_mask_list)
        else:
            example["alpha_masks"] = torch.stack(alpha_mask_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = torch.stack(latents_list) if latents_list[0] is not None else None
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        alpha_mask = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                alpha_mask = subset.alpha_mask
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                # TODO そもそも混在してても動くようにしたほうがいい
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert alpha_mask == subset.alpha_mask, "alpha_mask must be same in a batch"
                assert random_crop == subset.random_crop, "random_crop must be same in a batch"
                assert bucket_reso == image_info.bucket_reso, "bucket_reso must be same in a batch"

            caption = image_info.caption  # TODO cache some patterns of dropping, shuffling, etc.

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["alpha_mask"] = alpha_mask
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    # The is_training_dataset defines the type of dataset, training or validation
    # if is_training_dataset is True -> training dataset
    # if is_training_dataset is False -> validation dataset
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        assert resolution is not None, f"resolution is required / resolution（解像度）指定は必須です"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension, enable_wildcard):
            # captionの候補ファイル名を作る
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            logger.error(f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path}")
                            raise e
                        assert len(lines) > 0, f"caption file is empty / キャプションファイルが空です: {cap_path}"
                        if enable_wildcard:
                            caption = "\n".join([line.strip() for line in lines if line.strip() != ""])  # 空行を除く、改行で連結
                        else:
                            caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info(
                    f"using cached image info for this subset / このサブセットで、キャッシュされた画像情報を使います: {info_cache_file}"
                )
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        f"image info file not found. You can ignore this warning if this is the first time to use this subset"
                        + " / キャッシュファイルが見つかりませんでした。初回実行時はこの警告を無視してください: {metadata_file}"
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                # json: {`img_path`:{"caption": "caption...", "resolution": [width, height]}, ...}
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [meta["resolution"] for meta in metas.values()]

                # we may need to check image size and existence of image files, but it takes time, so user should check it before training
            else:
                img_paths = glob_images(subset.image_dir, "*")
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                # new caching: get image size from cache files
                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    logger.info("get image size from name of cache files")

                    # make image path to npz path mapping
                    npz_paths = glob.glob(os.path.join(subset.image_dir, "*" + strategy.cache_suffix))
                    npz_paths.sort(
                        key=lambda item: item.rsplit("_", maxsplit=2)[0]
                    )  # sort by name excluding resolution and cache_suffix
                    npz_path_index = 0

                    size_set_count = 0
                    for i, img_path in enumerate(tqdm(img_paths)):
                        l = len(os.path.splitext(img_path)[0])  # remove extension
                        found = False
                        while npz_path_index < len(npz_paths):  # until found or end of npz_paths
                            # npz_paths are sorted, so if npz_path > img_path, img_path is not found
                            if npz_paths[npz_path_index][:l] > img_path[:l]:
                                break
                            if npz_paths[npz_path_index][:l] == img_path[:l]:  # found
                                found = True
                                break
                            npz_path_index += 1  # next npz_path

                        if found:
                            w, h = strategy.get_image_size_from_disk_cache_path(img_path, npz_paths[npz_path_index])
                        else:
                            w, h = None, None

                        if w is not None and h is not None:
                            sizes[i] = (w, h)
                            size_set_count += 1
                    logger.info(f"set image size from cache files: {size_set_count}/{len(img_paths)}")

            # We want to create a training and validation split. This should be improved in the future
            # to allow a clearer distinction between training and validation. This can be seen as a
            # short-term solution to limit what is necessary to implement validation datasets
            #
            # We split the dataset for the subset based on if we are doing a validation split
            # The self.is_training_dataset defines the type of dataset, training or validation
            # if self.is_training_dataset is True -> training dataset
            # if self.is_training_dataset is False -> validation dataset
            if self.validation_split > 0.0:
                # For regularization images we do not want to split this dataset.
                if subset.is_reg is True:
                    # Skip any validation dataset for regularization images
                    if self.is_training_dataset is False:
                        img_paths = []
                        sizes = []
                    # Otherwise the img_paths remain as original img_paths and no split
                    # required for training images dataset of regularization images
                else:
                    img_paths, sizes = split_train_val(
                        img_paths, sizes, self.is_training_dataset, self.validation_split, self.validation_seed
                    )

            logger.info(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [img_path for img_path, caption in zip(img_paths, captions) if caption is None or caption == ""]
            else:
                # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(img_path, subset.caption_extension, subset.enable_wildcard)
                    if cap_for_img is None and subset.class_tokens is None:
                        logger.warning(
                            f"neither caption file nor class tokens are found. use empty caption for {img_path} / キャプションファイルもclass tokenも見つかりませんでした。空のキャプションを使用します: {img_path}"
                        )
                        captions.append("")
                        missing_captions.append(img_path)
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # タグ頻度を記録

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = number_of_missing_captions - number_of_missing_captions_to_show

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used. / {number_of_missing_captions}枚の画像にキャプションファイルが見つかりませんでした。これらの画像についてはキャプションなしで学習を続行します。class tokenが存在する場合はそれを使います。"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(missing_caption + f"... and {remaining_missing_captions} more")
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info(f"cache image info for / 画像情報をキャッシュします : {info_cache_file}")
                sizes = [self.get_image_size(img_path) for img_path in tqdm(img_paths, desc="get image size")]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info(f"cache image info done for / 画像情報を出力しました : {info_cache_file}")

            # if sizes are not set, image size will be read in make_buckets
            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            num_repeats = subset.num_repeats if self.is_training_dataset else 1
            if num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: {num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found / 画像が見つからないためサブセットを無視します"
                )
                continue

            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(img_path, num_repeats, caption, subset.is_reg, img_path)
                info.resize_interpolation = subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        images_split_name = "train" if self.is_training_dataset else "validation"
        logger.info(f"{num_train_images} {images_split_name} images with repeats.")

        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images with repeats.")
        if num_train_images < num_reg_images:
            logger.warning("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

        if num_reg_images == 0:
            logger.warning("no regularization images / 正則化画像が見つかりませんでした")
        else:
            # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_seed: int,
        validation_split: float,
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        self.batch_size = batch_size

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            # メタデータを読み込む
            if os.path.exists(subset.metadata_file):
                logger.info(f"loading existing metadata: {subset.metadata_file}")
                with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                raise ValueError(f"no metadata / メタデータファイルがありません: {subset.metadata_file}")

            if len(metadata) < 1:
                logger.warning(
                    f"ignore subset with '{subset.metadata_file}': no image entries found / 画像に関するデータが見つからないためサブセットを無視します"
                )
                continue

            tags_list = []
            for image_key, img_md in metadata.items():
                # path情報を作る
                abs_path = None

                # まず画像を優先して探す
                if os.path.exists(image_key):
                    abs_path = image_key
                else:
                    # わりといい加減だがいい方法が思いつかん
                    paths = glob_images(subset.image_dir, image_key)
                    if len(paths) > 0:
                        abs_path = paths[0]

                # なければnpzを探す
                if abs_path is None:
                    if os.path.exists(os.path.splitext(image_key)[0] + ".npz"):
                        abs_path = os.path.splitext(image_key)[0] + ".npz"
                    else:
                        npz_path = os.path.join(subset.image_dir, image_key + ".npz")
                        if os.path.exists(npz_path):
                            abs_path = npz_path

                assert abs_path is not None, f"no image / 画像がありません: {image_key}"

                caption = img_md.get("caption")
                tags = img_md.get("tags")
                if caption is None:
                    caption = tags  # could be multiline
                    tags = None

                if subset.enable_wildcard:
                    # tags must be single line
                    if tags is not None:
                        tags = tags.replace("\n", subset.caption_separator)

                    # add tags to each line of caption
                    if caption is not None and tags is not None:
                        caption = "\n".join(
                            [f"{line}{subset.caption_separator}{tags}" for line in caption.split("\n") if line.strip() != ""]
                        )
                else:
                    # use as is
                    if tags is not None and len(tags) > 0:
                        caption = caption + subset.caption_separator + tags
                        tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                image_info.image_size = img_md.get("train_resolution")

                if not subset.color_aug and not subset.random_crop:
                    # if npz exists, use them
                    image_info.latents_npz, image_info.latents_npz_flipped = self.image_key_to_npz_file(subset, image_key)

                self.register_image(image_info, subset)

            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)

        # check existence of all npz files
        use_npz_latents = all([not (subset.color_aug or subset.random_crop) for subset in self.subsets])
        if use_npz_latents:
            flip_aug_in_subset = False
            npz_any = False
            npz_all = True

            for image_info in self.image_data.values():
                subset = self.image_to_subset[image_info.image_key]

                has_npz = image_info.latents_npz is not None
                npz_any = npz_any or has_npz

                if subset.flip_aug:
                    has_npz = has_npz and image_info.latents_npz_flipped is not None
                    flip_aug_in_subset = True
                npz_all = npz_all and has_npz

                if npz_any and not npz_all:
                    break

            if not npz_any:
                use_npz_latents = False
                logger.warning(f"npz file does not exist. ignore npz files / npzファイルが見つからないためnpzファイルを無視します")
            elif not npz_all:
                use_npz_latents = False
                logger.warning(
                    f"some of npz file does not exist. ignore npz files / いくつかのnpzファイルが見つからないためnpzファイルを無視します"
                )
                if flip_aug_in_subset:
                    logger.warning("maybe no flipped files / 反転されたnpzファイルがないのかもしれません")
        # else:
        #   logger.info("npz files are not used with color_aug and/or random_crop / color_augまたはrandom_cropが指定されているためnpzファイルは使用されません")

        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            if image_info.image_size is None:
                sizes = None  # not calculated
                break
            sizes.add(image_info.image_size[0])
            sizes.add(image_info.image_size[1])
            resos.add(tuple(image_info.image_size))

        if sizes is None:
            if use_npz_latents:
                use_npz_latents = False
                logger.warning(
                    f"npz files exist, but no bucket info in metadata. ignore npz files / メタデータにbucket情報がないためnpzファイルを無視します"
                )

            assert (
                resolution is not None
            ), "if metadata doesn't have bucket info, resolution is required / メタデータにbucket情報がない場合はresolutionを指定してください"

            self.enable_bucket = enable_bucket
            if self.enable_bucket:
                min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                    resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
                )
                self.min_bucket_reso = min_bucket_reso
                self.max_bucket_reso = max_bucket_reso
                self.bucket_reso_steps = bucket_reso_steps
                self.bucket_no_upscale = bucket_no_upscale
        else:
            if not enable_bucket:
                logger.info("metadata has bucket info, enable bucketing / メタデータにbucket情報があるためbucketを有効にします")
            logger.info("using bucket info in metadata / メタデータ内のbucket情報を使います")
            self.enable_bucket = True

            assert (
                not bucket_no_upscale
            ), "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used / メタデータ内にbucket情報がある場合はbucketの解像度は計算済みのため、bucket_no_upscaleは使えません"

            # bucket情報を初期化しておく、make_bucketsで再作成しない
            self.bucket_manager = BucketManager(False, None, None, None, None)
            self.bucket_manager.set_predefined_resos(resos)

        # npz情報をきれいにしておく
        if not use_npz_latents:
            for image_info in self.image_data.values():
                image_info.latents_npz = image_info.latents_npz_flipped = None

    def image_key_to_npz_file(self, subset: FineTuningSubset, image_key):
        base_name = os.path.splitext(image_key)[0]
        npz_file_norm = base_name + ".npz"

        if os.path.exists(npz_file_norm):
            # image_key is full path
            npz_file_flip = base_name + "_flip.npz"
            if not os.path.exists(npz_file_flip):
                npz_file_flip = None
            return npz_file_norm, npz_file_flip

        # if not full path, check image_dir. if image_dir is None, return None
        if subset.image_dir is None:
            return None, None

        # image_key is relative path
        npz_file_norm = os.path.join(subset.image_dir, image_key + ".npz")
        npz_file_flip = os.path.join(subset.image_dir, image_key + "_flip.npz")

        if not os.path.exists(npz_file_norm):
            npz_file_norm = None
            npz_file_flip = None
        elif not os.path.exists(npz_file_flip):
            npz_file_flip = None

        return npz_file_norm, npz_file_flip


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],        
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_cropはControlNetDatasetではサポートされていません"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
        )

        # config_util等から参照される値をいれておく（若干微妙なのでなんとかしたい）
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed 
        self.resize_interpolation = resize_interpolation

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / 制御用画像が見つかりませんでした: {missing_imgs}"
        assert (
            len(extra_imgs) == 0
        ), f"extra conditioning data for {len(extra_imgs)} images / 余分な制御用画像があります: {extra_imgs}"

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]
            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / 画像サイズが合いません: {image_info.absolute_path}"

                cond_img = resize_image(cond_img, original_size_hw[1], original_size_hw[0], target_size_hw[1], target_size_hw[0], self.resize_interpolation)

                # TODO support random crop
                # 現在サポートしているcropはrandomではなく中央のみ
                h, w = target_size_hw
                ct = (cond_img.shape[0] - h) // 2
                cl = (cond_img.shape[1] - w) // 2
                cond_img = cond_img[ct : ct + h, cl : cl + w]
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = resize_image(cond_img, cond_img.shape[0], cond_img.shape[1], target_size_hw[1], target_size_hw[0], self.resize_interpolation)

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets: List[Union[DreamBoothDataset, FineTuningDataset]]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def set_text_encoder_output_caching_strategy(self, strategy: TextEncoderOutputsCachingStrategy):
        """
        DataLoader is run in multiple processes, so we need to set the strategy manually.
        """
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process, file_suffix)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(tokenizers, text_encoders, device, weight_dtype, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs_sd3(
                tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk, is_main_process, batch_size
            )

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return [(dataset.width, dataset.height) for dataset in self.datasets]

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self) -> bool:
        return all([dataset.is_text_encoder_output_cacheable() for dataset in self.datasets])

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool, alpha_mask: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_resoはWxHなので注意

    if not os.path.exists(npz_path):
        return False

    try:
        npz = np.load(npz_path)
        if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
            return False
        if npz["latents"].shape[1:3] != expected_latents_size:
            return False

        if flip_aug:
            if "latents_flipped" not in npz:
                return False
            if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                return False

        if alpha_mask:
            if "alpha_mask" not in npz:
                return False
            if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:  # HxW => WxH != reso
                return False
        else:
            if "alpha_mask" in npz:
                return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


# 戻り値は、latents_tensor, (original_size width, original_size height), (crop left, crop top)
# TODO update to use CachingStrategy
# def load_latents_from_disk(
#     npz_path,
# ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
#     npz = np.load(npz_path)
#     if "latents" not in npz:
#         raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

#     latents = npz["latents"]
#     original_size = npz["original_size"].tolist()
#     crop_ltrb = npz["crop_ltrb"].tolist()
#     flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
#     alpha_mask = npz["alpha_mask"] if "alpha_mask" in npz else None
#     return latents, original_size, crop_ltrb, flipped_latents, alpha_mask


# def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
#     kwargs = {}
#     if flipped_latents_tensor is not None:
#         kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
#     if alpha_mask is not None:
#         kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
#     np.savez(
#         npz_path,
#         latents=latents_tensor.float().cpu().numpy(),
#         original_size=np.array(original_size),
#         crop_ltrb=np.array(crop_ltrb),
#         **kwargs,
#     )


def debug_dataset(train_dataset, show_input_ids=False):
    logger.info(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    logger.info(
        "`S` for next step, `E` for next epoch no. , Escape for exit. / Sキーで次のステップ、Eキーで次のエポック、Escキーで中断、終了します"
    )

    epoch = 1
    while True:
        logger.info(f"")
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    # example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    logger.info(f"network multiplier: {example['network_multipliers'][j]}")
                if "custom_attributes" in example:
                    logger.info(f"custom attributes: {example['custom_attributes'][j]}")

                # if show_input_ids:
                #     logger.info(f"input ids: {iid}")
                #     if "input_ids2" in example:
                #         logger.info(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if "alpha_masks" in example and example["alpha_masks"] is not None:
                        alpha_mask = example["alpha_masks"][j]
                        logger.info(f"alpha mask size: {alpha_mask.size()}")
                        alpha_mask = (alpha_mask.numpy() * 255.0).astype(np.uint8)
                        if os.name == "nt":
                            cv2.imshow("alpha_mask", alpha_mask)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # 重複を排除
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths


class MinimalDataset(BaseDataset):
    def __init__(self, resolution, network_multiplier, debug_dataset=False):
        super().__init__(resolution, network_multiplier, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = self.get_input_ids(caption)
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return []


def load_arbitrary_dataset(args, tokenizer=None) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(tokenizer, args.max_token_length, args.resolution, args.debug_dataset)
    return train_dataset_group


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except (IOError, OSError) as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


# 画像を読み込む。戻り値はnumpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int], resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random cropの場合のcropされた値をどうcrop left/topに反映するべきか全くアイデアがない
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation)

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae: AutoencoderKL, cache_to_disk: bool, image_infos: List[ImageInfo], flip_aug: bool, use_alpha_mask: bool, random_crop: bool
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation)

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(image_infos, latents, flipped_latents, alpha_masks):
        # check NaN
        if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

        if cache_to_disk:
            # save_latents_to_disk(
            #     info.latents_npz,
            #     latent,
            #     info.latents_original_size,
            #     info.latents_crop_ltrb,
            #     flipped_latent,
            #     alpha_mask,
            # )
            pass
        else:
            info.latents = latent
            if flip_aug:
                info.latents_flipped = flipped_latent
            info.alpha_mask = alpha_mask

    if not HIGH_VRAM:
        clean_memory_on_device(vae.device)


def cache_batch_text_encoder_outputs(
    image_infos, tokenizers, text_encoders, max_token_length, cache_to_disk, input_ids1, input_ids2, dtype
):
    input_ids1 = input_ids1.to(text_encoders[0].device)
    input_ids2 = input_ids2.to(text_encoders[1].device)

    with torch.no_grad():
        b_hidden_state1, b_hidden_state2, b_pool2 = get_hidden_states_sdxl(
            max_token_length,
            input_ids1,
            input_ids2,
            tokenizers[0],
            tokenizers[1],
            text_encoders[0],
            text_encoders[1],
            dtype,
        )

        # ここでcpuに移動しておかないと、上書きされてしまう
        b_hidden_state1 = b_hidden_state1.detach().to("cpu")  # b,n*75+2,768
        b_hidden_state2 = b_hidden_state2.detach().to("cpu")  # b,n*75+2,1280
        b_pool2 = b_pool2.detach().to("cpu")  # b,1280

    for info, hidden_state1, hidden_state2, pool2 in zip(image_infos, b_hidden_state1, b_hidden_state2, b_pool2):
        if cache_to_disk:
            save_text_encoder_outputs_to_disk(info.text_encoder_outputs_npz, hidden_state1, hidden_state2, pool2)
        else:
            info.text_encoder_outputs1 = hidden_state1
            info.text_encoder_outputs2 = hidden_state2
            info.text_encoder_pool2 = pool2


def cache_batch_text_encoder_outputs_sd3(
    image_infos, tokenizer, text_encoders, max_token_length, cache_to_disk, input_ids, output_dtype
):
    # make input_ids for each text encoder
    l_tokens, g_tokens, t5_tokens = input_ids

    clip_l, clip_g, t5xxl = text_encoders
    with torch.no_grad():
        b_lg_out, b_t5_out, b_pool = sd3_utils.get_cond_from_tokens(
            l_tokens, g_tokens, t5_tokens, clip_l, clip_g, t5xxl, "cpu", output_dtype
        )
        b_lg_out = b_lg_out.detach()
        b_t5_out = b_t5_out.detach()
        b_pool = b_pool.detach()

    for info, lg_out, t5_out, pool in zip(image_infos, b_lg_out, b_t5_out, b_pool):
        # debug: NaN check
        if torch.isnan(lg_out).any() or torch.isnan(t5_out).any() or torch.isnan(pool).any():
            raise RuntimeError(f"NaN detected in text encoder outputs: {info.absolute_path}")

        if cache_to_disk:
            save_text_encoder_outputs_to_disk(info.text_encoder_outputs_npz, lg_out, t5_out, pool)
        else:
            info.text_encoder_outputs1 = lg_out
            info.text_encoder_outputs2 = t5_out
            info.text_encoder_pool2 = pool


def save_text_encoder_outputs_to_disk(npz_path, hidden_state1, hidden_state2, pool2):
    np.savez(
        npz_path,
        hidden_state1=hidden_state1.cpu().float().numpy(),
        hidden_state2=hidden_state2.cpu().float().numpy(),
        pool2=pool2.cpu().float().numpy(),
    )


def load_text_encoder_outputs_from_disk(npz_path):
    with np.load(npz_path) as f:
        hidden_state1 = torch.from_numpy(f["hidden_state1"])
        hidden_state2 = torch.from_numpy(f["hidden_state2"]) if "hidden_state2" in f else None
        pool2 = torch.from_numpy(f["pool2"]) if "pool2" in f else None
    return hidden_state1, hidden_state2, pool2


# endregion

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
    except:
        return "(unknown)"


# def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
#     replace_attentions_for_hypernetwork()
#     # unet is not used currently, but it is here for future use
#     unet.enable_xformers_memory_efficient_attention()
#     return
#     if mem_eff_attn:
#         unet.set_attn_processor(FlashAttnProcessor())
#     elif xformers:
#         unet.enable_xformers_memory_efficient_attention()


# def replace_unet_cross_attn_to_xformers():
#     logger.info("CrossAttention.forward has been replaced to enable xformers.")
#     try:
#         import xformers.ops
#     except ImportError:
#         raise ImportError("No xformers / xformersがインストールされていないようです")

#     def forward_xformers(self, x, context=None, mask=None):
#         h = self.heads
#         q_in = self.to_q(x)

#         context = default(context, x)
#         context = context.to(x.dtype)

#         if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
#             context_k, context_v = self.hypernetwork.forward(x, context)
#             context_k = context_k.to(x.dtype)
#             context_v = context_v.to(x.dtype)
#         else:
#             context_k = context
#             context_v = context

#         k_in = self.to_k(context_k)
#         v_in = self.to_v(context_v)

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
#         del q_in, k_in, v_in

#         q = q.contiguous()
#         k = k.contiguous()
#         v = v.contiguous()
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

#         out = rearrange(out, "b n h d -> b n (h d)", h=h)

#         # diffusers 0.7.0~
#         out = self.to_out[0](out)
#         out = self.to_out[1](out)
#         return out


#     diffusers.models.attention.CrossAttention.forward = forward_xformers
def replace_unet_modules(unet: UNet2DConditionModel, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)


"""
def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers):
    # vae is not used currently, but it is here for future use
    if mem_eff_attn:
        replace_vae_attn_to_memory_efficient()
    elif xformers:
        # とりあえずDiffusersのxformersを使う。AttentionがあるのはMidBlockのみ
        logger.info("Use Diffusers xformers for VAE")
        vae.encoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)
        vae.decoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)


def replace_vae_attn_to_memory_efficient():
    logger.info("AttentionBlock.forward has been replaced to FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, hidden_states):
        logger.info("forward_flash_attn")
        q_bucket_size = 512
        k_bucket_size = 1024

        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (query_proj, key_proj, value_proj)
        )

        out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    diffusers.models.attention.AttentionBlock.forward = forward_flash_attn
"""


# endregion


# region arguments


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


# this metadata is referred from train_network and various scripts, so we wrote here
SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def build_minimum_network_metadata(
    v2: Optional[str],
    base_model: Optional[str],
    network_module: str,
    network_dim: str,
    network_alpha: str,
    network_args: Optional[dict],
):
    # old LoRA doesn't have base_model
    metadata = {
        SS_METADATA_KEY_NETWORK_MODULE: network_module,
        SS_METADATA_KEY_NETWORK_DIM: network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: network_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


def get_sai_model_spec(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,  # None for TI and LoRA
    sd3: str = None,
    flux: str = None,
):
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    metadata = sai_model_spec.build_metadata(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,  # None or int
        sd3=sd3,
        flux=flux,
    )
    return metadata


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    # for pretrained models
    parser.add_argument(
        "--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む"
    )
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training / v-parameterization学習を有効にする"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(f"Value '{value}' is not a valid percentage")
        try:
            float_value = float(value)
            if float_value >= 1:
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, "
        "Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, "
        "DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, "
        "AdaFactor. "
        "Also, you can use any optimizer by specifying the full path to the class, like 'bitsandbytes.optim.AdEMAMix8bit' or 'bitsandbytes.optim.PagedAdEMAMix8bit'.",
    )

    # backward compatibility
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）",
    )

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない",
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )

    # parser.add_argument(
    #     "--optimizer_schedulefree_wrapper",
    #     action="store_true",
    #     help="use schedulefree_wrapper any optimizer / 任意のオプティマイザにschedulefree_wrapperを使用",
    # )

    # parser.add_argument(
    #     "--schedulefree_wrapper_args",
    #     type=str,
    #     default=None,
    #     nargs="*",
    #     help='additional arguments for schedulefree_wrapper (like "momentum=0.9 weight_decay_at_y=0.1 ...") / オプティマイザの追加引数（例： "momentum=0.9 weight_decay_at_y=0.1 ..."）',
    # )

    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps"
        " / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps"
        " / 学習率のスケジューラを減衰させるステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Combines backward pass and optimizer step to reduce VRAM usage. Only available in SDXL, SD3 and FLUX"
        " / バックワードパスとオプティマイザステップを組み合わせてVRAMの使用量を削減します。SDXL、SD3、FLUXでのみ利用可能",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`"
        + " / 逆平方根スケジューラのタイムスケール、デフォルトは`num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler and warmup decay scheduler"
        + " / 初期学習率の比率としての最小学習率を指定する、cosine with min lr と warmup decay スケジューラ で有効",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ"
    )
    parser.add_argument(
        "--output_name", type=str, default=None, help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名"
    )
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス",
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving / 保存時に精度を変更して保存する",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する",
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total) / 学習中のモデルを指定のエポック割合で保存する（たとえば5を指定すると最低5個のファイルが保存される）",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / 指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model / optimizerなど学習状態も含めたstateをモデル保存時に追加で保存する",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end / optimizerなど学習状態も含めたstateを学習完了時に保存する",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225],
        help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）",
    )
    parser.add_argument(
        "--mem_eff_attn",
        action="store_true",
        help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う",
    )
    parser.add_argument(
        "--torch_compile", action="store_true", help="use torch.compile (requires PyTorch 2.0) / torch.compile を使う"
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="inductor",
        # available backends:
        # https://github.com/huggingface/accelerate/blob/d1abd59114ada8ba673e1214218cb2878c13b82d/src/accelerate/utils/dataclasses.py#L376-L388C5
        # https://pytorch.org/docs/stable/torch.compiler.html
        choices=[
            "eager",
            "aot_eager",
            "inductor",
            "aot_ts_nvfuser",
            "nvprims_nvfuser",
            "cudagraphs",
            "ofi",
            "fx2trt",
            "onnxrt",
            "tensort",
            "ipex",
            "tvm",
        ],
        help="dynamo backend type (default is inductor) / dynamoのbackendの種類（デフォルトは inductor）",
    )
    parser.add_argument("--xformers", action="store_true", help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0) / CrossAttentionにsdpaを使う（PyTorch 2.0が必要）",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ",
    )

    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="enable gradient checkpointing / gradient checkpointingを有効にする"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度",
    )
    parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument(
        "--full_bf16", action="store_true", help="bf16 training including gradients / 勾配も含めてbf16で学習する"
    )  # TODO move to SDXL training, because it is not supported by SD1/2
    parser.add_argument("--fp8_base", action="store_true", help="use fp8 for base model / base modelにfp8を使う")

    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="DDP timeout (min, None for default of accelerate) / DDPのタイムアウト（分、Noneでaccelerateのデフォルト）",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="enable gradient_as_bucket_view for DDP / DDPでgradient_as_bucket_viewを有効にする",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        help="enable static_graph for DDP / DDPでstatic_graphを有効にする",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session / wandb ログに表示される特定の実行の名前",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging / ログ出力に使用するtrackerの設定ファイルのパス",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )
    parser.add_argument("--log_config", action="store_true", help="log training configuration / 学習設定をログに出力する")

    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）",
    )
    parser.add_argument(
        "--noise_offset_random_strength",
        action="store_true",
        help="use random strength between 0~noise_offset for noise offset. / noise offsetにおいて、0からnoise_offsetの間でランダムな強度を使用します。",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) / Multires noiseを有効にしてこのイテレーション数を設定する（有効にする場合は6-10程度を推奨）",
    )
    parser.add_argument(
        "--ip_noise_gamma",
        type=float,
        default=None,
        help="enable input perturbation noise. used for regularization. recommended value: around 0.1 (from arxiv.org/abs/2301.11706) "
        + "/  input perturbation noiseを有効にする。正則化に使用される。推奨値: 0.1程度 (arxiv.org/abs/2301.11706 より)",
    )
    parser.add_argument(
        "--ip_noise_gamma_random_strength",
        action="store_true",
        help="Use random strength between 0~ip_noise_gamma for input perturbation noise."
        + "/ input perturbation noiseにおいて、0からip_noise_gammaの間でランダムな強度を使用します。",
    )
    # parser.add_argument(
    #     "--perlin_noise",
    #     type=int,
    #     default=None,
    #     help="enable perlin noise and set the octaves / perlin noiseを有効にしてoctavesをこの値に設定する",
    # )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations) / Multires noiseのdiscount値を設定する（--multires_noise_iterations指定時のみ有効）",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default) / latentの平均値の絶対値 * この値をnoise_offsetに加算する（Noneの場合は無効、デフォルト）",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR / noise schedulerのbetasを修正して、zero terminal SNRを強制する",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for U-Net training (0~999, default is 0) / U-Net学習時のtime stepの最小値を設定する（0~999で指定、省略時はデフォルト値(0)） ",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000) / U-Net学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2", "huber", "smooth_l1"],
        help="The type of loss function to use (L1, L2, Huber, or smooth L1), default is L2 / 使用する損失関数の種類（L1、L2、Huber、またはsmooth L1）、デフォルトはL2",
    )
    parser.add_argument(
        "--huber_schedule",
        type=str,
        default="snr",
        choices=["constant", "exponential", "snr"],
        help="The scheduling method for Huber loss (constant, exponential, or SNR-based). Only used when loss_type is 'huber' or 'smooth_l1'. default is snr"
        + " / Huber損失のスケジューリング方法（constant、exponential、またはSNRベース）。loss_typeが'huber'または'smooth_l1'の場合に有効、デフォルトは snr",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.1,
        help="The Huber loss decay parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type. default is 0.1"
        " / Huber損失の減衰パラメータ。loss_typeがhuberまたはsmooth l1の場合に有効。デフォルトは0.1",
    )

    parser.add_argument(
        "--huber_scale",
        type=float,
        default=1.0,
        help="The Huber loss scale parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type. default is 1.0"
        " / Huber損失のスケールパラメータ。loss_typeがhuberまたはsmooth l1の場合に有効。デフォルトは1.0",
    )

    parser.add_argument(
        "--lowram",
        action="store_true",
        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込む等（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）",
    )
    parser.add_argument(
        "--highvram",
        action="store_true",
        help="disable low VRAM optimization. e.g. do not clear CUDA cache after each latent caching (for machines which have bigger VRAM) "
        + "/ VRAMが少ない環境向け最適化を無効にする。たとえば各latentのキャッシュ後のCUDAキャッシュクリアを行わない等（VRAMが多い環境向け）",
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する",
    )
    parser.add_argument(
        "--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル",
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す",
    )
    parser.add_argument(
        "--output_config", action="store_true", help="output command line args to given .toml file / 引数を.tomlファイルに出力する"
    )

    # SAI Model spec
    parser.add_argument(
        "--metadata_title",
        type=str,
        default=None,
        help="title for model metadata (default is output_name) / メタデータに書き込まれるモデルタイトル、省略時はoutput_name",
    )
    parser.add_argument(
        "--metadata_author",
        type=str,
        default=None,
        help="author name for model metadata / メタデータに書き込まれるモデル作者名",
    )
    parser.add_argument(
        "--metadata_description",
        type=str,
        default=None,
        help="description for model metadata / メタデータに書き込まれるモデル説明",
    )
    parser.add_argument(
        "--metadata_license",
        type=str,
        default=None,
        help="license for model metadata / メタデータに書き込まれるモデルライセンス",
    )
    parser.add_argument(
        "--metadata_tags",
        type=str,
        default=None,
        help="tags for model metadata, separated by comma / メタデータに書き込まれるモデルタグ、カンマ区切り",
    )

    if support_dreambooth:
        # DreamBooth training
        parser.add_argument(
            "--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み"
        )


def add_masked_loss_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory / 条件付けデータのディレクトリ",
    )
    parser.add_argument(
        "--masked_loss",
        action="store_true",
        help="apply mask for calculating loss. conditioning_data_dir is required for dataset. / 損失計算時にマスクを適用する。datasetにはconditioning_data_dirが必要",
    )


def add_dit_training_arguments(parser: argparse.ArgumentParser):
    # Text encoder related arguments
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)"
        + " / text encoderのバッチサイズ（デフォルト: None, データセットのバッチサイズを使用）",
    )

    # Model loading optimization
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )

    # Training arguments. partial copy from Diffusers
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none", "uniform"],
        help="weighting scheme for timestep distribution. Default is uniform, uniform and none are the same behavior"
        " / タイムステップ分布の重み付けスキーム、デフォルトはuniform、uniform と none は同じ挙動",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合の平均",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合のstd",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme` / モード重み付けスキームのスケール",
    )

    # offloading
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="[EXPERIMENTAL] "
        "Sets the number of blocks to swap during the forward and backward passes."
        "Increasing this number lowers the overall VRAM used during training at the expense of training speed (s/it)."
        " / 順伝播および逆伝播中にスワップするブロックの数を設定します。"
        "この数を増やすと、トレーニング中のVRAM使用量が減りますが、トレーニング速度（s/it）も低下します。",
    )


def get_sanitized_config_or_none(args: argparse.Namespace):
    # if `--log_config` is enabled, return args for logging. if not, return None.
    # when `--log_config is enabled, filter out sensitive values from args
    # if wandb is not enabled, the log is not exposed to the public, but it is fine to filter out sensitive values to be safe

    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        # filter out sensitive values and convert to string if necessary
        if k not in sensitive_args + sensitive_path_args:
            # Accelerate values need to have type `bool`,`str`, `float`, `int`, or `None`.
            if v is None or isinstance(v, bool) or isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                filtered_args[k] = v
            # accelerate does not support lists
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            # accelerate does not support objects
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


# verify command line args for training
def verify_command_line_training_args(args: argparse.Namespace):
    # if wandb is enabled, the command line is exposed to the public
    # check whether sensitive options are included in the command line arguments
    # if so, warn or inform the user to move them to the configuration file
    # wandbが有効な場合、コマンドラインが公開される
    # 学習用のコマンドライン引数に敏感なオプションが含まれているかどうかを確認し、
    # 含まれている場合は設定ファイルに移動するようにユーザーに警告または通知する

    wandb_enabled = args.log_with is not None and args.log_with != "tensorboard"  # "all" or "wandb"
    if not wandb_enabled:
        return

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]

    for arg in sensitive_args:
        if getattr(args, arg, None) is not None:
            logger.warning(
                f"wandb is enabled, but option `{arg}` is included in the command line. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれています。コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
            )

    # if path is absolute, it may include sensitive information
    for arg in sensitive_path_args:
        if getattr(args, arg, None) is not None and os.path.isabs(getattr(args, arg)):
            logger.info(
                f"wandb is enabled, but option `{arg}` is included in the command line and it is an absolute path. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file or use relative path."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれており、絶対パスです。コマンドラインは公開されるため、`.toml`ファイルに移動するか、相対パスを使用することをお勧めします。"
            )

    if getattr(args, "config_file", None) is not None:
        logger.info(
            f"wandb is enabled, but option `config_file` is included in the command line. Because the command line is exposed to the public, please be careful about the information included in the path."
            + f" / wandbが有効で、かつオプション `config_file` がコマンドラインに含まれています。コマンドラインは公開されるため、パスに含まれる情報にご注意ください。"
        )

    # other sensitive options
    if args.huggingface_repo_id is not None and args.huggingface_repo_visibility != "public":
        logger.info(
            f"wandb is enabled, but option huggingface_repo_id is included in the command line and huggingface_repo_visibility is not 'public'. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
            + f" / wandbが有効で、かつオプション huggingface_repo_id がコマンドラインに含まれており、huggingface_repo_visibility が 'public' ではありません。コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
        )


def enable_high_vram(args: argparse.Namespace):
    if args.highvram:
        logger.info("highvram is enabled / highvramが有効です")
        global HIGH_VRAM
        HIGH_VRAM = True


def verify_training_args(args: argparse.Namespace):
    r"""
    Verify training arguments. Also reflect highvram option to global variable
    学習用引数を検証する。あわせて highvram オプションの指定をグローバル変数に反映する
    """
    enable_high_vram(args)

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        logger.warning(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled / cache_latents_to_diskが有効なため、cache_latentsを有効にします"
        )

    # noise_offset, perlin_noise, multires_noise_iterations cannot be enabled at the same time
    # # Listを使って数えてもいいけど並べてしまえ
    # if args.noise_offset is not None and args.multires_noise_iterations is not None:
    #     raise ValueError(
    #         "noise_offset and multires_noise_iterations cannot be enabled at the same time / noise_offsetとmultires_noise_iterationsを同時に有効にできません"
    #     )
    # if args.noise_offset is not None and args.perlin_noise is not None:
    #     raise ValueError("noise_offset and perlin_noise cannot be enabled at the same time / noise_offsetとperlin_noiseは同時に有効にできません")
    # if args.perlin_noise is not None and args.multires_noise_iterations is not None:
    #     raise ValueError(
    #         "perlin_noise and multires_noise_iterations cannot be enabled at the same time / perlin_noiseとmultires_noise_iterationsを同時に有効にできません"
    #     )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset / adaptive_noise_scaleを使用するにはnoise_offsetが必要です")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization / scale_v_pred_loss_like_noise_predはv_parameterizationが有効なときのみ有効にできます"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError(
            "v_pred_like_loss cannot be enabled with v_parameterization / v_pred_like_lossはv_parameterizationが有効なときには有効にできません"
        )

    if args.zero_terminal_snr and not args.v_parameterization:
        logger.warning(
            f"zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
            + " / zero_terminal_snrが有効ですが、v_parameterizationが有効ではありません。学習結果は想定外になる可能性があります"
        )

    if args.sample_every_n_epochs is not None and args.sample_every_n_epochs <= 0:
        logger.warning(
            "sample_every_n_epochs is less than or equal to 0, so it will be disabled / sample_every_n_epochsに0以下の値が指定されたため無効になります"
        )
        args.sample_every_n_epochs = None

    if args.sample_every_n_steps is not None and args.sample_every_n_steps <= 0:
        logger.warning(
            "sample_every_n_steps is less than or equal to 0, so it will be disabled / sample_every_n_stepsに0以下の値が指定されたため無効になります"
        )
        args.sample_every_n_steps = None


def add_dataset_arguments(
    parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool, support_caption_dropout: bool
):
    # dataset common
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ"
    )
    parser.add_argument(
        "--cache_info",
        action="store_true",
        help="cache meta information (caption and image size) for faster dataset loading. only available for DreamBooth"
        + " / メタ情報（キャプションとサイズ）をキャッシュしてデータセット読み込みを高速化する。DreamBooth方式のみ有効",
    )
    parser.add_argument(
        "--shuffle_caption", action="store_true", help="shuffle separated caption / 区切られたcaptionの各要素をshuffleする"
    )
    parser.add_argument("--caption_separator", type=str, default=",", help="separator for caption / captionの区切り文字")
    parser.add_argument(
        "--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子"
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings) / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す（トークンはカンマ区切りの各部分を意味する）",
    )
    parser.add_argument(
        "--keep_tokens_separator",
        type=str,
        default="",
        help="A custom separator to divide the caption into fixed and flexible parts. Tokens before this separator will not be shuffled. If not specified, '--keep_tokens' will be used to determine the fixed number of tokens."
        + " / captionを固定部分と可変部分に分けるためのカスタム区切り文字。この区切り文字より前のトークンはシャッフルされない。指定しない場合、'--keep_tokens'が固定部分のトークン数として使用される。",
    )
    parser.add_argument(
        "--secondary_separator",
        type=str,
        default=None,
        help="a secondary separator for caption. This separator is replaced to caption_separator after dropping/shuffling caption"
        + " / captionのセカンダリ区切り文字。この区切り文字はcaptionのドロップやシャッフル後にcaption_separatorに置き換えられる",
    )
    parser.add_argument(
        "--enable_wildcard",
        action="store_true",
        help="enable wildcard for caption (e.g. '{image|picture|rendition}') / captionのワイルドカードを有効にする（例：'{image|picture|rendition}'）",
    )
    parser.add_argument(
        "--caption_prefix",
        type=str,
        default=None,
        help="prefix for caption text / captionのテキストの先頭に付ける文字列",
    )
    parser.add_argument(
        "--caption_suffix",
        type=str,
        default=None,
        help="suffix for caption text / captionのテキストの末尾に付ける文字列",
    )
    parser.add_argument(
        "--color_aug", action="store_true", help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする"
    )
    parser.add_argument(
        "--flip_aug", action="store_true", help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする"
    )
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / 学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="enable random crop (for style training in face-centered crop augmentation) / ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）",
    )
    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをメインメモリにcacheする（augmentationは使用不可） ",
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=1, help="batch size for caching latents / latentのcache時のバッチサイズ"
    )
    parser.add_argument(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをディスクにcacheする（augmentationは使用不可）",
    )
    parser.add_argument(
        "--skip_cache_check",
        action="store_true",
        help="skip the content validation of cache (latent and text encoder output). Cache file existence check is always performed, and cache processing is performed if the file does not exist"
        " / cacheの内容の検証をスキップする（latentとテキストエンコーダの出力）。キャッシュファイルの存在確認は常に行われ、ファイルがなければキャッシュ処理が行われる",
    )
    parser.add_argument(
        "--enable_bucket",
        action="store_true",
        help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする",
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=256,
        help="minimum resolution for buckets, must be divisible by bucket_reso_steps "
        " / bucketの最小解像度、bucket_reso_stepsで割り切れる必要があります",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=1024,
        help="maximum resolution for buckets, must be divisible by bucket_reso_steps "
        " / bucketの最大解像度、bucket_reso_stepsで割り切れる必要があります",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します",
    )
    parser.add_argument(
        "--resize_interpolation",
        type=str,
        default=None,
        choices=["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"],
        help="Resize interpolation when required. Default: area Options: lanczos, nearest, bilinear, bicubic, area / 必要に応じてサイズ補間を変更します。デフォルト: area オプション: lanczos, nearest, bilinear, bicubic, area",
    )
    parser.add_argument(
        "--token_warmup_min",
        type=int,
        default=1,
        help="start learning at N tags (token means comma separated strinfloatgs) / タグ数をN個から増やしながら学習する",
    )
    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps (or N*max_train_steps if N<1) / N（N<1ならN*max_train_steps）ステップでタグ長が最大になる。デフォルトは0（最初から最大）",
    )
    parser.add_argument(
        "--alpha_mask",
        action="store_true",
        help="use alpha channel as mask for training / 画像のアルファチャンネルをlossのマスクに使用する",
    )

    parser.add_argument(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class) / 任意のデータセットを用いるときのクラス名 (package.module.Class)",
    )

    if support_caption_dropout:
        # Textual Inversion はcaptionのdropoutをsupportしない
        # いわゆるtensorのDropoutと紛らわしいのでprefixにcaptionを付けておく　every_n_epochsは他と平仄を合わせてdefault Noneに
        parser.add_argument(
            "--caption_dropout_rate", type=float, default=0.0, help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合"
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合",
        )

    if support_dreambooth:
        # DreamBooth dataset
        parser.add_argument(
            "--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ"
        )

    if support_caption:
        # caption dataset
        parser.add_argument(
            "--in_json", type=str, default=None, help="json metadata for dataset / データセットのmetadataのjsonファイル"
        )
        parser.add_argument(
            "--dataset_repeats",
            type=int,
            default=1,
            help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数",
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）",
    )


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    if args.output_config:
        # check if config file exists
        if os.path.exists(config_path):
            logger.error(f"Config file already exists. Aborting... / 出力先の設定ファイルが既に存在します: {config_path}")
            exit(1)

        # convert args to dictionary
        args_dict = vars(args)

        # remove unnecessary keys
        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        # get default args from parser
        default_args = vars(parser.parse_args([]))

        # remove default values: cannot use args_dict.items directly because it will be changed during iteration
        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        # convert Path to str in dictionary
        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        # convert to toml and output to file
        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        logger.info(f"Saved config file / 設定ファイルを保存しました: {config_path}")
        exit(0)

    if not os.path.exists(config_path):
        logger.info(f"{config_path} not found.")
        exit(1)

    logger.info(f"Loading settings from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]

    return args


# endregion

# region utils


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        logger.info(f"resume training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return

    logger.info(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

    list_files = huggingface_util.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
    if len(results) == 0:
        raise ValueError(
            "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした"
        )
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)


def get_optimizer(args, trainable_params) -> tuple[str, str, object]:
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, AdEMAMix8bit, PagedAdEMAMix8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

    optimizer_type = args.optimizer_type
    if args.use_8bit_adam:
        assert (
            not args.use_lion_optimizer
        ), "both option use_8bit_adam and use_lion_optimizer are specified / use_8bit_adamとuse_lion_optimizerの両方のオプションが指定されています"
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_8bit_adam and optimizer_type are specified / use_8bit_adamとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "AdamW8bit"

    elif args.use_lion_optimizer:
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_lion_optimizer and optimizer_type are specified / use_lion_optimizerとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    if args.fused_backward_pass:
        assert (
            optimizer_type == "Adafactor".lower()
        ), "fused_backward_pass currently only works with optimizer_type Adafactor / fused_backward_passは現在optimizer_type Adafactorでのみ機能します"
        assert (
            args.gradient_accumulation_steps == 1
        ), "fused_backward_pass does not work with gradient_accumulation_steps > 1 / fused_backward_passはgradient_accumulation_steps>1では機能しません"

    # 引数を分解する
    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)

            # value = value.split(",")
            # for i in range(len(value)):
            #     if value[i].lower() == "true" or value[i].lower() == "false":
            #         value[i] = value[i].lower() == "true"
            #     else:
            #         value[i] = ast.float(value[i])
            # if len(value) == 1:
            #     value = value[0]
            # else:
            #     value = tuple(value)

            optimizer_kwargs[key] = value
    # logger.info(f"optkwargs {optimizer}_{kwargs}")

    lr = args.learning_rate
    optimizer = None
    optimizer_class = None

    if optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです")
        logger.info(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

        if optimizer_type == "AdamW8bit".lower():
            logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "SGDNesterov8bit".lower():
            logger.info(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
            if "momentum" not in optimizer_kwargs:
                logger.warning(
                    f"8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / 8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します"
                )
                optimizer_kwargs["momentum"] = 0.9

            optimizer_class = bnb.optim.SGD8bit
            optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

        elif optimizer_type == "Lion8bit".lower():
            logger.info(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.Lion8bit
            except AttributeError:
                raise AttributeError(
                    "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later. / Lion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.38.0以上をインストールしてください"
                )
        elif optimizer_type == "PagedAdamW8bit".lower():
            logger.info(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedAdamW8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedAdamW8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
                )
        elif optimizer_type == "PagedLion8bit".lower():
            logger.info(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedLion8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedLion8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedLion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
                )

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW".lower():
        logger.info(f"use PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")
        try:
            optimizer_class = bnb.optim.PagedAdamW
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedAdamWが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
            )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW32bit".lower():
        logger.info(f"use 32-bit PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")
        try:
            optimizer_class = bnb.optim.PagedAdamW32bit
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW32bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedAdamW32bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
            )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        logger.info(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            logger.info(
                f"SGD with Nesterov must be with momentum, set momentum to 0.9 / SGD with Nesterovはmomentum指定が必須のため0.9に設定します"
            )
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type.startswith("DAdapt".lower()) or optimizer_type == "Prodigy".lower():
        # check lr and lr_count, and logger.info warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            logger.warning(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0 / 学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください: lr={actual_lr}"
            )
            logger.warning("recommend option: lr=1.0 / 推奨は1.0です")
        if lr_count > 1:
            logger.warning(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect / D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
            )

        if optimizer_type.startswith("DAdapt".lower()):
            # DAdaptation family
            # check dadaptation is installed
            try:
                import dadaptation
                import dadaptation.experimental as experimental
            except ImportError:
                raise ImportError("No dadaptation / dadaptation がインストールされていないようです")

            # set optimizer
            if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdamPreprint".lower():
                optimizer_class = experimental.DAdaptAdamPreprint
                logger.info(f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdaGrad".lower():
                optimizer_class = dadaptation.DAdaptAdaGrad
                logger.info(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdam".lower():
                optimizer_class = dadaptation.DAdaptAdam
                logger.info(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdan".lower():
                optimizer_class = dadaptation.DAdaptAdan
                logger.info(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdanIP".lower():
                optimizer_class = experimental.DAdaptAdanIP
                logger.info(f"use D-Adaptation AdanIP optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptLion".lower():
                optimizer_class = dadaptation.DAdaptLion
                logger.info(f"use D-Adaptation Lion optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptSGD".lower():
                optimizer_class = dadaptation.DAdaptSGD
                logger.info(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        else:
            # Prodigy
            # check Prodigy is installed
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("No Prodigy / Prodigy がインストールされていないようです")

            logger.info(f"use Prodigy optimizer | {optimizer_kwargs}")
            optimizer_class = prodigyopt.Prodigy
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            logger.info(
                f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
            )
            optimizer_kwargs["relative_step"] = True
        logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            logger.info(f"relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                logger.warning(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
            args.learning_rate = None

            # trainable_paramsがgroupだった時の処理：lrを削除する
            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    # 一応argsを無効にしておく TODO 依存関係が逆転してるのであまり望ましくない
                    logger.warning(f"unet_lr and text_encoder_lr are ignored / unet_lrとtext_encoder_lrは無視されます")
                    args.unet_lr = None
                    args.text_encoder_lr = None

            if args.lr_scheduler != "adafactor":
                logger.info(f"use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
            args.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

            lr = None
        else:
            if args.max_grad_norm != 0.0:
                logger.warning(
                    f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
                )
            if args.lr_scheduler != "constant_with_warmup":
                logger.warning(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                logger.warning(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("schedulefree".lower()):
        try:
            import schedulefree as sf
        except ImportError:
            raise ImportError("No schedulefree / schedulefreeがインストールされていないようです")

        if optimizer_type == "RAdamScheduleFree".lower():
            optimizer_class = sf.RAdamScheduleFree
            logger.info(f"use RAdamScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "AdamWScheduleFree".lower():
            optimizer_class = sf.AdamWScheduleFree
            logger.info(f"use AdamWScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "SGDScheduleFree".lower():
            optimizer_class = sf.SGDScheduleFree
            logger.info(f"use SGDScheduleFree optimizer | {optimizer_kwargs}")
        else:
            optimizer_class = None

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        # 任意のoptimizerを使う
        case_sensitive_optimizer_type = args.optimizer_type  # not lower
        logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

        if "." not in case_sensitive_optimizer_type:  # from torch.optim
            optimizer_module = torch.optim
        else:  # from other library
            values = case_sensitive_optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            case_sensitive_optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    """
    # wrap any of above optimizer with schedulefree, if optimizer is not schedulefree
    if args.optimizer_schedulefree_wrapper and not optimizer_type.endswith("schedulefree".lower()):
        try:
            import schedulefree as sf
        except ImportError:
            raise ImportError("No schedulefree / schedulefreeがインストールされていないようです")

        schedulefree_wrapper_kwargs = {}
        if args.schedulefree_wrapper_args is not None and len(args.schedulefree_wrapper_args) > 0:
            for arg in args.schedulefree_wrapper_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                schedulefree_wrapper_kwargs[key] = value

        sf_wrapper = sf.ScheduleFreeWrapper(optimizer, **schedulefree_wrapper_kwargs)
        sf_wrapper.train()  # make optimizer as train mode

        # we need to make optimizer as a subclass of torch.optim.Optimizer, we make another Proxy class over SFWrapper
        class OptimizerProxy(torch.optim.Optimizer):
            def __init__(self, sf_wrapper):
                self._sf_wrapper = sf_wrapper

            def __getattr__(self, name):
                return getattr(self._sf_wrapper, name)

            # override properties
            @property
            def state(self):
                return self._sf_wrapper.state

            @state.setter
            def state(self, state):
                self._sf_wrapper.state = state

            @property
            def param_groups(self):
                return self._sf_wrapper.param_groups

            @param_groups.setter
            def param_groups(self, param_groups):
                self._sf_wrapper.param_groups = param_groups

            @property
            def defaults(self):
                return self._sf_wrapper.defaults

            @defaults.setter
            def defaults(self, defaults):
                self._sf_wrapper.defaults = defaults

            def add_param_group(self, param_group):
                self._sf_wrapper.add_param_group(param_group)

            def load_state_dict(self, state_dict):
                self._sf_wrapper.load_state_dict(state_dict)

            def state_dict(self):
                return self._sf_wrapper.state_dict()

            def zero_grad(self):
                self._sf_wrapper.zero_grad()

            def step(self, closure=None):
                self._sf_wrapper.step(closure)

            def train(self):
                self._sf_wrapper.train()

            def eval(self):
                self._sf_wrapper.eval()

            # isinstance チェックをパスするためのメソッド
            def __instancecheck__(self, instance):
                return isinstance(instance, (type(self), Optimizer))

        optimizer = OptimizerProxy(sf_wrapper)

        logger.info(f"wrap optimizer with ScheduleFreeWrapper | {schedulefree_wrapper_kwargs}")
    """

    # for logging
    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    if hasattr(optimizer, "train") and callable(optimizer.train):
        # make optimizer as train mode before training for schedulefree optimizer. the optimizer will be in eval mode in sampling and saving.
        optimizer.train()

    return optimizer_name, optimizer_args, optimizer


def get_optimizer_train_eval_fn(optimizer: Optimizer, args: argparse.Namespace) -> Tuple[Callable, Callable]:
    if not is_schedulefree_optimizer(optimizer, args):
        # return dummy func
        return lambda: None, lambda: None

    # get train and eval functions from optimizer
    train_fn = optimizer.train
    eval_fn = optimizer.eval

    return train_fn, eval_fn


def is_schedulefree_optimizer(optimizer: Optimizer, args: argparse.Namespace) -> bool:
    return args.optimizer_type.lower().endswith("schedulefree".lower())  # or args.optimizer_schedulefree_wrapper


def get_dummy_scheduler(optimizer: Optimizer) -> Any:
    # dummy scheduler for schedulefree optimizer. supports only empty step(), get_last_lr() and optimizers.
    # this scheduler is used for logging only.
    # this isn't be wrapped by accelerator because of this class is not a subclass of torch.optim.lr_scheduler._LRScheduler
    class DummyScheduler:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

        def step(self):
            pass

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    return DummyScheduler(optimizer)


# Modified version of get_scheduler() function from diffusers.optimizer.get_scheduler
# Add some checking and features to the original function.


def get_scheduler_fix(args, optimizer: Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    # if schedulefree optimizer, return dummy scheduler
    if is_schedulefree_optimizer(optimizer, args):
        return get_dummy_scheduler(optimizer)

    name = args.lr_scheduler
    num_training_steps = args.max_train_steps * num_processes  # * args.gradient_accumulation_steps
    num_warmup_steps: Optional[int] = (
        int(args.lr_warmup_steps * num_training_steps) if isinstance(args.lr_warmup_steps, float) else args.lr_warmup_steps
    )
    num_decay_steps: Optional[int] = (
        int(args.lr_decay_steps * num_training_steps) if isinstance(args.lr_decay_steps, float) else args.lr_decay_steps
    )
    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
    num_cycles = args.lr_scheduler_num_cycles
    power = args.lr_scheduler_power
    timescale = args.lr_scheduler_timescale
    min_lr_ratio = args.lr_scheduler_min_lr_ratio

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
        for arg in args.lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    # using any lr_scheduler from other library
    if args.lr_scheduler_type:
        lr_scheduler_type = args.lr_scheduler_type
        logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:  # default to use torch.optim
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return wrap_check_needless_num_warmup_steps(lr_scheduler)

    if name.startswith("adafactor"):
        assert (
            type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        initial_lr = float(name.split(":")[1])
        # logger.info(f"adafactor scheduler init lr {initial_lr}")
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
        name = DiffusersSchedulerType(name)
        schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
        return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power, **lr_scheduler_kwargs
        )

    if name == SchedulerType.COSINE_WITH_MIN_LR:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles / 2,
            min_lr_rate=min_lr_ratio,
            **lr_scheduler_kwargs,
        )

    # these schedulers do not require `num_decay_steps`
    if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    # All other schedulers require `num_decay_steps`
    if num_decay_steps is None:
        raise ValueError(f"{name} requires `num_decay_steps`, please provide that argument.")
    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_decay_steps=num_decay_steps,
            num_cycles=num_cycles / 2,
            min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
            **lr_scheduler_kwargs,
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_decay_steps=num_decay_steps,
        **lr_scheduler_kwargs,
    )


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    # backward compatibility
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    # assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(",")])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert (
            len(args.resolution) == 2
        ), f"resolution must be 'size' or 'width,height' / resolution（解像度）は'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(",")])
        assert (
            len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            logger.warning(
                f"latents in npz is ignored when color_aug or random_crop is True / color_augまたはrandom_cropを有効にした場合、npzファイルのlatentsは無視されます"
            )


def prepare_accelerator(args: argparse.Namespace):
    """
    this function also prepares deepspeed plugin
    """

    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    # torch.compile のオプション。 NO の場合は torch.compile は使わない
    dynamo_backend = "NO"
    if args.torch_compile:
        dynamo_backend = args.dynamo_backend

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_backend=dynamo_backend,
        deepspeed_plugin=deepspeed_plugin,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def _load_target_model(args: argparse.Namespace, weight_dtype, device="cpu", unet_use_linear_projection_in_v2=False):
    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.path.realpath(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers
    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
            args.v2, name_or_path, device, unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2
        )
    else:
        # Diffusers model is loaded to CPU
        logger.info(f"load Diffusers pretrained models: {name_or_path}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        # TODO *.ckpt/*.safetensorsのv2と同じ形式にここで変換すると良さそう
        # logger.info(f"unet config: {unet.config}")
        original_unet = UNet2DConditionModel(
            unet.config.sample_size,
            unet.config.attention_head_dim,
            unet.config.cross_attention_dim,
            unet.config.use_linear_projection,
            unet.config.upcast_attention,
        )
        original_unet.load_state_dict(unet.state_dict())
        unet = original_unet
        logger.info("U-Net converted to original U-Net")

    # VAEを読み込む
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        logger.info("additional VAE loaded")

    return text_encoder, vae, unet, load_stable_diffusion_format


def load_target_model(args, weight_dtype, accelerator, unet_use_linear_projection_in_v2=False):
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(
                args,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2,
            )
            # work on low-ram device
            if args.lowram:
                text_encoder.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()
    return text_encoder, vae, unet, load_stable_diffusion_format


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def get_hidden_states(args: argparse.Namespace, input_ids, tokenizer, text_encoder, weight_dtype=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n,77
    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if args.clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-args.clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if args.max_token_length is not None:
        if args.v2:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2]  # <BOS> の後から 最後の前まで
                if i > 0:
                    for j in range(len(chunk)):
                        if input_ids[j, 1] == tokenizer.eos_token:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                            chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                states_list.append(
                    encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2]
                )  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
):
    r"""
    workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token
    If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

    Original code from CLIP's pooling function:

    \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
    \# take features from the eot embedding (eot_token is the highest number in each sequence)
    \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]
    """

    # input_ids: b*n,77
    # find index for EOS token

    # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
    # eos_token_index = torch.where(input_ids == eos_token_id)[1]
    # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # get hidden states for EOS token
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int,
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
    weight_dtype: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    unwrapped_text_encoder2 = text_encoder2 if accelerator is None else accelerator.unwrap_model(text_encoder2)
    pool2 = pool_workaround(unwrapped_text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def default_if_none(value, default):
    return default if value is None else value


def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name, step_no) + ext


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
    # save_every_n_steps=10, save_last_n_steps=30の場合、50step目には30step分残し、10step目を削除する
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
    )


def save_sd_model_on_epoch_end_or_stepwise_common(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(args, epoch_no)
    else:
        # 保存するか否かは呼び出し側で判断済み

        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch  # 例: 最初のepochの途中で保存したら0になる、SDモデルに保存される
        remove_no = get_remove_step_no(args, global_step)

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(args, ext, remove_no)

            remove_ckpt_file = os.path.join(args.output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                logger.info(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
        else:
            out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step))

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no))
            else:
                remove_out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no))

            if os.path.exists(remove_out_dir):
                logger.info(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    if args.save_state:
        if on_epoch_end:
            save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_and_remove_state_stepwise(args, accelerator, global_step)


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    if args.save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def save_sd_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver
    )


def save_sd_model_on_train_end_common(
    args: argparse.Namespace,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=True)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name, force_sync_upload=True)


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = torch.rand(1, device=latents.device) * args.noise_offset
        else:
            noise_offset = args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noise, noisy_latents, timesteps


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        alphas_cumprod = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps.cpu())
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full((b_size,), args.huber_c * args.huber_scale, device=timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor, target: torch.Tensor, loss_type: str, reduction: str, huber_c: Optional[torch.Tensor] = None
):
    """
    NOTE: if you're using the scheduled version, huber_c has to depend on the timesteps already
    """
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, 1, 1, 1)
        loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, 1, 1, 1)
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")
    names.append("text_encoder3")  # SD3

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    lrs = lr_scheduler.get_last_lr()

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_type.lower() == "Prodigy".lower():
            logs["lr/d*lr/" + name] = (
                lr_scheduler.optimizers[-1].param_groups[lr_index]["d"] * lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )


# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def get_my_scheduler(
    *,
    sample_sampler: str,
    v_parameterization: bool,
):
    sched_init_args = {}
    if sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sample_sampler == "lms" or sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sample_sampler == "euler" or sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sample_sampler == "euler_a" or sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sample_sampler == "dpmsolver" or sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sample_sampler
    elif sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sample_sampler == "dpm_2" or sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sample_sampler == "dpm_2_a" or sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


def sample_images(*args, **kwargs):
    return sample_images_common(StableDiffusionLongPromptWeightingPipeline, *args, **kwargs)


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析エラー: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> List[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            from library.train_util import line_to_prompt_dict

            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def sample_images_common(
    pipe_class,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    steps: int,
    device,
    vae,
    tokenizer,
    text_encoder,
    unet,
    prompt_replacement=None,
    controlnet=None,
):
    """
    StableDiffusionLongPromptWeightingPipelineの改造版を使うようにしたので、clip skipおよびプロンプトの重みづけに対応した
    TODO Use strategies here
    """

    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            # sample_every_n_steps は無視する
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    org_vae_device = vae.device  # CPUにいるはず
    vae.to(distributed_state.device)  # distributed_state.device is same as accelerator.device

    # unwrap unet and text_encoder(s)
    unet = accelerator.unwrap_model(unet)
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)

    # read prompts
    if args.sample_prompts.endswith(".txt"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif args.sample_prompts.endswith(".toml"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif args.sample_prompts.endswith(".json"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    default_scheduler = get_my_scheduler(sample_sampler=args.sample_sampler, v_parameterization=args.v_parameterization)

    pipeline = pipe_class(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=default_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=args.clip_skip,
    )
    pipeline.to(distributed_state.device)
    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
                )
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
                    )

    # clear pipeline and cache to reduce vram usage
    del pipeline

    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    pipeline: Union[StableDiffusionLongPromptWeightingPipeline, SdxlStableDiffusionLongPromptWeightingPipeline],
    save_dir,
    prompt_dict,
    epoch,
    steps,
    prompt_replacement,
    controlnet=None,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()

    scheduler = get_my_scheduler(
        sample_sampler=sampler_name,
        v_parameterization=args.v_parameterization,
    )
    pipeline.scheduler = scheduler

    if controlnet_image is not None:
        controlnet_image = Image.open(controlnet_image).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)

    height = max(64, height - height % 8)  # round to divisible by 8
    width = max(64, width - width % 8)  # round to divisible by 8
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")
    with accelerator.autocast():
        latents = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            guidance_scale=scale,
            negative_prompt=negative_prompt,
            controlnet=controlnet,
            controlnet_image=controlnet_image,
        )

    if torch.cuda.is_available():
        with torch.cuda.device(torch.cuda.current_device()):
            torch.cuda.empty_cache()

    image = pipeline.latents_to_image(latents)[0]

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # send images to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")

        import wandb

        # not to commit images to avoid inconsistency between training and logging steps
        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)  # positive prompt as a caption


def init_trackers(accelerator: Accelerator, args: argparse.Namespace, default_tracker_name: str):
    """
    Initialize experiment trackers with tracker specific behaviors
    """
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            default_tracker_name if args.log_tracker_name is None else args.log_tracker_name,
            config=get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            import wandb

            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

            # Define specific metrics to handle validation and epochs "steps"
            wandb_tracker.define_metric("epoch", hidden=True)
            wandb_tracker.define_metric("val_step", hidden=True)

            wandb_tracker.define_metric("global_step", hidden=True)


# endregion


# region 前処理用


class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor_pil = transforms.functional.pil_to_tensor(image)
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor_pil, img_path)


# endregion


# collate_fn用 epoch,stepはmultiprocessing.Value
class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        losses = len(self.loss_list)
        if losses == 0:
            return 0
        return self.loss_total / losses

