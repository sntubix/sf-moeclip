import os
import json
import math
import random
import torch
from torch.utils.data import Dataset
from utils import AddGaussianNoise
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from .constants import ASPECT_PRESERVING_DATASETS, CLASS_NAMES, DATA_PATH, DOMAINS


CLIP_MEAN_RGB = (123, 117, 104)

# Per-dataset photometric correction applied at inference only.
# PVF1 RGB images are noticeably dark, so we lift the dark midtones with a
# gamma < 1 then apply a small linear brightness multiply. Gamma does most of
# the lifting, so the multiply is kept modest to avoid clipping highlights.
# Factor 1.0 / gamma 1.0 = no change. Order: gamma first, then brightness.
TEST_BRIGHTNESS_FACTOR = {
    "PVF1": 1.2,
}
TEST_GAMMA = {
    "PVF1": 0.7,
}


class LetterboxResize:
    """Aspect-preserving resize + center-pad to a (target, target) square.

    Used for non-square datasets (PVF1, PVF1UV) where stretching to a square
    would distort defects. Pad value defaults to the CLIP normalization mean
    so padded borders become ~0 after Normalize and stay neutral to the model.
    """

    def __init__(self, target_size, interpolation, fill=0):
        self.target_size = target_size
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), self.interpolation)
        canvas = Image.new(img.mode, (self.target_size, self.target_size), self.fill)
        canvas.paste(
            img,
            ((self.target_size - new_w) // 2, (self.target_size - new_h) // 2),
        )
        return canvas


def _make_resize(img_size, interpolation, aspect_preserving, fill):
    if aspect_preserving:
        return LetterboxResize(img_size, interpolation, fill=fill)
    return transforms.Resize((img_size, img_size), interpolation)


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        text: bool = False,
        aspect_preserving: bool = False,
        strong_aug: bool = False,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.text = text
        self.strong_aug = strong_aug
        self.meta = []
        self.full_shot = "full-shot" in meta_path
        with open(meta_path, "r") as f:
            for line in f:
                self.meta.append(json.loads(line))

        self.transforms_list = [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        transform_x = []
        if not text:
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.7)
            )
        if strong_aug:
            # Photometric + structural augs targeted at low-color modalities
            # (UV, electroluminescence). Acquisition exposure varies a lot lab
            # to lab, so we widen brightness/contrast and add intensity
            # normalizers (autocontrast/equalize) plus sharpness/blur to make
            # the defect signature invariant to capture conditions.
            transform_x.extend([
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.6, contrast=0.6)], p=0.8
                ),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
                ),
            ])
        self.transform_x = transforms.Compose(
            transform_x
            + [
                _make_resize(img_size, Image.BICUBIC, aspect_preserving, fill=CLIP_MEAN_RGB),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        # Applied to the image only, after the joint image+mask geometric
        # augs in __getitem__. Erasing the mask would falsify supervision.
        self.post_transform = (
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0
            )
            if strong_aug
            else None
        )
        self.transform_mask = transforms.Compose(
            [
                _make_resize(img_size, Image.NEAREST, aspect_preserving, fill=0),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        random_transform = transforms.Compose(self.transforms_list)
        transform_tensor = torch.cat([img, mask], dim=0)
        assert transform_tensor.shape[0] == 4
        transform_tensor = random_transform(transform_tensor)
        img = transform_tensor[0:3, :, :]
        mask = transform_tensor[3:4, :, :]

        if self.post_transform is not None:
            img = self.post_transform(img)

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        class_name: str,
        logger=None,
        aspect_preserving: bool = False,
        brightness_factor: float = 1.0,
        gamma: float = 1.0,
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                m = json.loads(line.strip())
                if m["class_name"] == class_name:
                    self.meta.append(m)

        transform_x = []
        if gamma != 1.0:
            transform_x.append(
                transforms.Lambda(lambda img: TF.adjust_gamma(img, gamma))
            )
        if brightness_factor != 1.0:
            transform_x.append(
                transforms.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor))
            )
        transform_x += [
            _make_resize(img_size, Image.BICUBIC, aspect_preserving, fill=CLIP_MEAN_RGB),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
        self.transform_x = transforms.Compose(transform_x)
        self.transform_mask = transforms.Compose(
            [
                _make_resize(img_size, Image.NEAREST, aspect_preserving, fill=0),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        if meta["label"]:

            if "mask_path" in meta and meta["mask_path"]:

                mask_path = os.path.join(self.data_path, meta["mask_path"])
                mask = Image.open(mask_path).convert("L")
                mask = self.transform_mask(mask)
                mask = (mask != 0).float()
            else:

                mask = torch.zeros([1, self.img_size, self.img_size])

        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


def get_dataset(
    dataset_name: str,
    img_size: int,
    training_mode: str,
    shot: int = -1,
    stage: str = "train",
    logger=None,
    tiled: bool = False,
    meta_file: str = None,
    strong_aug: bool = False,
):
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )

    # Tiled inputs are already square min(w,h) crops, so plain Resize to
    # (img_size, img_size) is correct and letterboxing would only re-pad them.
    aspect_preserving = (dataset_name in ASPECT_PRESERVING_DATASETS) and not tiled
    full_shot_name = "full-shot-tiled.jsonl" if tiled else "full-shot.jsonl"

    if stage == "train":
        if meta_file:
            meta_path = (
                meta_file
                if os.path.isabs(meta_file) or os.path.exists(meta_file)
                else os.path.join("./dataset/metadata", dataset_name, meta_file)
            )
        elif training_mode == "few_shot":
            assert shot > 0, "shot should be positive"
            shot_name = f"{shot}-shot-tiled.jsonl" if tiled else f"{shot}-shot.jsonl"
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, shot_name
            )
        else:
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, full_shot_name
            )

        data_path = DATA_PATH[dataset_name.split("-")[0]]
        text_dataset = BaseDataset(
            data_path, meta_path, img_size, text=True, aspect_preserving=aspect_preserving
        )
        image_dataset = BaseDataset(
            data_path,
            meta_path,
            img_size,
            text=True,
            aspect_preserving=aspect_preserving,
            strong_aug=strong_aug,
        )
        return text_dataset, image_dataset
    elif stage == "test":
        if meta_file:
            meta_path = (
                meta_file
                if os.path.isabs(meta_file) or os.path.exists(meta_file)
                else os.path.join("./dataset/metadata", dataset_name, meta_file)
            )
        else:
            meta_path = os.path.join("./dataset/metadata", dataset_name, full_shot_name)
        class_names = CLASS_NAMES[dataset_name]
        brightness_factor = TEST_BRIGHTNESS_FACTOR.get(dataset_name, 1.0)
        gamma = TEST_GAMMA.get(dataset_name, 1.0)
        if logger is not None and (brightness_factor != 1.0 or gamma != 1.0):
            logger.info(
                "Applying inference-time gamma %.2f + brightness factor %.2f to %s",
                gamma,
                brightness_factor,
                dataset_name,
            )
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=logger,
                aspect_preserving=aspect_preserving,
                brightness_factor=brightness_factor,
                gamma=gamma,
            )
            datasets[class_name] = image_dataset
        return datasets
    elif stage == "visualize":
        class_names = CLASS_NAMES[dataset_name]
        meta_path = os.path.join("./dataset/metadata", dataset_name, full_shot_name)
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=None,
                aspect_preserving=aspect_preserving,
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")