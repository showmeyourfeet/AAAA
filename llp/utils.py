"""Utility functions for ACT."""

import json
import random
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np
import torch


def compute_dict_mean(epoch_dicts: Sequence[Dict[str, torch.Tensor]]):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d: Dict[str, torch.Tensor]):
    return {k: v.detach().cpu() for k, v in d.items()}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def is_multi_gpu_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("model.module.") for k in state_dict.keys())


def initialize_model_and_tokenizer(encoder: str):
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        DistilBertModel,
        DistilBertTokenizer,
    )

    if encoder == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif encoder == "clip":
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    else:
        raise ValueError("Unknown encoder type. Please use 'distilbert' or 'clip'.")
    return tokenizer, model


def encode_text(text: Iterable[str] | str, encoder: str, tokenizer, model):
    if isinstance(text, str):
        batch = [text]
    else:
        batch = list(text)
        if len(batch) == 0:
            raise ValueError("encode_text received empty text iterable")
    if encoder == "distilbert":
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().tolist()
    if encoder == "clip":
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=77,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().tolist()
    raise ValueError(f"Unsupported encoder: {encoder}")


def crop_resize(
    image: np.ndarray,
    crop_h: int = 240,
    crop_w: int = 320,
    resize_h: int = 480,
    resize_w: int = 640,
    resize: bool = True,
) -> np.ndarray:
    h, w, _ = image.shape
    y1 = h - crop_h - 20
    x1 = (w - crop_w) // 2
    cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    if resize:
        return cv2.resize(cropped, (resize_w, resize_h))
    return cropped


def random_crop(image: np.ndarray, crop_percentage: float = 0.95) -> np.ndarray:
    """
    Randomly crop an image by a percentage and resize back to original size.
    image: H x W x C (numpy, uint8 or float)
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * crop_percentage), int(w * crop_percentage)
    if new_h <= 0 or new_w <= 0:
        return image
    top = random.randint(0, max(0, h - new_h))
    left = random.randint(0, max(0, w - new_w))
    cropped = image[top : top + new_h, left : left + new_w, ...]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
