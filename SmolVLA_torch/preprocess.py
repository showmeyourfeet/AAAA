import torch
import torch.nn.functional as F


def pad_tensor(tensor, target_len, pad_value=0):
    """
    Pad tensor on sequence dimension to target_len.
    Supports shapes (B, L, ...) or (B, L).
    """
    if tensor.shape[1] >= target_len:
        return tensor
    pad_shape = (tensor.shape[0], target_len, *tensor.shape[2:])
    out = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    out[:, : tensor.shape[1]] = tensor
    return out


def pad_vector(vec, target_dim, pad_value=0):
    """
    Pad feature dimension to target_dim.
    Supports shapes (B, F) or (B, T, F).
    """
    if vec.shape[-1] >= target_dim:
        return vec
    new_shape = (*vec.shape[:-1], target_dim)
    out = torch.full(new_shape, pad_value, dtype=vec.dtype, device=vec.device)
    out[..., : vec.shape[-1]] = vec
    return out


def resize_with_pad(img, width, height, pad_value=-1):
    """
    Resize preserving aspect ratio, then pad to (height, width).
    Expects img shape (B, C, H, W).
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, got {img.shape}")

    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)
    resized = F.interpolate(img, size=(resized_h, resized_w), mode="bilinear", align_corners=False)

    pad_h = max(0, height - resized_h)
    pad_w = max(0, width - resized_w)
    # pad format: (pad_left, pad_right, pad_top, pad_bottom)
    padded = F.pad(resized, (0, pad_w, 0, pad_h), value=pad_value)
    return padded


def normalize_to_minus1_1(img):
    """
    Normalize image from [0,1] to [-1,1].
    """
    return img * 2.0 - 1.0

