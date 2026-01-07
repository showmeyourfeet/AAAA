import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SmolVLADataset(Dataset):
    """
    SmolVLA 专用 dataset，适配当前 stage/run 数据格式。

    目录结构示例:
      root_dir/
        stage1/
          run_001/
            data.txt
            left_frame/*.jpg
            right_frame/*.jpg
        stage2/
          ...

    每个样本:
      - 从某个 stageX/runY 中随机抽取一个 start_ts
      - 取该时刻的 qpos 作为 state
      - 取后续最多 chunk_size 步动作作为 actions，并在时间维度 pad 到 chunk_size
      - 读取对应时刻的多相机图像，按 [num_cams, C, H, W] 返回
      - 根据 stage_idx 查找对应的自然语言指令文本
    """

    def __init__(
        self,
        root_dir: str,
        camera_names: Sequence[str],
        chunk_size: int,
        instruction_json: str,
        image_size: int = 224,
        use_augmentation: bool = False,
        training: bool = True,
        use_state: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.camera_names = list(camera_names)
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.training = training
        self.use_state = use_state

        self.to_tensor = transforms.ToTensor()

        # 收集所有 runs: (stage_idx, run_path)
        self.runs: List[Tuple[int, str]] = []
        for stage_dir in sorted(os.listdir(root_dir)):
            stage_path = os.path.join(root_dir, stage_dir)
            if not (stage_dir.startswith("stage") and os.path.isdir(stage_path)):
                continue
            try:
                stage_idx = int(stage_dir[5:]) - 1  # stage1 -> 0
            except ValueError:
                continue
            for run_dir in sorted(os.listdir(stage_path)):
                run_path = os.path.join(stage_path, run_dir)
                if run_dir.startswith("run") and os.path.isdir(run_path):
                    self.runs.append((stage_idx, run_path))

        if len(self.runs) == 0:
            print(f"[SmolVLADataset] No runs found under {root_dir}")

        # 读取 stage -> 原始文本 指令
        self.stage_to_text: Dict[int, str] = {}
        if os.path.exists(instruction_json):
            with open(instruction_json, "r", encoding="utf-8") as f:
                inst = json.load(f)
            for entry in inst.get("task_instructions", []):
                stage = int(entry.get("stage"))
                text = entry.get("text", "")
                # 保存为 0-indexed 的 stage_idx
                self.stage_to_text[stage - 1] = text
        else:
            print(f"[SmolVLADataset] instruction_json not found: {instruction_json}")

    def __len__(self) -> int:
        return len(self.runs)

    @staticmethod
    def _parse_run(run_path: str):
        """
        解析 data.txt，返回 actions 和 qpos 序列。
        """
        data_file = os.path.join(run_path, "data.txt")
        if not os.path.exists(data_file):
            return None, None

        with open(data_file, "r", encoding="utf-8") as f:
            content = f.read()

        actions: List[np.ndarray] = []
        qpos: List[np.ndarray] = []

        frame_blocks = re.split(r"Frame_\d+:", content)[1:]
        for block in frame_blocks:
            m_act = re.search(r"action:\s*\[([^\]]+)\]", block)
            m_lr = re.search(r"lrstate\s*:\s*\[([^\]]+)\]", block)
            if m_act and m_lr:
                try:
                    act = np.array([float(x.strip()) for x in m_act.group(1).split(",")], dtype=np.float32)
                    st = np.array([float(x.strip()) for x in m_lr.group(1).split(",")], dtype=np.float32)
                    actions.append(act)
                    qpos.append(st)
                except Exception:
                    continue

        if not actions:
            return None, None
        return np.stack(actions), np.stack(qpos)

    def __getitem__(self, index: int):
        stage_idx, run_path = self.runs[index]

        actions_np, qpos_np = self._parse_run(run_path)
        if actions_np is None:
            raise RuntimeError(f"No valid actions in {run_path}")

        T, act_dim = actions_np.shape
        _, state_dim = qpos_np.shape

        # 随机起始时间步
        start_ts = np.random.randint(0, T)
        end_ts = min(start_ts + self.chunk_size - 1, T - 1)
        seq_len = end_ts - start_ts + 1

        # 状态：使用起始时刻的 qpos
        if self.use_state:
            state = torch.from_numpy(qpos_np[start_ts]).float()
        else:
            state = torch.zeros(state_dim, dtype=torch.float32)

        # 动作序列 + padding 到 chunk_size
        action_slice = actions_np[start_ts : end_ts + 1]  # [seq_len, act_dim]
        padded = np.zeros((self.chunk_size, act_dim), dtype=np.float32)
        padded[:seq_len] = action_slice
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[seq_len:] = True

        actions = torch.from_numpy(padded).float()  # [T, act_dim]
        is_pad = torch.from_numpy(is_pad).bool()

        # 图像：当前时间步的多相机图像
        images_np = []
        for cam in self.camera_names:
            cam_dir = os.path.join(run_path, cam)
            files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_full.jpg")])
            if not files:
                raise FileNotFoundError(f"No *_full.jpg files found under {cam_dir}")
            idx_img = min(start_ts, len(files) - 1)
            img = Image.open(os.path.join(cam_dir, files[idx_img])).convert("RGB")
            img_np = np.array(img)
            images_np.append(img_np)

        # 如需要可加数据增强；这里仅转换为 tensor，后续由 SmolVLA 自己做 resize/normalize
        images_t = [self.to_tensor(x) for x in images_np]  # list of [C,H,W]
        images = torch.stack(images_t, dim=0)  # [num_cams, C, H, W]

        # 取 stage 对应的语言文本（如果不存在则为空串）
        lang_text = self.stage_to_text.get(stage_idx, "")
        # 官方会在结尾加一个换行
        if lang_text and not lang_text.endswith("\n"):
            lang_text = lang_text + "\n"

        return images, state, actions, is_pad, lang_text


def smolvla_collate_fn(batch):
    """
    简单的 collate：将 list[tuple] 打包成 batch tensor 和语言文本 list。
    返回:
      images: [B, num_cams, C, H, W]
      states: [B, state_dim]
      actions: [B, T, act_dim]
      is_pad: [B, T]
      texts:  List[str]
    """
    imgs, states, acts, pads, texts = zip(*batch)
    images = torch.stack(imgs, dim=0)
    states = torch.stack(states, dim=0)
    actions = torch.stack(acts, dim=0)
    is_pad = torch.stack(pads, dim=0)
    return images, states, actions, is_pad, list(texts)


