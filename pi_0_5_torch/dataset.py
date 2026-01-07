import json
import os
import re
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Pi05Dataset(Dataset):
    """
    Dataset for Pi0.5, adapted to the same on-disk layout as `SmolVLADataset`.

    Directory structure example (aligned with SmolVLA_torch/dataset.py):
      root_dir/
        stage1/
          run_001/
            data.txt
            <camera_name>/*.jpg
        stage2/
          ...

    Each sample:
      - randomly sample a start_ts from a stageX/runY
      - use the qpos at that time as state
      - take the next `chunk_size` steps of actions as actions, and pad to `chunk_size`
      - read the current camera image at that time (primary_camera)
      - according to stage_idx, find the corresponding natural language instruction text

    Output fields (aligned with the subsequent preprocess):
      - image: PIL.Image.Image
      - state: torch.FloatTensor [state_dim]
      - actions: torch.FloatTensor [chunk_size, act_dim]
      - action_is_pad: torch.BoolTensor [chunk_size]
      - text: str
      
    **Hierarchical Policy Fields (Pi0.5 Paper)**:
      - high_level_task: str  # High-level task description (e.g., "clean the bedroom")
      - subtask: str          # Subtask label (e.g., "pick up pillow")
    """

    def __init__(
        self,
        root_dir: str,
        primary_camera: str,
        chunk_size: int,
        instruction_json: Optional[str] = None,
        use_state: bool = True,
        enable_hierarchical: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.primary_camera = primary_camera
        self.chunk_size = chunk_size
        self.use_state = use_state
        self.enable_hierarchical = enable_hierarchical

        # collect all runs: (stage_idx, run_path)
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
            print(f"[Pi05Dataset] No runs found under {root_dir}")

        # Read instruction data
        # For hierarchical mode (Pi0.5), we support:
        # - "high_level_task": the overall task (e.g., "clean the bedroom")
        # - "subtask" / "text": the specific subtask (e.g., "pick up pillow")
        self.stage_to_text: Dict[int, str] = {}
        self.stage_to_high_level_task: Dict[int, str] = {}
        self.stage_to_subtask: Dict[int, str] = {}
        self.global_high_level_task: str = ""
        
        if instruction_json is not None and os.path.exists(instruction_json):
            with open(instruction_json, "r", encoding="utf-8") as f:
                inst = json.load(f)
            
            # Global high-level task (applies to all stages)
            self.global_high_level_task = inst.get("high_level_task", "")
            
            for entry in inst.get("task_instructions", []):
                try:
                    stage = int(entry.get("stage"))
                except (TypeError, ValueError):
                    continue
                
                # Standard text field (used as subtask if hierarchical enabled)
                text = entry.get("text", "")
                self.stage_to_text[stage - 1] = text
                
                # Hierarchical fields (Pi0.5 specific)
                high_level = entry.get("high_level_task", self.global_high_level_task)
                subtask = entry.get("subtask", text)  # Default to text if no subtask
                
                self.stage_to_high_level_task[stage - 1] = high_level
                self.stage_to_subtask[stage - 1] = subtask
        else:
            if instruction_json is not None:
                print(f"[Pi05Dataset] instruction_json not found: {instruction_json}")

    def __len__(self) -> int:
        return len(self.runs)

    @staticmethod
    def _parse_run(run_path: str):
        """
        parse the data.txt, return the actions and qpos sequence.

        try to keep the same as SmolVLADataset._parse_run, to compatible with the existing data preprocessing script.
        """
        data_file = os.path.join(run_path, "data.txt")
        if not os.path.exists(data_file):
            return None, None

        with open(data_file, "r", encoding="utf-8") as f:
            content = f.read()

        actions: List[np.ndarray] = []
        qpos: List[np.ndarray] = []

        frame_blocks = re.split(r"Frame_\\d+:", content)[1:]
        for block in frame_blocks:
            m_act = re.search(r"action:\\s*\\[([^\\]]+)\\]", block)
            m_lr = re.search(r"lrstate\\s*:\\s*\\[([^\\]]+)\\]", block)
            if m_act and m_lr:
                try:
                    act = np.array(
                        [float(x.strip()) for x in m_act.group(1).split(",")],
                        dtype=np.float32,
                    )
                    st = np.array(
                        [float(x.strip()) for x in m_lr.group(1).split(",")],
                        dtype=np.float32,
                    )
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
            raise RuntimeError(f"[Pi05Dataset] No valid actions in {run_path}")

        T, act_dim = actions_np.shape
        _, state_dim = qpos_np.shape

        # randomly sample the start time step
        start_ts = np.random.randint(0, T)
        end_ts = min(start_ts + self.chunk_size - 1, T - 1)
        seq_len = end_ts - start_ts + 1

        # low-dimensional state: use the qpos at the start time step
        if self.use_state:
            state = torch.from_numpy(qpos_np[start_ts]).float()  # [state_dim]
        else:
            state = torch.zeros(state_dim, dtype=torch.float32)

        # action sequence + padding to chunk_size
        action_slice = actions_np[start_ts : end_ts + 1]  # [seq_len, act_dim]
        padded = np.zeros((self.chunk_size, act_dim), dtype=np.float32)
        padded[:seq_len] = action_slice
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[seq_len:] = True

        actions = torch.from_numpy(padded).float()  # [chunk_size, act_dim]
        action_is_pad = torch.from_numpy(is_pad).bool()

        # image: the single image at the current time step (primary_camera)
        cam_dir = os.path.join(run_path, self.primary_camera)
        if not os.path.exists(cam_dir):
            raise FileNotFoundError(
                f"[Pi05Dataset] Camera directory not found: {cam_dir}"
            )

        files = sorted(
            [f for f in os.listdir(cam_dir) if f.lower().endswith((".jpg", ".png"))]
        )
        if not files:
            raise FileNotFoundError(
                f"[Pi05Dataset] No image files found under {cam_dir}"
            )

        idx_img = min(start_ts, len(files) - 1)
        img_path = os.path.join(cam_dir, files[idx_img])
        image = Image.open(img_path).convert("RGB")

        # text instruction
        lang_text = self.stage_to_text.get(stage_idx, "")
        if lang_text and not lang_text.endswith("\\n"):
            lang_text = lang_text + "\\n"

        result = {
            "image": image,
            "state": state,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "text": lang_text,
        }
        
        # Add hierarchical fields if enabled (Pi0.5 paper)
        if self.enable_hierarchical:
            # High-level task (e.g., "clean the bedroom")
            high_level_task = self.stage_to_high_level_task.get(
                stage_idx, self.global_high_level_task
            )
            if high_level_task and not high_level_task.endswith("\\n"):
                high_level_task = high_level_task + "\\n"
            
            # Subtask (e.g., "pick up pillow")
            subtask = self.stage_to_subtask.get(stage_idx, lang_text)
            if subtask and not subtask.endswith("\\n"):
                subtask = subtask + "\\n"
            
            result["high_level_task"] = high_level_task
            result["subtask"] = subtask
        
        return result


def pi05_collate_fn(batch):
    """
    pack the samples of Pi05Dataset into a batch.

    return:
      - images: List[PIL.Image.Image], length is B
      - states: torch.FloatTensor [B, state_dim], length is B
      - actions: torch.FloatTensor [B, chunk_size, act_dim]
      - action_is_pad: torch.BoolTensor [B, chunk_size]
      - texts: List[str]
      
    Hierarchical fields (if present in batch):
      - high_level_tasks: List[str]  # High-level task descriptions
      - subtasks: List[str]          # Subtask labels
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["actions"] for item in batch], dim=0)
    action_is_pad = torch.stack([item["action_is_pad"] for item in batch], dim=0)

    result = {
        "images": images,
        "states": states,
        "actions": actions,
        "action_is_pad": action_is_pad,
        "texts": texts,
    }
    
    # Add hierarchical fields if present
    if "high_level_task" in batch[0]:
        result["high_level_tasks"] = [item["high_level_task"] for item in batch]
    if "subtask" in batch[0]:
        result["subtasks"] = [item["subtask"] for item in batch]
    
    return result


# ============================================================================
# Pi05HierarchicalDataset - Explicit hierarchical dataset for Pi0.5 training
# ============================================================================


class Pi05HierarchicalDataset(Dataset):
    """
    Dataset specifically designed for Pi0.5 hierarchical policy training.
    
    This dataset provides separate fields for:
    - High-level task prompts (e.g., "clean the bedroom")
    - Subtask labels (e.g., "pick up pillow", "adjust blanket")
    - Low-level actions
    
    Supports multiple training modes:
    1. Action-only: Standard flow matching training
    2. Subtask prediction: Train to predict subtask from high-level task
    3. Joint training: Both subtask prediction and action generation
    
    Directory structure:
      root_dir/
        task1/  # High-level task
          subtask1/
            run_001/
              data.txt
              <camera_name>/*.jpg
          subtask2/
            ...
        task2/
          ...
          
    Or simpler structure:
      root_dir/
        stage1/
          run_001/
            ...
    """
    
    def __init__(
        self,
        root_dir: str,
        primary_camera: str,
        chunk_size: int,
        instruction_json: Optional[str] = None,
        use_state: bool = True,
        training_mode: str = "joint",  # "action_only", "subtask_only", "joint"
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.primary_camera = primary_camera
        self.chunk_size = chunk_size
        self.use_state = use_state
        self.training_mode = training_mode
        
        # Data storage: (high_level_task, subtask, run_path)
        self.samples: List[Tuple[str, str, str]] = []
        
        # Load instruction mapping
        self.instruction_data: Dict = {}
        if instruction_json is not None and os.path.exists(instruction_json):
            with open(instruction_json, "r", encoding="utf-8") as f:
                self.instruction_data = json.load(f)
        
        self._scan_directory()
        
        if len(self.samples) == 0:
            print(f"[Pi05HierarchicalDataset] No samples found under {root_dir}")
    
    def _scan_directory(self):
        """Scan directory for hierarchical structure or fall back to stage structure."""
        # Try hierarchical structure first (task/subtask/run)
        hierarchical_found = False
        
        for task_dir in sorted(os.listdir(self.root_dir)):
            task_path = os.path.join(self.root_dir, task_dir)
            if not os.path.isdir(task_path):
                continue
            
            # Check if this is a task directory (contains subtask directories)
            for subtask_dir in sorted(os.listdir(task_path)):
                subtask_path = os.path.join(task_path, subtask_dir)
                if not os.path.isdir(subtask_path):
                    continue
                
                # Check for run directories
                for run_dir in sorted(os.listdir(subtask_path)):
                    run_path = os.path.join(subtask_path, run_dir)
                    if run_dir.startswith("run") and os.path.isdir(run_path):
                        hierarchical_found = True
                        
                        # Get text labels from instruction data or directory names
                        high_level = self._get_task_text(task_dir)
                        subtask = self._get_subtask_text(task_dir, subtask_dir)
                        
                        self.samples.append((high_level, subtask, run_path))
        
        # Fall back to stage structure
        if not hierarchical_found:
            global_high_level = self.instruction_data.get("high_level_task", "perform the task")
            
            for stage_dir in sorted(os.listdir(self.root_dir)):
                stage_path = os.path.join(self.root_dir, stage_dir)
                if not (stage_dir.startswith("stage") and os.path.isdir(stage_path)):
                    continue
                
                try:
                    stage_idx = int(stage_dir[5:]) - 1
                except ValueError:
                    continue
                
                for run_dir in sorted(os.listdir(stage_path)):
                    run_path = os.path.join(stage_path, run_dir)
                    if run_dir.startswith("run") and os.path.isdir(run_path):
                        subtask = self._get_stage_text(stage_idx)
                        self.samples.append((global_high_level, subtask, run_path))
    
    def _get_task_text(self, task_dir: str) -> str:
        """Get high-level task text from instruction data or directory name."""
        tasks = self.instruction_data.get("tasks", {})
        if task_dir in tasks:
            return tasks[task_dir].get("text", task_dir.replace("_", " "))
        return task_dir.replace("_", " ")
    
    def _get_subtask_text(self, task_dir: str, subtask_dir: str) -> str:
        """Get subtask text from instruction data or directory name."""
        tasks = self.instruction_data.get("tasks", {})
        if task_dir in tasks:
            subtasks = tasks[task_dir].get("subtasks", {})
            if subtask_dir in subtasks:
                return subtasks[subtask_dir].get("text", subtask_dir.replace("_", " "))
        return subtask_dir.replace("_", " ")
    
    def _get_stage_text(self, stage_idx: int) -> str:
        """Get stage text from instruction data."""
        for entry in self.instruction_data.get("task_instructions", []):
            try:
                stage = int(entry.get("stage"))
                if stage - 1 == stage_idx:
                    return entry.get("subtask", entry.get("text", ""))
            except (TypeError, ValueError):
                continue
        return ""
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        high_level_task, subtask, run_path = self.samples[index]
        
        # Parse run data (same as Pi05Dataset)
        actions_np, qpos_np = Pi05Dataset._parse_run(run_path)
        if actions_np is None:
            raise RuntimeError(f"[Pi05HierarchicalDataset] No valid actions in {run_path}")
        
        T, act_dim = actions_np.shape
        _, state_dim = qpos_np.shape
        
        # Random sampling
        start_ts = np.random.randint(0, T)
        end_ts = min(start_ts + self.chunk_size - 1, T - 1)
        seq_len = end_ts - start_ts + 1
        
        # State
        if self.use_state:
            state = torch.from_numpy(qpos_np[start_ts]).float()
        else:
            state = torch.zeros(state_dim, dtype=torch.float32)
        
        # Actions
        action_slice = actions_np[start_ts : end_ts + 1]
        padded = np.zeros((self.chunk_size, act_dim), dtype=np.float32)
        padded[:seq_len] = action_slice
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[seq_len:] = True
        
        actions = torch.from_numpy(padded).float()
        action_is_pad = torch.from_numpy(is_pad).bool()
        
        # Image
        cam_dir = os.path.join(run_path, self.primary_camera)
        if not os.path.exists(cam_dir):
            raise FileNotFoundError(f"Camera directory not found: {cam_dir}")
        
        files = sorted([f for f in os.listdir(cam_dir) if f.lower().endswith((".jpg", ".png"))])
        if not files:
            raise FileNotFoundError(f"No image files found under {cam_dir}")
        
        idx_img = min(start_ts, len(files) - 1)
        img_path = os.path.join(cam_dir, files[idx_img])
        image = Image.open(img_path).convert("RGB")
        
        # Format text fields
        if high_level_task and not high_level_task.endswith("\\n"):
            high_level_task = high_level_task + "\\n"
        if subtask and not subtask.endswith("\\n"):
            subtask = subtask + "\\n"
        
        return {
            "image": image,
            "state": state,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "text": subtask,  # For backward compatibility
            "high_level_task": high_level_task,
            "subtask": subtask,
        }


def pi05_hierarchical_collate_fn(batch):
    """Collate function for Pi05HierarchicalDataset."""
    return pi05_collate_fn(batch)  # Reuse the standard collate function


