# Pi0.5 上层策略训练完整指南

本文档详细说明 Pi0.5 上层策略（High-Level Policy）的数据格式、输入输出、标签和损失计算。

## 1. 数据集格式

### 1.1 目录结构

支持两种目录结构：

**方式 A：层级结构（推荐）**
```
root_dir/
  task1/              # 高层任务（如 "clean the bedroom"）
    subtask1/        # 子任务（如 "pick up pillow"）
      run_001/
        actions.npy      # [T, action_dim]
        qpos.npy         # [T, state_dim]
        <camera_name>/   # 图像目录
          *.jpg
      run_002/
        ...
    subtask2/
      ...
  task2/
    ...
```

**方式 B：Stage 结构（兼容）**
```
root_dir/
  stage1/            # 阶段1
    run_001/
      actions.npy
      qpos.npy
      <camera_name>/*.jpg
    run_002/
      ...
  stage2/
    ...
```

### 1.2 指令 JSON 文件（`instruction_json`）

```json
{
  "high_level_task": "clean the bedroom",  // 全局高层任务（可选）
  "task_instructions": [
    {
      "stage": 1,
      "high_level_task": "clean the bedroom",
      "subtask": "pick up pillow",
      "text": "original low-level instruction"  // 可选
    },
    {
      "stage": 2,
      "high_level_task": "clean the bedroom",
      "subtask": "adjust blanket",
      "text": "..."
    }
  ],
  "tasks": {  // 可选：任务映射
    "task1": {
      "text": "clean the bedroom",
      "subtasks": {
        "subtask1": {"text": "pick up pillow"},
        "subtask2": {"text": "adjust blanket"}
      }
    }
  }
}
```

### 1.3 数据字段

每个样本包含：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `image` | PIL.Image | - | RGB 图像（单帧） |
| `state` | Tensor | `[state_dim]` | 机器人本体状态（关节角度、gripper pose、base velocity 等） |
| `actions` | Tensor | `[chunk_size, action_dim]` | 动作序列（chunk_size=50，约1秒） |
| `action_is_pad` | Tensor | `[chunk_size]` | 动作填充掩码 |
| `high_level_task` | str | - | 高层任务描述（如 "clean the bedroom"） |
| `subtask` | str | - | 子任务标签（如 "pick up pillow"） |
| `text` | str | - | 原始低层指令（向后兼容） |

## 2. 模型输入

### 2.1 预处理流程

**步骤 1：图像处理**
- 多相机支持：`[B, 3, H, W]` 或 `List[Tensor]`
- 数据增强（训练时）：随机裁剪、旋转、颜色抖动
- 归一化到 `[-1, 1]` 范围
- 分辨率：`image_resolution` (默认 224×224)

**步骤 2：状态处理**
```python
# 1. Padding 到 max_state_dim
states_padded = pad_vector(states, max_state_dim)  # [B, max_state_dim]

# 2. 归一化到 [-1, 1]（使用 quantile 或 mean/std）
state_norm = normalize_feature(states_padded, dataset_stats.state_stats)

# 3. 离散化为 0-255 整数（关键！）
state_idx = discretize_state_256(state_norm)  # [B, max_state_dim]
# 映射：[-1, 1] → [0, 255]
```

**步骤 3：Prompt 构建**

**高层 Prompt（用于子任务预测）：**
```
"Task: {high_level_task}, State: {idx1 idx2 idx3 ...};\nSubtask: "
```

示例：
```
"Task: clean the bedroom, State: 128 145 67 89 201 ...;\nSubtask: "
```

**低层 Prompt（用于动作生成）：**
```
"Task: {subtask}, State: {idx1 idx2 idx3 ...};\nAction: "
```

示例：
```
"Task: pick up pillow, State: 128 145 67 89 201 ...;\nAction: "
```

**步骤 4：Tokenization**
```python
# 使用 PaliGemma tokenizer
high_level_input_ids = tokenizer(high_level_prompts)      # [B, L]
subtask_input_ids = tokenizer(subtask_targets)            # [B, S]  # 标签
low_level_input_ids = tokenizer(low_level_prompts)        # [B, L']
```

### 2.2 最终输入张量

| 输入 | 形状 | 说明 |
|------|------|------|
| `pixel_values` | `[B, 3, H, W]` | 图像张量 |
| `high_level_input_ids` | `[B, L]` | 高层 prompt token IDs |
| `high_level_attention_mask` | `[B, L]` | 高层 prompt 注意力掩码 |
| `subtask_input_ids` | `[B, S]` | **标签**：目标子任务 token IDs |
| `subtask_attention_mask` | `[B, S]` | 子任务标签掩码 |
| `actions` | `[B, chunk_size, action_dim]` | 归一化动作序列 |
| `action_is_pad` | `[B, chunk_size]` | 动作填充掩码 |

## 3. 模型输出

### 3.1 上层策略输入组成（关键理解）

**上层策略的输入由三部分组成，全部通过 `embed_prefix` 拼接：**

```python
prefix_embs = embed_prefix(
    images=[pixel_values],           # 1. Image tokens (SigLIP vision encoder)
    lang_tokens=high_level_input_ids # 2. Language tokens (包含 task + discrete state)
)
```

**详细分解：**

1. **Image Tokens** `[B, num_img_tokens, hidden_dim]`
   - 通过 SigLIP vision encoder 从 `pixel_values` 得到
   - 图像被编码为视觉 token 序列

2. **Task Tokens** `[B, num_task_tokens, hidden_dim]`
   - Prompt: `"Task: {high_level_task}, State: {discrete_state};\nSubtask: "`
   - 通过 `embed_language_tokens(input_ids)` 得到
   - **注意**：Discrete state 已经作为文本字符串（如 "128 145 67 ..."）包含在 prompt 中，被 tokenizer 一起编码成 token IDs

3. **拼接后的 Prefix Embeddings** `[B, prefix_len, hidden_dim]`
   - `prefix_len = num_img_tokens + num_task_tokens`
   - 顺序：`[image_tokens, task_tokens]`

### 3.2 前向传播流程

**阶段 1：高层文本路径（子任务预测）**

```python
# 1. Embed prefix（图像 + 高层 prompt）
prefix_embs = embed_prefix_simple(pixel_values, high_level_input_ids)
# prefix_embs 包含：
#   - Image tokens (通过 SigLIP)
#   - Task tokens (包含 "Task: {task}, State: {discrete_state};\nSubtask: ")
# prefix_embs: [B, prefix_len, hidden_dim]

# 2. Forward through PaliGemma language model only
lm_outputs = paligemma.language_model(prefix_embs)
prefix_out_text = lm_outputs.last_hidden_state  # [B, prefix_len, hidden_dim]

# 3. Get logits from LM head
logits = lm_head(prefix_out_text)  # [B, prefix_len, vocab_size]

# 4. Extract subtask prediction logits (last S positions)
shift_logits = logits[:, -S-1:-1, :]  # [B, S, vocab_size]
# 这些 logits 对应 prompt 末尾 "Subtask: " 之后的预测位置
```

**阶段 2：低层动作路径（动作生成）**

```python
# 1. Detach prefix embeddings（关键！防止动作梯度影响高层）
prefix_embs_detached = prefix_embs.detach()

# 2. Sample noise and time for flow matching
noise = sample_noise(actions.shape)  # [B, T, A]
time = sample_time(B)                # [B]

# 3. Build noisy actions
x_t = time * noise + (1 - time) * actions
u_t = noise - actions  # Target vector field

# 4. Embed suffix（noisy actions）
suffix_embs = embed_suffix(x_t, time)

# 5. Joint forward（prefix + suffix）
(_, suffix_out) = paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs_detached, suffix_embs]
)

# 6. Project to action space
v_t = action_out_proj(suffix_out)  # [B, T, A]
```

### 3.2 输出结构

```python
Pi05HierarchicalOutput(
    action_loss: Tensor,        # [B, T, A] Flow matching loss (reduction="none")
    text_loss: Tensor,          # [B, S] Cross-entropy loss for subtask
    total_loss: Tensor,         # Scalar: text_loss + α * action_loss
    prefix_embeds: Tensor,      # [B, prefix_len, hidden_dim]
    suffix_embeds: Tensor,      # [B, T, hidden_dim]
)
```

## 4. 标签（Ground Truth）

### 4.1 文本标签（子任务预测）

**标签格式：**
- **类型**：字符串 → Token IDs
- **形状**：`[B, S]`，其中 `S` 是子任务 token 序列长度
- **示例**：
  ```python
  subtask_text = "pick up pillow"
  subtask_ids = tokenizer.encode(subtask_text)  # [2, 45, 123, 789, ...]
  ```

**标签掩码：**
- `subtask_attention_mask`: `[B, S]`，标记有效 token 位置
- 用于在计算 loss 时忽略 padding

### 4.2 动作标签（动作生成）

**标签格式：**
- **类型**：连续动作序列
- **形状**：`[B, chunk_size, action_dim]`
- **预处理**：
  1. 归一化到 `[-1, 1]`（使用 quantile 或 mean/std）
  2. Padding 到 `chunk_size`（默认 50）
  3. 生成 `action_is_pad` 掩码

**Flow Matching 目标：**
- 目标向量场：`u_t = noise - actions`
- 模型预测：`v_t = action_out_proj(suffix_out)`
- Loss：`||u_t - v_t||²`

## 5. Loss 计算

### 5.1 联合损失公式（Pi0.5 论文 Equation 1）

$$
L_{\text{total}} = L_{\text{text}} + \alpha \cdot L_{\text{action}}
$$

其中：
- $L_{\text{text}}$：子任务预测的交叉熵损失
- $L_{\text{action}}$：动作生成的 Flow Matching 损失
- $\alpha$：动作损失权重（默认 10.0）

### 5.2 文本损失（Cross-Entropy）

```python
# 1. 获取 logits（shift 用于 next-token prediction）
shift_logits = logits[:, -S-1:-1, :]  # [B, S, vocab_size]
shift_labels = subtask_ids             # [B, S]

# 2. 计算交叉熵（reduction="none"）
flat_logits = shift_logits.view(-1, vocab_size)  # [B*S, vocab_size]
flat_labels = shift_labels.view(-1)              # [B*S]
text_loss_per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
text_loss = text_loss_per_token.view(B, S)  # [B, S]

# 3. 应用掩码（忽略 padding）
if subtask_mask is not None:
    text_loss = text_loss * subtask_mask.float()

# 4. 聚合
text_loss_mean = text_loss.mean()  # Scalar
```

**更新参数：**
- ✅ PaliGemma VLM（图像编码器 + 语言模型）
- ✅ LM head（文本生成层）

### 5.3 动作损失（Flow Matching）

```python
# 1. 计算 MSE（reduction="none"）
action_loss = F.mse_loss(u_t, v_t, reduction="none")  # [B, T, A]

# 2. 应用动作填充掩码
if action_is_pad.any():
    mask = ~action_is_pad.unsqueeze(-1).expand_as(action_loss)
    action_loss_masked = (action_loss * mask).sum() / mask.sum()
else:
    action_loss_masked = action_loss.mean()

# 3. 加权
action_loss_weighted = alpha * action_loss_masked  # alpha = 10.0
```

**更新参数：**
- ✅ Action Expert（Gemma expert）
- ✅ `action_in_proj`、`action_out_proj`
- ✅ `time_mlp_in`、`time_mlp_out`
- ❌ **不更新** PaliGemma VLM（通过 `detach()` 截断梯度）

### 5.4 梯度截断机制（关键！）

**目的：** 防止动作专家的梯度影响高层规划器（VLM）

**实现：**
```python
# 在 forward_hierarchical 中
prefix_embs_detached = prefix_embs.detach()  # 截断梯度

# 动作路径使用 detached embeddings
(_, suffix_out) = paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs_detached, suffix_embs]  # prefix 已 detach
)
```

**效果：**
- 文本损失 → 更新 VLM ✅
- 动作损失 → 只更新 Action Expert ✅
- 动作损失 → **不更新** VLM ❌

## 6. 训练模式

### 6.1 联合训练（`--hierarchical`）

```bash
python -m pi_0_5_torch.train \
    --hierarchical \
    --alpha 10.0 \
    --data_root <path> \
    --instruction_json <path>
```

- 同时训练文本和动作
- Loss：`L = L_text + 10.0 * L_action`

### 6.2 高层策略微调（`--planar_mode`）

**标准模式（只训练文本）：**
```bash
python -m pi_0_5_torch.train \
    --planar_mode \
    --data_root <path> \
    --instruction_json <path>
```

- 只训练子任务预测
- `alpha = 0.0`（忽略动作损失）
- Loss：`L = L_text`

**预训练风格模式（推荐用于任务迁移）：**
```bash
python -m pi_0_5_torch.train \
    --planar_mode \
    --use_fast_tokens \
    --fast_tokenizer_path "physical-intelligence/fast" \
    --data_root <path> \
    --instruction_json <path>
```

- 训练子任务预测 + FAST 离散动作 token
- `alpha = 0.0`（只使用文本交叉熵损失）
- Loss：`L = H(subtask_tokens, logits) + H(fast_action_tokens, logits)`
- **优势**：
  - ✅ 子任务和动作 token 在同一个语义空间对齐
  - ✅ 复用预训练阶段的知识（论文预训练就是这样做的）
  - ✅ 对任务迁移有帮助，因为语义对齐有助于跨域理解
  - ✅ 符合论文预训练阶段的训练方式（$\alpha = 0$，只训练文本预测）

**实现说明：**
- 需要先使用 `domain_transfer.py` 训练/适配 FAST tokenizer 到你的数据域
- 动作会被 FAST tokenizer 离散化为 token IDs
- 这些 token IDs 会作为文本序列的一部分，与 subtask tokens 一起参与交叉熵损失
- 这样 subtask 和 action tokens 在同一个语义空间训练，有助于任务迁移

### 6.3 动作专家训练（`--action_expert_mode`）

```bash
python -m pi_0_5_torch.train \
    --action_expert_mode \
    --data_root <path>
```

- 只训练动作生成
- 不使用层级数据
- Loss：`L = L_action`

## 7. 数据准备检查清单

- [ ] 目录结构正确（层级或 stage 结构）
- [ ] `actions.npy` 和 `qpos.npy` 存在
- [ ] 图像文件在 `<camera_name>/` 目录下
- [ ] `instruction_json` 包含 `high_level_task` 和 `subtask` 字段
- [ ] 状态维度 ≤ `max_state_dim`（默认 32）
- [ ] 动作维度 ≤ `max_action_dim`（默认 32）
- [ ] 数据集统计信息（`dataset_stats`）已计算（用于归一化）

## 8. 关键要点总结

### 8.1 上层策略（High-Level Policy）训练

**输入组成：**
- ✅ **Image Tokens**: 通过 SigLIP vision encoder 从图像得到
- ✅ **Task Tokens**: `"Task: {high_level_task}, State: {discrete_state};\nSubtask: "` 被 tokenize
- ✅ **Discrete State Tokens**: 状态被离散化为 0-255 整数，作为字符串（如 "128 145 67 ..."）包含在 prompt 中，一起被 tokenize

**输出：**
- ✅ **Predicted Subtask Logits**: `[B, S, vocab_size]` - 在 prompt 末尾位置预测子任务 token

**标签：**
- ✅ **GT Subtask Token IDs**: `[B, S]` - 目标子任务的 token IDs

**损失计算：**
- ✅ **交叉熵损失**：`CE(pred_subtask_logits, gt_subtask_ids)`
- ✅ **只更新 VLM**（PaliGemma 的图像编码器和语言模型）

### 8.2 低层动作（Low-Level Action）训练

**重要：动作不需要离散化！**
- ❌ **动作不是离散 token**
- ✅ **动作是连续值**，使用 Flow Matching 训练
- ✅ **动作归一化到 `[-1, 1]`**，但保持连续
- ✅ **损失**：`MSE(u_t, v_t)`，其中 `u_t = noise - actions` 是目标向量场

### 8.3 其他关键点

1. **状态离散化**：机器人状态被离散化为 0-255 整数，作为文本 token 输入（仅用于 prompt）
2. **双路径训练**：文本路径和动作路径分离，通过 `detach()` 防止梯度干扰
3. **联合损失**：`L = L_text + α * L_action`，默认 α=10.0
4. **梯度保护**：动作损失不更新 VLM，只更新 Action Expert

