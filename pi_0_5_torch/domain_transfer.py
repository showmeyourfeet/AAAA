"""
Domain Transfer Utilities for Pi0.5

This module provides utilities for transferring Pi0.5 to a new domain
(e.g., from home robotics to surgical robotics).

Key Steps:
1. Prepare action data (chunk, normalize)
2. Fit FAST tokenizer on new domain data (REQUIRED: adapts to new action distribution)
3. Expand VLM vocabulary with action tokens
4. Resize model embeddings

Why Re-train FAST Tokenizer?
-----------------------------
When transferring to a new domain (e.g., home → surgical), the action distribution
can be significantly different:
- Different action ranges and dynamics
- Different normalization statistics (quantile percentiles)
- Different BPE vocabulary needs

The FAST tokenizer.fit() method:
- Recomputes normalization statistics (1st/99th percentiles) from new domain data
- Retrains the BPE vocabulary to efficiently compress new action patterns
- Ensures the discrete action tokens align with the new domain's action space

Required Data:
--------------
- action_trajectories: Raw action trajectories from the new domain
    - Format: List[np.ndarray] where each array is [T_i, action_dim]
    - Or: np.ndarray of shape [N, T, action_dim]
    - Should represent typical action sequences in the new domain
    - No need for paired observations or task labels (actions only)
    
- vlm_model: Pre-trained VLM model (e.g., PaliGemma)
- vlm_tokenizer: Pre-trained VLM tokenizer

Usage:
    from domain_transfer import prepare_surgical_domain
    
    # Collect raw action trajectories from surgical domain
    surgical_actions = load_surgical_trajectories()  # List of [T, action_dim] arrays
    
    # One-stop domain transfer
    fast_tokenizer, updated_vlm_tokenizer, info = prepare_surgical_domain(
        surgical_actions=surgical_actions,  # Raw action trajectories from new domain
        vlm_model=paligemma_model,
        vlm_tokenizer=paligemma_tokenizer,
        output_dir="./surgical_fast",
        control_freq=50,
    )
    
    # The FAST tokenizer is now adapted to surgical actions
    # The VLM tokenizer has been expanded with action tokens
    # Save the updated model for training
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# Action Data Preprocessing
# ============================================================================


@dataclass
class ActionPreprocessConfig:
    """Configuration for action data preprocessing."""
    
    chunk_duration: float = 1.0  # Duration of each chunk in seconds
    control_freq: int = 50  # Control frequency in Hz
    normalize: bool = True  # Whether to pre-normalize to [-1, 1]
    
    @property
    def chunk_size(self) -> int:
        return int(self.chunk_duration * self.control_freq)


def compute_quantile_stats(
    actions: np.ndarray,
    q_low: float = 1.0,
    q_high: float = 99.0,
) -> Dict[str, np.ndarray]:
    """
    Compute quantile statistics for normalization.
    
    Args:
        actions: [N, T, action_dim] or [N, action_dim] action data
        q_low: Lower quantile (default 1st percentile)
        q_high: Upper quantile (default 99th percentile)
        
    Returns:
        Dict with 'q_low', 'q_high' arrays
    """
    # Flatten to [total_samples, action_dim]
    if actions.ndim == 3:
        flat = actions.reshape(-1, actions.shape[-1])
    elif actions.ndim == 2:
        flat = actions
    else:
        raise ValueError(f"Expected 2D or 3D array, got {actions.ndim}D")
    
    stats = {
        "q_low": np.percentile(flat, q_low, axis=0),
        "q_high": np.percentile(flat, q_high, axis=0),
    }
    
    logger.info(f"Computed quantile stats: q_low shape={stats['q_low'].shape}")
    return stats


def normalize_actions(
    actions: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    """
    Normalize actions to [-1, 1] using quantile statistics.
    
    Following FAST paper Section V.B:
    "We first normalize the input actions, such that the 1st and 99th quantile
    of values in the training dataset for each action dimension maps to [-1, 1]"
    """
    # Avoid division by zero
    range_vals = q_high - q_low
    range_vals = np.where(range_vals < 1e-6, 1.0, range_vals)
    
    # Center and scale
    center = (q_low + q_high) / 2
    normalized = 2 * (actions - center) / range_vals
    
    return np.clip(normalized, -1.0, 1.0)


def chunk_trajectories(
    trajectories: List[np.ndarray],
    chunk_size: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Split trajectories into fixed-size chunks.
    
    Args:
        trajectories: List of [T_i, action_dim] trajectories
        chunk_size: Number of timesteps per chunk
        stride: Stride between chunks (default=chunk_size, no overlap)
        
    Returns:
        [N_chunks, chunk_size, action_dim] array of chunks
    """
    if stride is None:
        stride = chunk_size
    
    chunks = []
    for traj in trajectories:
        T = traj.shape[0]
        for start in range(0, T - chunk_size + 1, stride):
            chunk = traj[start:start + chunk_size]
            chunks.append(chunk)
    
    if not chunks:
        raise ValueError(
            f"No valid chunks found. Ensure trajectories have at least "
            f"{chunk_size} timesteps."
        )
    
    result = np.stack(chunks, axis=0)
    logger.info(f"Created {len(chunks)} chunks of shape {result.shape}")
    return result


def prepare_action_dataset(
    raw_trajectories: Union[List[np.ndarray], np.ndarray],
    config: ActionPreprocessConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepare action dataset for FAST tokenizer training.
    
    This is the main preprocessing function that:
    1. Chunks trajectories into 1-second segments
    2. Computes normalization statistics
    3. Normalizes actions to [-1, 1]
    
    Args:
        raw_trajectories: Raw action trajectories
            - List[np.ndarray]: List of [T_i, action_dim] arrays
            - np.ndarray: Single [N, T, action_dim] array
        config: Preprocessing configuration
        
    Returns:
        Tuple of:
        - Normalized action chunks: [N, chunk_size, action_dim]
        - Normalization statistics dict
        
    Example:
        >>> # For surgical robotics at 100Hz
        >>> config = ActionPreprocessConfig(
        ...     chunk_duration=1.0,
        ...     control_freq=50,  # Downsample to 50Hz
        ... )
        >>> chunks, stats = prepare_action_dataset(surgical_trajs, config)
    """
    # Convert to list of trajectories if needed
    if isinstance(raw_trajectories, np.ndarray):
        if raw_trajectories.ndim == 3:
            trajectories = [raw_trajectories[i] for i in range(len(raw_trajectories))]
        else:
            trajectories = [raw_trajectories]
    else:
        trajectories = raw_trajectories
    
    # 1. Chunk trajectories
    chunks = chunk_trajectories(
        trajectories,
        chunk_size=config.chunk_size,
        stride=config.chunk_size,  # No overlap for training
    )
    
    # 2. Compute normalization stats
    stats = compute_quantile_stats(chunks)
    
    # 3. Normalize if requested
    if config.normalize:
        chunks = normalize_actions(chunks, stats["q_low"], stats["q_high"])
    
    return chunks, stats


# ============================================================================
# FAST Tokenizer Training (using official implementation)
# ============================================================================


def fit_fast_tokenizer(
    action_chunks: np.ndarray,
    pretrained_path: str = "physical-intelligence/fast",
) -> Any:
    """
    Fit FAST tokenizer on new domain data using official implementation.
    
    This function adapts the FAST tokenizer to a new action distribution by:
    1. Recomputing normalization statistics (quantile percentiles) from new data
    2. Retraining the BPE vocabulary to efficiently compress new action patterns
    
    Why re-train?
    -------------
    Different domains (e.g., home vs. surgical) have different action distributions:
    - Different action ranges and dynamics
    - Different normalization statistics needed
    - Different action patterns that benefit from domain-specific BPE tokens
    
    Args:
        action_chunks: [N, chunk_size, action_dim] normalized action chunks
            Should be normalized to [-1, 1] range (done by prepare_action_dataset)
        pretrained_path: Path or HF hub ID for pretrained FAST tokenizer
            The pretrained tokenizer provides the base architecture and initialization
        
    Returns:
        Fitted FAST tokenizer adapted to the new domain's action distribution
        
    Note:
        The tokenizer.fit() method retrains both:
        - Normalization statistics (quantile-based normalization)
        - BPE vocabulary (Byte Pair Encoding for action compression)
    """
    from transformers import AutoProcessor
    
    logger.info(f"Loading FAST tokenizer from {pretrained_path}")
    tokenizer = AutoProcessor.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
    )
    
    logger.info(f"Fitting FAST tokenizer on {len(action_chunks)} action chunks")
    # The official tokenizer.fit() retrains BPE and normalization stats
    new_tokenizer = tokenizer.fit(action_chunks)
    
    return new_tokenizer


# ============================================================================
# VLM Vocabulary Expansion
# ============================================================================


def get_action_token_names(
    vocab_size: int,
    prefix: str = "<action_",
    suffix: str = ">",
) -> List[str]:
    """Generate action token names."""
    return [f"{prefix}{i}{suffix}" for i in range(vocab_size)]


def expand_vlm_vocabulary(
    vlm_tokenizer,
    action_vocab_size: int = 1024,
    action_token_prefix: str = "<action_",
) -> Tuple[Any, Dict[int, int]]:
    """
    Expand VLM tokenizer vocabulary with action tokens.
    
    Args:
        vlm_tokenizer: HuggingFace tokenizer for VLM
        action_vocab_size: Number of action tokens to add
        action_token_prefix: Prefix for action token names
        
    Returns:
        Tuple of:
        - Updated tokenizer
        - Mapping from action token index to VLM token ID
    """
    # Generate action token names
    action_tokens = get_action_token_names(
        action_vocab_size,
        prefix=action_token_prefix,
    )
    
    # Record old vocab size
    old_vocab_size = len(vlm_tokenizer)
    
    # Check if tokens already exist
    existing_tokens = []
    missing_tokens = []
    for token_name in action_tokens:
        if token_name in vlm_tokenizer.get_vocab():
            existing_tokens.append(token_name)
        else:
            missing_tokens.append(token_name)
    
    # Add tokens (only missing ones)
    if missing_tokens:
        num_added = vlm_tokenizer.add_tokens(missing_tokens, special_tokens=False)
        logger.info(
            f"Expanded VLM vocabulary: {old_vocab_size} → {len(vlm_tokenizer)} "
            f"(+{num_added} new action tokens, {len(existing_tokens)} already existed)"
        )
    else:
        num_added = 0
        logger.info(
            f"VLM vocabulary already contains all {len(action_tokens)} action tokens. "
            f"No expansion needed (vocab size: {len(vlm_tokenizer)})"
        )
    
    # Create mapping: action_idx -> vlm_token_id
    # This mapping is always needed, even if tokens already existed
    action_to_vlm = {}
    for i, token_name in enumerate(action_tokens):
        vlm_id = vlm_tokenizer.convert_tokens_to_ids(token_name)
        action_to_vlm[i] = vlm_id
    
    return vlm_tokenizer, action_to_vlm


# ============================================================================
# Model Embedding Resize
# ============================================================================


def resize_model_embeddings(
    model,
    new_vocab_size: int,
    init_method: str = "mean",
):
    """
    Resize model embeddings to accommodate new tokens.
    
    Args:
        model: HuggingFace model (e.g., PaliGemma, Gemma)
        new_vocab_size: Target vocabulary size
        init_method: Initialization for new embeddings
            - "mean": Mean of existing embeddings (recommended)
            - "random": Random initialization
            - "zero": Zero initialization
    """
    # Get old vocab size
    if hasattr(model, "config"):
        old_vocab_size = model.config.vocab_size
    else:
        embeddings = model.get_input_embeddings()
        old_vocab_size = embeddings.weight.shape[0]
    
    # Check if resize is needed
    if old_vocab_size == new_vocab_size:
        logger.info(
            f"Model embeddings already have correct size ({old_vocab_size}). "
            f"No resize needed."
        )
        return
    
    # Resize
    model.resize_token_embeddings(new_vocab_size)
    
    # Initialize new embeddings
    if init_method == "mean" and hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        with torch.no_grad():
            # Use mean of existing embeddings for new tokens
            mean_emb = embeddings.weight[:old_vocab_size].mean(dim=0)
            embeddings.weight[old_vocab_size:] = mean_emb
    elif init_method == "zero" and hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        with torch.no_grad():
            embeddings.weight[old_vocab_size:].zero_()
    # "random" is the default behavior of resize_token_embeddings
    
    # Also resize output embeddings (lm_head) if tied
    if hasattr(model, "get_output_embeddings"):
        output_emb = model.get_output_embeddings()
        if output_emb is not None and hasattr(model, "tie_weights"):
            model.tie_weights()
    
    logger.info(
        f"Resized model embeddings: {old_vocab_size} → {new_vocab_size} "
        f"(init: {init_method})"
    )


# ============================================================================
# Main Domain Transfer Function
# ============================================================================


def prepare_domain_transfer(
    action_trajectories: Union[List[np.ndarray], np.ndarray],
    vlm_model,
    vlm_tokenizer,
    output_dir: str,
    control_freq: int = 50,
    chunk_duration: float = 1.0,
    fast_vocab_size: int = 1024,
    fast_pretrained: str = "physical-intelligence/fast",
    embedding_init: str = "mean",
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Complete domain transfer preparation for Pi0.5.
    
    This is the main entry point that:
    1. Preprocesses action data (chunk, normalize)
    2. Fits FAST tokenizer on new domain
    3. Expands VLM vocabulary
    4. Resizes model embeddings
    5. Saves everything
    
    Args:
        action_trajectories: Raw action data
            - List[np.ndarray]: List of [T_i, action_dim] trajectories
            - np.ndarray: [N, T, action_dim] array
        vlm_model: VLM model to update (e.g., PaliGemma)
        vlm_tokenizer: VLM tokenizer to expand
        output_dir: Directory to save outputs
        control_freq: Control frequency in Hz (default 50)
        chunk_duration: Chunk duration in seconds (default 1.0)
        fast_vocab_size: FAST vocabulary size (default 1024)
        fast_pretrained: Pretrained FAST tokenizer path
        embedding_init: How to initialize new embeddings
        
    Returns:
        Tuple of:
        - Fitted FAST tokenizer
        - Updated VLM tokenizer
        - Info dict with stats and mappings
        
    Example:
        >>> # For surgical robotics
        >>> fast_tok, vlm_tok, info = prepare_domain_transfer(
        ...     action_trajectories=surgical_data,
        ...     vlm_model=paligemma,
        ...     vlm_tokenizer=paligemma_tokenizer,
        ...     output_dir="./surgical_pi05",
        ...     control_freq=50,
        ... )
        >>> 
        >>> # Now train Pi0.5 with the updated model and tokenizers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Pi0.5 Domain Transfer")
    logger.info("=" * 60)
    
    # =========================================
    # Step 1: Preprocess actions
    # =========================================
    logger.info("\n[Step 1/4] Preprocessing action data...")
    
    preprocess_config = ActionPreprocessConfig(
        chunk_duration=chunk_duration,
        control_freq=control_freq,
        normalize=True,
    )
    
    action_chunks, norm_stats = prepare_action_dataset(
        action_trajectories,
        preprocess_config,
    )
    
    logger.info(f"  - Action chunks shape: {action_chunks.shape}")
    logger.info(f"  - Chunk size: {preprocess_config.chunk_size}")
    
    # Save normalization stats
    np.savez(
        os.path.join(output_dir, "action_norm_stats.npz"),
        **norm_stats,
    )
    
    # =========================================
    # Step 2: Fit FAST tokenizer
    # =========================================
    logger.info("\n[Step 2/4] Fitting FAST tokenizer...")
    
    fast_tokenizer = fit_fast_tokenizer(
        action_chunks,
        pretrained_path=fast_pretrained,
    )
    
    # Save FAST tokenizer
    fast_save_path = os.path.join(output_dir, "fast_tokenizer")
    fast_tokenizer.save_pretrained(fast_save_path)
    logger.info(f"  - FAST tokenizer saved to: {fast_save_path}")
    
    # =========================================
    # Step 3: Expand VLM vocabulary
    # =========================================
    logger.info("\n[Step 3/4] Expanding VLM vocabulary...")
    
    vlm_tokenizer, action_to_vlm = expand_vlm_vocabulary(
        vlm_tokenizer,
        action_vocab_size=fast_vocab_size,
    )
    
    # Save updated VLM tokenizer
    vlm_save_path = os.path.join(output_dir, "vlm_tokenizer")
    vlm_tokenizer.save_pretrained(vlm_save_path)
    logger.info(f"  - VLM tokenizer saved to: {vlm_save_path}")
    
    # =========================================
    # Step 4: Resize model embeddings
    # =========================================
    logger.info("\n[Step 4/4] Resizing model embeddings...")
    
    new_vocab_size = len(vlm_tokenizer)
    resize_model_embeddings(
        vlm_model,
        new_vocab_size=new_vocab_size,
        init_method=embedding_init,
    )
    
    # =========================================
    # Save info and return
    # =========================================
    info = {
        "action_dim": action_chunks.shape[-1],
        "chunk_size": preprocess_config.chunk_size,
        "control_freq": control_freq,
        "fast_vocab_size": fast_vocab_size,
        "vlm_vocab_size": new_vocab_size,
        "action_to_vlm_mapping": action_to_vlm,
        "norm_stats": norm_stats,
    }
    
    # Save info
    import pickle
    with open(os.path.join(output_dir, "domain_transfer_info.pkl"), "wb") as f:
        pickle.dump(info, f)
    
    logger.info("\n" + "=" * 60)
    logger.info("Domain Transfer Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Action dim: {info['action_dim']}")
    logger.info(f"VLM vocab size: {info['vlm_vocab_size']}")
    logger.info("=" * 60)
    
    return fast_tokenizer, vlm_tokenizer, info


# ============================================================================
# Convenience function for surgical domain
# ============================================================================


def prepare_surgical_domain(
    surgical_actions: Union[List[np.ndarray], np.ndarray],
    vlm_model,
    vlm_tokenizer,
    output_dir: str = "./surgical_pi05",
    control_freq: int = 50,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Convenience function for surgical robotics domain transfer.
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> 
        >>> # Load base VLM
        >>> paligemma = AutoModelForCausalLM.from_pretrained("google/paligemma-3b-pt-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        >>> 
        >>> # Load your surgical action data
        >>> surgical_actions = load_surgical_data()  # List of [T, action_dim] arrays
        >>> 
        >>> # Do domain transfer
        >>> fast_tok, vlm_tok, info = prepare_surgical_domain(
        ...     surgical_actions,
        ...     paligemma,
        ...     tokenizer,
        ... )
        >>> 
        >>> # Save updated model
        >>> paligemma.save_pretrained("./surgical_pi05/model")
    """
    return prepare_domain_transfer(
        action_trajectories=surgical_actions,
        vlm_model=vlm_model,
        vlm_tokenizer=vlm_tokenizer,
        output_dir=output_dir,
        control_freq=control_freq,
        chunk_duration=1.0,
        fast_vocab_size=1024,
    )

