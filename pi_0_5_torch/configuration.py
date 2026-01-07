# configuration_pi05.py
from typing import Optional, Tuple, Dict
from transformers import PretrainedConfig

class PI05Config(PretrainedConfig):
    model_type = "pi05"

    def __init__(
        self,
        # model variant configuration
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: str = "bfloat16",
        
        # dimension configuration
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        image_resolution: Tuple[int, int] = (224, 224),
        
        # sequence length configuration
        n_obs_steps: int = 1,
        chunk_size: int = 50,  # Action Horizon
        n_action_steps: int = 50,
        
        # Flow Matching parameters
        num_inference_steps: int = 10,
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        
        # training/fine-tuning options
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        
        # data normalization configuration
        normalization_mapping: Optional[Dict] = None,
        tokenizer_max_length: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.dtype = dtype
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_resolution = image_resolution
        
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        
        self.num_inference_steps = num_inference_steps
        self.time_sampling_beta_alpha = time_sampling_beta_alpha
        self.time_sampling_beta_beta = time_sampling_beta_beta
        self.time_sampling_scale = time_sampling_scale
        self.time_sampling_offset = time_sampling_offset
        self.min_period = min_period
        self.max_period = max_period
        
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.tokenizer_max_length = tokenizer_max_length
        
        self.normalization_mapping = normalization_mapping or {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES",
            "ACTION": "QUANTILES",
        }