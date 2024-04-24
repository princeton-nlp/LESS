from dataclasses import asdict, dataclass, field, fields

from transformers import TrainingArguments as TA
from transformers.utils import logging

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

fsdp_config = {
    "mpt7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "opt125m_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["OPTDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "mpt7b_lora": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_13b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "mistral_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
}


@dataclass
class TrainingArguments(TA):
    analysis_mode: float = field(
        default=False,
        metadata={
            "help": (
                "Whether to run in analysis mode. "
            )
        },
    )
    analysis_dataset: str = field(
        default="bbh",
        metadata={
            "help": (
                "The dataset to use for analysis mode. "
            )
        },
    )
    train_dataset_names: str = field(
        default=None,
        metadata={
            "help": (
                "The dataset to use for training. "
            )
        },
    )

    def __post_init__(self):
        if isinstance(self.fsdp_config, str):
            self.fsdp_config = fsdp_config[self.fsdp_config]
        if self.train_dataset_names is not None:
            self.train_dataset_names = self.train_dataset_names.split(" ")
        super().__post_init__()
