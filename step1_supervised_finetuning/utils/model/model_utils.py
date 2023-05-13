import torch
from transformers import (
    AutoConfig,
)


def create_hf_model(model_class,
                    model_name_or_path,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            torch_dtype=torch.float16)
    return model
