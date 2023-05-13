from transformers import AutoConfig
import os

def create_hf_model(model_class,
                    model_name_or_path,
                    use_8bit,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    if use_8bit:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                load_in_8bit=True, 
                device_map=device_map)
    if not use_8bit:
        model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config)
    return model
