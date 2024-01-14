import json
from typing import List, Dict, Any

import torch
from torch import nn 
from transformers import get_scheduler, PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def load_json(fp: str) -> Any:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def json_stringify(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=4)


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        data = []
        for idx, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except Exception as e:
                pass
    return data


def write_jsonline(fp: str, mode: str, obj: List[Any]):
    with open(fp, mode, encoding="utf-8") as f:
        for x in obj:
            if isinstance(x, str):
                f.write(x.strip() + "\n")
            else:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_json(fp: str, obj: Any):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def load_txt(fp: str) -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        return [i.strip() for i in f]


def create_lr_scheduler(
        num_training_steps: int, optimizer: torch.optim.Optimizer, args: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    return lr_scheduler


def create_optimizer(model: PreTrainedModel, args: Any) -> torch.optim.Optimizer:
    decay_parameters = get_decay_parameter_names(model=model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0
        }
    ]
    return torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-7
    )


def get_decay_parameter_names(model: PreTrainedModel) -> List[str]:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def find_all_linear_names(model: nn.Module) -> List[str]:
    linear_class = nn.Linear
    lora_module_names = set([])

    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)