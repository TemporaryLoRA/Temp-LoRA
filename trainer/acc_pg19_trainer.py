import sys

sys.path.append(".")

import argparse
from tqdm import tqdm
import os, gc, sys, copy

import torch
from torch.utils.data import DataLoader, SequentialSampler

from peft import LoraConfig, TaskType, get_peft_model

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple

from data_modules.pg19 import PG19Dataset, PG19SlowRawDataset
from helper import (
    create_lr_scheduler, create_optimizer, write_json, find_all_linear_names, load_json, load_txt, load_jsonline,
    write_jsonline, json_stringify
)

logger = get_logger(name=__name__)


@dataclass
class TrainingArguments:
    model_name: str
    train_fp: str
    eval_fp: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    warmup_steps: int
    output_dir: str
    training_input_length: int
    stride_size: int
    eval_input_length: int
    num_train_epochs: int
    gradient_checkpointing: str
    use_flash_attention_2: bool


def to_device(obj: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device) if hasattr(v, "to") else v for k, v in obj.items()}


def train_once(model: PreTrainedModel, batch: Dict[str, Any]) -> Tuple[float, torch.Tensor]:
    model.train()
    with accelerator.accumulate(model):
        outputs: CausalLMOutputWithPast = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
        )
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    return torch.exp(loss.detach().clone()).item(), loss


@torch.no_grad()
def inference_once(model: PreTrainedModel, batch: Dict[str, Any], disable_lora: bool) -> Tuple[float, torch.Tensor]:
    model.eval()
    inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
    new_tokens = batch["new_tokens"]
    if disable_lora:
        with accelerator.unwrap_model(model=model).disable_adapter():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)
    return torch.exp(outputs.loss).item() * new_tokens, outputs.loss


@torch.no_grad()
def evaluate(
        eval_dataloader: DataLoader, total_tokens: int, eval_total_tokens: int, eval_base_ppl: float,
        eval_lora_ppl: float, eval_last_idx: int, last_infer_input_ids: torch.Tensor
) -> Tuple[float, float, int, int]:
    for e_idx, batch in enumerate(eval_dataloader):
        if e_idx <= eval_last_idx:
            continue
        batch = to_device(obj=batch, device=device)
        new_tokens = batch["new_tokens"]
        eval_inference_ids = batch["input_ids"][:, -args.stride_size:].detach().contiguous()
        step_base_ppl, step_base_loss = inference_once(model=model, batch=batch, disable_lora=True)
        step_lora_ppl, step_lora_loss = inference_once(model=model, batch=batch, disable_lora=False)

        eval_base_ppl += step_base_ppl
        eval_lora_ppl += step_lora_ppl
        eval_total_tokens += new_tokens

        if eval_total_tokens >= total_tokens:
            eval_last_idx = e_idx
            assert eval_total_tokens == total_tokens
            min_length = min(eval_inference_ids.size(-1), last_infer_input_ids.size(-1))
            assert torch.all(eval_inference_ids[:, -min_length:] == last_infer_input_ids[:, -min_length:])
            break

        assert total_tokens == eval_total_tokens
    return eval_base_ppl, eval_lora_ppl, eval_total_tokens, eval_last_idx


def parse_args() -> TrainingArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_fp", type=str)
    parser.add_argument("--eval_fp", type=str)
    parser.add_argument("--data_save_dir", type=str)
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--training_input_length", type=int, default=3072)
    parser.add_argument("--stride_size", type=int)
    parser.add_argument("--eval_input_length", type=int, default=3072)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--gradient_checkpointing", type=str)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    args = parser.parse_args()
    obj = {}

    for f in fields(TrainingArguments):
        v = getattr(args, f.name, None)
        if v is not None:
            obj[f.name] = v
    # obj["eval_input_prompt_lengths"] = [int(i.strip()) for i in obj["eval_input_prompt_lengths"].split(",")]
    return TrainingArguments(**obj)


if __name__ == '__main__':
    args: TrainingArguments = parse_args()
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.output_dir, gradient_accumulation_steps=1)

    accelerator.wait_for_everyone()
    set_seed(42)

    device = torch.device("cuda")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        truth_remote_code=True,
        use_fast=False
    )
    tokenizer.padding_side = "right"

    if "qwen" in args.model_name.lower():
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    model_kwargs = {
        "pretrained_model_name_or_path": args.model_name,
        "trust_remote_code": True,
        "use_cache": False if args.use_flash_attention_2 else None,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda"
    }
    if args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if args.gradient_checkpointing == "true":
        print("gradient checkpointing...")
        model.gradient_checkpointing_enable()

    if args.lora_rank > 0:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=find_all_linear_names(model=model)
        )
        model.enable_input_require_grads()
        model = get_peft_model(model=model, peft_config=lora_config)
        model.print_trainable_parameters()

    with open(args.train_fp, "r", encoding="utf-8") as f:
        text = "".join(f.readlines()).strip()
        input_ids = torch.tensor(data=[tokenizer.encode(text)], dtype=torch.int64)

    _dataset: PG19SlowRawDataset = PG19SlowRawDataset(
        fp=args.train_fp, tokenizer=tokenizer, prefix_length=args.training_input_length,
        stride_size=args.stride_size
    )
    _dataset.load_from_input_ids(input_ids=input_ids)
    train_dataset = PG19Dataset(dataset=_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False, sampler=SequentialSampler(data_source=train_dataset),
        num_workers=0, collate_fn=lambda i: i[0], pin_memory=True
    )
    num_training_steps = len(train_dataloader)

    _dataset: PG19SlowRawDataset = PG19SlowRawDataset(
        fp=args.train_fp, tokenizer=tokenizer, prefix_length=args.eval_input_length,
        stride_size=args.stride_size
    )
    _dataset.load_from_input_ids(input_ids=input_ids)
    eval_dataset = PG19Dataset(dataset=_dataset)
    eval_dataloader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False, sampler=SequentialSampler(data_source=eval_dataset),
        num_workers=0, collate_fn=lambda i: i[0], pin_memory=True
    )

    optimizer = create_optimizer(model=model, args=args)
    lr_scheduler = create_lr_scheduler(num_training_steps=num_training_steps, optimizer=optimizer, args=args)
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    pbr = tqdm(range(0, num_training_steps), disable=not accelerator.is_local_main_process)
    total_tokens, base_ppl, lora_ppl = 0, 0, 0
    step_record = []

    eval_last_index = -1
    eval_total_tokens = 0
    eval_base_ppl = 0
    eval_lora_ppl = 0

    for b_idx, batch in enumerate(train_dataloader):
        batch = to_device(obj=batch, device=device)
        new_tokens = batch["new_tokens"]
        serial_id = batch["serial_id"]
        total_tokens += new_tokens

        step_base_ppl, step_base_loss = inference_once(model=model, batch=batch, disable_lora=True)
        step_lora_ppl, step_lora_loss = inference_once(model=model, batch=batch, disable_lora=False)
        base_ppl += step_base_ppl
        lora_ppl += step_lora_ppl

        _record = {
            "b_idx": b_idx,
            "inference": {
                "step_base_ppl": step_base_ppl,
                "step_lora ppl": step_lora_ppl,
                "step_base_loss": step_base_loss.cpu().item(),
                "step_lora_loss": step_lora_loss.cpu().item(),
            },
            "train": {},
            "lr": lr_scheduler.get_lr(),
            "total_tokens": total_tokens,
            "new_tokens": new_tokens,
            "serial_id": serial_id,
            "eval_step_record": {}
        }

        last_infer_input_ids = batch["input_ids"][:, -args.stride_size:].detach().contiguous()
        eval_base_ppl, eval_lora_ppl, eval_total_tokens, eval_last_index = evaluate(
            eval_dataloader=eval_dataloader,
            total_tokens=total_tokens,
            eval_total_tokens=eval_total_tokens,
            eval_base_ppl=eval_base_ppl,
            eval_lora_ppl=eval_lora_ppl,
            eval_last_idx=eval_last_index,
            last_infer_input_ids=last_infer_input_ids
        )

        _record["eval_step_record"][str(args.eval_input_length)] = {
            "base_ppl": eval_base_ppl,
            "lora_ppl": eval_lora_ppl,
            "mean_base_ppl": eval_base_ppl / eval_total_tokens,
            "mean_lora_ppl": eval_lora_ppl / eval_total_tokens,
            "total_tokens": eval_total_tokens
        }
        step_record.append(_record)
        step_lora_loss = 0

        for i in range(0, args.num_train_epochs):
            _step_lora_ppl, step_lora_loss = train_once(model=model, batch=batch)
            step_record[-1]["train"][f"train_lora_{i + 1}"] = {
                "ppl": _step_lora_ppl,
                "loss": step_lora_loss.detach().cpu().item(),
            }
        lr_scheduler.step()
        pbr.update(n=1)
        if b_idx % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        if b_idx % 10 == 0:
            write_json(
                fp=os.path.join(args.output_dir, args.train_fp.split("/")[-1].replace(".txt", ".json")),
                obj=[{
                    "done": False,
                    "file": args.train_fp.split("/")[-1],
                    "base_ppl": base_ppl,
                    "lora_ppl": lora_ppl,
                    "num_tokens": total_tokens,
                    "mean_base_ppl": base_ppl / total_tokens,
                    "mean_lora_ppl": lora_ppl / total_tokens,
                    "steps": step_record
                }]
            )
    pbr.close()
    write_json(
        fp=os.path.join(args.output_dir, args.train_fp.split("/")[-1].replace(".txt", ".json")),
        obj=[{
            "done": True,
            "file": args.train_fp.split("/")[-1],
            "base_ppl": base_ppl,
            "lora_ppl": lora_ppl,
            "num_tokens": total_tokens,
            "mean_base_ppl": base_ppl / total_tokens,
            "mean_lora_ppl": lora_ppl / total_tokens,
            "steps": step_record
        }]
    )
