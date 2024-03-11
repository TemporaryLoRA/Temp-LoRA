import sys
sys.path.append(".")

import argparse
from tqdm import tqdm
import os, gc, sys, copy, json

sys.path.append("../")

import torch
from torch.utils.data import DataLoader, SequentialSampler

from peft import LoraConfig, TaskType, get_peft_model, LoraModel

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig,
    StoppingCriteria, StoppingCriteriaList
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationConfig, GenerateOutput

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Tuple, TypedDict, Iterable, Optional


from data_modules.pg19 import PG19Dataset, PG19SimpleDataset
from helper import (
    create_lr_scheduler, create_optimizer, write_json, find_all_linear_names, load_json, load_txt, load_jsonline,
    write_jsonline, json_stringify
)

logger = get_logger(name=__name__)


@torch.no_grad()
def generate_once(
    generate_length: int,
    training_input_length: int,
    eval_input_length: int,
    prev_input_ids: torch.Tensor,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
    generation_config.min_new_tokens = generate_length
    generation_config.max_new_tokens = generate_length

    past_length = past_key_values[0][0].size(2) if past_key_values is not None else 0
    unwrap_model: LoraModel = accelerator.unwrap_model(model=model)
    unwrap_model.merge_adapter()
    unwrap_model.eval()

    input_ids = prev_input_ids[:, -(past_length + eval_input_length):]
    input_length = input_ids.size(-1)
    outputs: GenerateOutput = unwrap_model.generate(
        inputs=input_ids,
        generation_config=generation_config,
        past_key_values=past_key_values
    )
    unwrap_model.unmerge_adapter()
    generated = outputs.sequences[:, input_length:]
    assert generated.size(-1) == generate_length

    prev_input_ids = torch.cat(tensors=[prev_input_ids, generated], dim=-1)
    batch_input_ids = prev_input_ids[:, -(training_input_length + generate_length):]

    return {
        "input_ids": batch_input_ids,
        "labels": torch.cat(tensors=[
            torch.zeros(
                size=[1, training_input_length], dtype=torch.int64, device=generated.device
            ) - 100,
            generated
        ], dim=-1),
        "attention_mask": torch.ones_like(basic_input_ids),
    }, prev_input_ids, outputs.past_key_values


@dataclass 
class ExampleArgs:
    model_name: str 
    training_input_length: int 
    eval_input_length: int 
    kv_reuse_times: int
    stride_size: int 
    gradient_checkpointing: str 
    use_flash_attention_2: bool
    output_dir: str 
    weight_decay: float
    learning_rate: float
    lr_scheduler_type: str
    warmup_steps: int


def parse_args() -> ExampleArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--training_input_length", type=int)
    parser.add_argument("--eval_input_length", type=int)
    parser.add_argument("--kv_reuse_times", type=int)
    parser.add_argument("--stride_size", type=int)
    parser.add_argument("--gradient_checkpointing", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=2)
    args = parser.parse_args()
    obj = {}
    for f in fields(ExampleArgs):
        obj[f.name] = getattr(args, f.name)
    return ExampleArgs(**obj)




if __name__ == "__main__":
    args: ExampleArgs = parse_args()

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
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda"
    }
    if args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if args.gradient_checkpointing == "true":
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model=model)
    )
    model.enable_input_require_grads()
    model = get_peft_model(model=model, peft_config=lora_config)
    model.print_trainable_parameters()

    fp = "data/pg19/1684.txt"
    with open(fp, "r", encoding="utf-8") as f:
        text = "".join(f.readlines()).strip()
    
    basic_input_ids = torch.tensor(data=[tokenizer.encode(text)], dtype=torch.int64)
    basic_input_ids = basic_input_ids[:, :max(args.training_input_length, args.eval_input_length)]
    basic_input_ids = basic_input_ids.to(device=model.device)

    batch_input_ids = basic_input_ids.detach().clone().to("cpu")
    inputs = {
        "input_ids": batch_input_ids,
        "attention_mask": None, 
        "labels": None
    }

    num_training_steps = 50 
    train_dataset = PG19SimpleDataset(dataset=[inputs])

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False, sampler=SequentialSampler(data_source=train_dataset),
        num_workers=0, collate_fn=lambda i: i[0], pin_memory=True
    )
    eval_dataloader = copy.deepcopy(train_dataloader)

    optimizer = create_optimizer(model=model, args=args)
    lr_scheduler = create_lr_scheduler(num_training_steps=num_training_steps, optimizer=optimizer, args=args)
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1,repetition_penalty=1.12,
        temperature=1.0, max_new_tokens=args.stride_size, min_new_tokens=args.stride_size,
        bad_words_ids=[[tokenizer.eos_token_id]],
        use_cache=True, return_dict_in_generate=True
    )
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    reuse_times = 0 
    for b_idx in tqdm(range(0, 50), total=50):
        if args.kv_reuse_times == 0:
            past_key_values = None
        if past_key_values is not None:
            reuse_times += 1

        batch, basic_input_ids, past_key_values = generate_once(
            generate_length=args.stride_size, 
            training_input_length=args.training_input_length,
            eval_input_length=args.eval_input_length,
            prev_input_ids=basic_input_ids,
            past_key_values=past_key_values
        )

        if args.kv_reuse_times != 0 and reuse_times != 0 and reuse_times % args.kv_reuse_times == 0:
            past_key_values = None
            reuse_times = 0

        model.train()
        for _ in range(0, 2):
            with accelerator.accumulate(model):
                outputs: CausalLMOutputWithPast = model(**batch)
                loss = outputs.loss 
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        lr_scheduler.step()

        
