import sys
sys.path.append(".")

import argparse
from tqdm import tqdm
import os, gc, sys, copy, json

sys.path.append("../")

import torch
from torch.utils.data import DataLoader, SequentialSampler

from peft import LoraConfig, TaskType, get_peft_model

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig,
    StoppingCriteria, StoppingCriteriaList
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Tuple, TypedDict, Iterable, Optional

from data_modules.guo_feng import get_dataset, GuoFengDataset
from helper import (
    create_lr_scheduler, create_optimizer, write_json, find_all_linear_names, load_json, load_txt, load_jsonline,
    write_jsonline, json_stringify
)

logger = get_logger(name=__name__)


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self):
        super().__init__()
        self.stop_token_ids = [tokenizer.encode(i)[0] for i in ["<|im_end|>", "\n"]]
        self.stop_sentences: Optional[List[bool]] = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.stop_sentences is None:
            self.stop_sentences = [False] * input_ids.size(0)

        for i in range(0, input_ids.size(0)):
            g_id = input_ids[i][-1].item()
            if g_id in self.stop_token_ids:
                self.stop_sentences[i] = True

        if all(self.stop_sentences):
            self.stop_sentences = None
            return True
        return False


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
    num_train_epochs: int
    training_input_length: int
    eval_input_length: int

    use_flash_attention_2: bool = field(default=True)
    rope_name: str = field(default="")
    rope_factor: float = field(default=1.0)
    gradient_checkpointing: str = field(default="")


MTPair = TypedDict("MTPair", {"zh": str, "en": str})
Chapter = TypedDict("Chapter", {"id": str, "pairs": List[MTPair]})
Book = TypedDict("Book", {"id": str, "chapters": List[Chapter]})
Inputs = TypedDict("Inputs", {"input_ids": torch.Tensor, "labels": torch.Tensor, "attention_mask": torch.Tensor})
EvalData = TypedDict("EvalData", {
    "prompt": str, "truth": str, "prediction": str, "ppl": float, "new_tokens": int,
    "base_prediction": str, "lora_prediction": str, "base_ppl": float, "lora_ppl": float
})
EvalOutput = TypedDict("EvalOutput", {"base_ppl": float, "lora_ppl": float, "new_tokens": int})
Instance = TypedDict("Instance", {"contexts": Dict[str, List[MTPair]], "inference": List[MTPair]})


def to_device(obj: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device) for k, v in obj.items()}


def batchify_for_train(ins: Instance, cache: Dict[str, List[int]], k: int) -> Inputs:
    prefix_input_ids = copy.deepcopy(prompt_prefix_input_ids)
    middle_input_ids = copy.deepcopy(prompt_middle_input_ids)

    prefix_labels = [-100] * len(prefix_input_ids)
    middle_labels = [-100] * len(middle_input_ids)

    prefix_content = prompt_prefix
    middle_content = prompt_middle

    for p_idx, p in enumerate(ins["contexts"][f"k{k}"]):
        p: MTPair
        zh = p["zh"] + "\n"
        en = p["en"] + "\n"
        prefix_content += zh
        middle_content += en

        if zh not in cache:
            cache[zh] = tokenizer.encode(zh)
        if en not in cache:
            cache[en] = tokenizer.encode(en)
        zh_input_ids, en_input_ids = cache[zh], cache[en]

        prefix_input_ids += zh_input_ids
        prefix_labels += [-100] * len(zh_input_ids)
        middle_input_ids += en_input_ids
        middle_labels += [-100] * len(en_input_ids)

    for p_idx, p in enumerate(ins["inference"]):
        p: MTPair
        zh = p["zh"] + "\n" if p_idx != len(ins["inference"]) - 1 else p["zh"]
        en = p["en"] + "\n" if p_idx != len(ins["inference"]) - 1 else p["en"]

        prefix_content += zh
        middle_content += en
        if zh not in cache:
            cache[zh] = tokenizer.encode(zh)
        if en not in cache:
            cache[en] = tokenizer.encode(en)
        zh_input_ids, en_input_ids = cache[zh], cache[en]

        prefix_input_ids += zh_input_ids
        prefix_labels += [-100] * len(zh_input_ids)

        middle_input_ids += en_input_ids
        middle_labels += en_input_ids

    input_ids = prefix_input_ids + middle_input_ids + [im_end_token_id]
    labels = prefix_labels + middle_labels + [im_end_token_id]
    content = prefix_content + middle_content + "<|im_end|>"

    write_jsonline(
        fp=train_record_fp, obj=[{"content": content, "inputs": {"input_ids": input_ids, "labels": labels}}], mode="a"
    )

    input_ids = torch.tensor(data=[input_ids], dtype=torch.int64, device=model.device)
    labels = torch.tensor(data=[labels], dtype=torch.int64, device=model.device)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}


def batchify_for_eval(ins: Instance, k: int) -> Iterable[EvalData]:
    prefix_content = prompt_prefix
    middle_content = prompt_middle

    for p_idx, p in enumerate(ins["contexts"][f"k{k}"]):
        p: MTPair
        zh, en = p["zh"] + "\n", p["en"] + "\n"
        prefix_content += zh
        middle_content += en

    for p_idx, p in enumerate(ins["inference"]):
        p: MTPair
        yield EvalData(
            prompt=prefix_content + p["zh"] + middle_content, truth=p["en"], prediction="", ppl=0,
            new_tokens=0, base_prediction="", lora_prediction="", base_ppl=0, lora_ppl=0
        )
        zh = p["zh"] + "\n" if p_idx != len(ins["inference"]) - 1 else p["zh"]
        en = p["en"] + "\n" if p_idx != len(ins["inference"]) - 1 else p["en"]

        prefix_content += zh
        middle_content += en


def train_once(model: PreTrainedModel, ins: Instance, cache: Dict[str, List[int]], k: int) -> Tuple[float, float]:
    model.train()
    with accelerator.accumulate(model):
        batch = batchify_for_train(ins=ins, cache=cache, k=k)
        outputs: CausalLMOutputWithPast = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
        )
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    return torch.exp(loss).item(), loss.item()


@torch.no_grad()
def inference_once(model: PreTrainedModel, ins: Instance, disable_lora: bool, k: int) -> Iterable[EvalData]:
    model.eval()
    for data in batchify_for_eval(ins=ins, k=k):
        data: EvalData
        input_ids = tokenizer.encode_plus(data["prompt"], return_tensors="pt")["input_ids"].to(model.device)
        prompt_input_ids = copy.deepcopy(input_ids)
        input_length = input_ids.size(-1)
        if disable_lora:
            with accelerator.unwrap_model(model=model).disable_adapter():
                generated = model.generate(
                    inputs=input_ids, generation_config=generation_config, stopping_criteria=stopping_criteria
                )
        else:
            generated = model.generate(
                inputs=input_ids, generation_config=generation_config, stopping_criteria=stopping_criteria
            )
        prediction = tokenizer.decode(generated[0][input_length:], skip_special_tokens=False)
        data["prediction"] = prediction

        prompt_labels = torch.zeros_like(input=prompt_input_ids) - 100
        truth_input_ids: torch.Tensor = tokenizer.encode_plus(
            data["truth"], return_tensors="pt"
        )["input_ids"].to(model.device)
        data["new_tokens"] = truth_input_ids.size(-1)

        input_ids = torch.cat(tensors=[prompt_input_ids, truth_input_ids], dim=-1).to(torch.int64)
        labels = torch.cat(tensors=[prompt_labels, truth_input_ids], dim=-1).to(torch.int64)
        attention_mask = torch.ones_like(input=input_ids)

        if disable_lora:
            with accelerator.unwrap_model(model=model).disable_adapter():
                outputs: CausalLMOutputWithPast = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
        else:
            outputs: CausalLMOutputWithPast = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        data["ppl"] = torch.exp(outputs.loss).item() * data["new_tokens"]
        yield data


@torch.no_grad()
def evaluate_once(model: PreTrainedModel, ins: Instance, k: int) -> EvalOutput:
    base_results = inference_once(model=model, ins=ins, disable_lora=True, k=k)
    lora_results = inference_once(model=model, ins=ins, disable_lora=False, k=k)

    output = EvalOutput(base_ppl=0, lora_ppl=0, new_tokens=0)
    for base, lora in zip(base_results, lora_results):
        base: EvalData
        lora: EvalData
        assert all([base[k] == lora[k] for k in ["new_tokens", "prompt", "truth"]])
        r = {
            "prompt": base["prompt"], "truth": base["truth"], "new_tokens": base["new_tokens"],
            "base_ppl": base["ppl"], 
            "mean_base_ppl": base["ppl"] / base["new_tokens"],
            "base_prediction": base["prediction"],
            "lora_ppl": lora["ppl"], 
            "mean_lora_ppl": lora["ppl"] / lora["new_tokens"],
            "lora_prediction": lora["prediction"]
        }
        output["base_ppl"] += r["base_ppl"]
        output["lora_ppl"] += r["lora_ppl"]
        output["new_tokens"] += r["new_tokens"]
        write_jsonline(fp=eval_record_fp, obj=[r], mode="a")
    return output


def parse_args() -> TrainingArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_fp", type=str)
    parser.add_argument("--eval_fp", type=str)
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--gradient_checkpointing", type=str)
    parser.add_argument("--training_input_length", type=int, default=1024)
    parser.add_argument("--eval_input_length", type=int, default=1)
    parser.add_argument("--stride_size", type=int)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    args = parser.parse_args()

    obj = {}
    for f in fields(TrainingArguments):
        v = getattr(args, f.name, None)
        if v is not None:
            obj[f.name] = v
    return TrainingArguments(**obj)


if __name__ == '__main__':
    args: TrainingArguments = parse_args()

    accelerator = Accelerator(log_with="tensorboard", project_dir=args.output_dir, gradient_accumulation_steps=1)

    accelerator.wait_for_everyone()
    set_seed(42)

    fname = args.train_fp.split("/")[-1].split(".")[0]
    train_record_fp = os.path.join(args.output_dir, fname + ".train")
    eval_record_fp = os.path.join(args.output_dir, fname + ".eval")

    if accelerator.is_local_main_process:
        os.system(f"mkdir -p {args.output_dir}")
        if os.path.exists(train_record_fp):
            os.system(f"rm {train_record_fp}")
        if os.path.exists(eval_record_fp):
            os.system(f"rm {eval_record_fp}")

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
        model.gradient_checkpointing_enable()

    prompt_prefix = "<|im_start|>user\n请将下面的文本翻译成英文。\n"
    prompt_middle = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_prefix_input_ids = tokenizer.encode(prompt_prefix)
    prompt_middle_input_ids = tokenizer.encode(prompt_middle)

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

    train_dataset = get_dataset(fp=args.train_fp, tokenizer=tokenizer)
    eval_dataset = get_dataset(fp=args.eval_fp, tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False, sampler=SequentialSampler(data_source=train_dataset),
        num_workers=0, collate_fn=lambda i: i[0], pin_memory=True
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset, batch_size=1, shuffle=False, sampler=SequentialSampler(data_source=eval_dataset),
        num_workers=0, collate_fn=lambda i: i[0], pin_memory=True
    )
    num_training_steps = len(train_dataloader)

    optimizer = create_optimizer(model=model, args=args)
    lr_scheduler = create_lr_scheduler(num_training_steps=num_training_steps, optimizer=optimizer, args=args)
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    generation_config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.12, max_new_tokens=256)
    stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria()])
    im_end_token_id = tokenizer.encode("<|im_end|>")[0]

    pbr = tqdm(range(0, num_training_steps), disable=not accelerator.is_local_main_process)
    total_tokens, base_ppl, lora_ppl = 0, 0, 0
    cache: Dict[str, List[int]] = {}

    for b_idx, batch in enumerate(train_dataloader):
        batch: Instance
        outputs = evaluate_once(model=model, ins=batch, k=args.eval_input_length)
        base_ppl += outputs["base_ppl"]
        lora_ppl += outputs["lora_ppl"]
        total_tokens += outputs["new_tokens"]

        for _ in range(0, args.num_train_epochs):
            _, step_lora_loss = train_once(model=model, ins=batch, cache=cache, k=args.training_input_length)

        lr_scheduler.step()
        pbr.update(n=1)
        print(
            f"Batch idx: {b_idx} / {num_training_steps} | base ppl: {base_ppl / total_tokens} | lora ppl: {lora_ppl / total_tokens}"
        )
        if b_idx % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    pbr.close()

    write_jsonline(fp=train_record_fp, obj=["done"], mode="a")
    write_jsonline(fp=eval_record_fp, obj=["done"], mode="a")
