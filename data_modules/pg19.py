import sys
sys.path.append(".")

import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import TypedDict, List, Any, Dict, Optional

Instance = TypedDict("Instance", {
    "input_ids": torch.Tensor, "attention_mask": torch.Tensor, "labels": torch.Tensor,
    "serial_id": int, "new_tokens": int
})

SimpleInstance = TypedDict("SimpleInstance", {
    "serial_id": int, "new_tokens": int, "start": int, "middle": int, "end": int
})


class PG19SlowRawDataset:
    def __init__(self, fp: str, tokenizer: PreTrainedTokenizer, prefix_length: int, stride_size: int):
        self.tokenizer = tokenizer
        self.fp = fp
        self.prefix_length = prefix_length
        self.init_window_size = 1024
        self.stride_size = stride_size
        self.input_ids: Optional[torch.Tensor] = None
        self.raw_dataset: List[SimpleInstance] = []

    def load_from_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids
        self.raw_dataset.clear()
        input_length = input_ids.size(-1)

        self.raw_dataset.append(SimpleInstance(
            serial_id=0,
            new_tokens=min(input_length, self.stride_size),
            start=0,
            middle=0,
            end=min(input_length, self.init_window_size)
        ))
        if input_length < self.init_window_size:
            return
        serial_id = 1
        for i in range(self.init_window_size, input_length, self.stride_size):
            self.raw_dataset.append(SimpleInstance(
                serial_id=serial_id,
                new_tokens=min(i + self.stride_size, input_length) - i,
                start=max(0, i - self.prefix_length),
                middle=i,
                end=min(i + self.stride_size, input_length)
            ))
            serial_id += 1
            if i + self.stride_size >= input_length:
                break

class PG19Dataset(Dataset):
    def __init__(self, dataset: PG19SlowRawDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ins: SimpleInstance = self.dataset.raw_dataset[idx]
        start, middle, end = ins["start"], ins["middle"], ins["end"]

        if middle == start:
            input_ids = self.dataset.input_ids[:, start: end]
            attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids, "labels": input_ids, "attention_mask": attention_mask,
                "serial_id": ins["serial_id"], "new_tokens": ins["new_tokens"]
            }
        context_input_ids = self.dataset.input_ids[:, start: middle]
        context_labels = torch.zeros_like(input=context_input_ids) - 100
        inference_input_ids = self.dataset.input_ids[:, middle: end]

        window_input_ids = torch.cat(tensors=[context_input_ids, inference_input_ids], dim=-1)
        return {
            "input_ids": window_input_ids,
            "attention_mask": torch.ones_like(window_input_ids),
            "labels": torch.cat(tensors=[context_labels, inference_input_ids], dim=-1),
            "serial_id": ins["serial_id"],
            "new_tokens": ins["new_tokens"]
        }

