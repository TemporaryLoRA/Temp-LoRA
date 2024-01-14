import sys
sys.path.append(".")

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from typing import TypedDict, List, Dict

from helper import load_json

MTPair = TypedDict("MTPair", {"zh": str, "en": str})
Instance = TypedDict("Instance", {"contexts": Dict[str, List[MTPair]], "inference": List[MTPair]})


class GuoFengDataset(Dataset):
    def __init__(self, fp: str, tokenizer: PreTrainedTokenizer):
        self.raw_datasets = load_json(fp=fp)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_datasets)

    def __getitem__(self, idx: int) -> Instance:
        return self.raw_datasets[idx]


def get_dataset(fp: str, tokenizer: PreTrainedTokenizer):
    dataset = GuoFengDataset(fp=fp, tokenizer=tokenizer)
    return dataset
