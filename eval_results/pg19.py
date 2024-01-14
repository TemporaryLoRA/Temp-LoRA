import sys
sys.path.append(".")

import json
import argparse
import os.path
from typing import Any, List, Dict, Tuple, Optional

from helper import load_json, load_txt, load_jsonline, write_jsonline, write_json, json_stringify

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--eval_input_length", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dir_path, chunk_size, start, end, eval_input_length = args.dir_path, args.chunk_size, args.start, args.end, args.eval_input_length

    results = []
    for fname in os.listdir(dir_path):
        fp = os.path.join(dir_path, fname)
        results.append(load_json(fp)[0])

    print("PPL by document length.")
    records = {}
    for i in range(start, end, chunk_size):
        left, right = i, i + chunk_size
        key = f"[{left}k,{right}k)"
        records[key] = {"base_ppl": 0, "num_tokens": 0, "lora_ppl": 0}
        for r in results:
            if left * 1024 <= r["num_tokens"] < right * 1024:
                records[key]["num_tokens"] += r["num_tokens"]
                records[key]["lora_ppl"] += r["lora_ppl"]
                records[key]["base_ppl"] += r["base_ppl"]

    print("Ranges: ", " ".join([k for k in records]))
    print("base ppl: ", " ".join([str(round(v["base_ppl"] / v["num_tokens"], 5)) for _, v in records.items()]))
    print("lora ppl: ", " ".join([str(round(v["lora_ppl"] / v["num_tokens"], 5)) for _, v in records.items()]))

    print("PPL by context length.")
    records = {}
    for i in range(start, end, chunk_size):
        left, right = i, i + chunk_size
        key = f"[{left}k,{right}k)"
        for r in results:
            for s_idx, s in enumerate(r["steps"]):
                if s["eval_step_record"][eval_input_length]["total_tokens"] >= right * 1024:
                    break
                if s["eval_step_record"][eval_input_length]["total_tokens"] < left * 1024:
                    continue
                if s_idx == 0:
                    new_tokens = s["eval_step_record"][eval_input_length]["total_tokens"]
                    new_base_ppl = s["eval_step_record"][eval_input_length]["base_ppl"]
                    new_lora_ppl = s["eval_step_record"][eval_input_length]["lora_ppl"]
                else:
                    prev = r["steps"][s_idx - 1]["eval_step_record"][eval_input_length]
                    new_tokens = s["eval_step_record"][eval_input_length]["total_tokens"] - prev["total_tokens"]
                    new_base_ppl = s["eval_step_record"][eval_input_length]["base_ppl"] - prev["base_ppl"]
                    new_lora_ppl = s["eval_step_record"][eval_input_length]["lora_ppl"] - prev["lora_ppl"]
                records[key]["num_tokens"] += new_tokens
                records[key]["base_ppl"] += new_base_ppl
                records[key]["lora_ppl"] += new_lora_ppl

    print("Ranges: ", " ".join([k for k in records]))
    print("base ppl: ", " ".join([str(round(v["base_ppl"] / v["num_tokens"], 5)) for _, v in records.items()]))
    print("lora ppl: ", " ".join([str(round(v["lora_ppl"] / v["num_tokens"], 5)) for _, v in records.items()]))
