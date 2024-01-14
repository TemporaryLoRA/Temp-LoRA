import sys
sys.path.append(".")

import json
import argparse
import os.path
import evaluate
from typing import Any, List, Dict, Tuple, Optional
from helper import load_json, load_txt, load_jsonline, write_jsonline, write_json

from comet import download_model, load_from_checkpoint


def get_file_tokens(fp: str) -> int:
    records = load_jsonline(fp=fp)
    return sum([i["new_tokens"] for i in records])


def get_file_ppl(fp: str, ranges: Tuple[int, int]) -> Tuple[float, float, int]:
    records = load_jsonline(fp=fp)
    start, end = ranges

    num_tokens, base_ppl, lora_ppl, range_tokens = 0, 0, 0, 0
    for r in records:
        num_tokens += r["new_tokens"]
        if num_tokens < start:
            continue
        if num_tokens >= end:
            break
        base_ppl += r["base_ppl"]
        lora_ppl += r["lora_ppl"]
        range_tokens += r["new_tokens"]
    return base_ppl, lora_ppl, range_tokens


def get_zh_text_from_prompt(prompt: str) -> str:
    assert prompt.startswith("<|im_start|>user\n请将下面的文本翻译成英文。\n")
    prompt = prompt.replace("<|im_start|>user\n请将下面的文本翻译成英文。\n", "").strip()
    pos = prompt.find("<|im_end|>")
    assert pos != -1
    return prompt[:pos].strip().split("\n")[-1].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dir_path, chunk_size, start, end = args.dir_path, args.chunk_size, args.start, args.end

    records = []
    for fname in os.listdir(dir_path):
        if not fname.endswith(".eval"):
            continue
        fp = os.path.join(dir_path, fname)
        num_tokens = get_file_tokens(fp=fp)
        records.append({"file": fname, "fp": fp, "num_tokens": get_file_tokens(fp=fp)})

    print("PPL by book length.")
    results = {}
    for i in range(start, end, chunk_size):
        left, right = i, i + chunk_size
        k = f"[{left}k,{right}k)"
        results[k] = {"base_ppl": 0, "lora_ppl": 0, "num_tokens": 0}
        for r_idx in range(0, len(records)):
            num_tokens = records[r_idx]["num_tokens"]
            if num_tokens < left * 1024 or num_tokens >= right * 1024:
                continue
            base_ppl, lora_ppl, num_tokens = get_file_ppl(fp=records[r_idx]["fp"], ranges=(0, right * 1024))
            results[k]["base_ppl"] += base_ppl
            results[k]["lora_ppl"] += lora_ppl
            results[k]["num_tokens"] += num_tokens

    print("Ranges: ", " ".join([k for k in records]))
    print("base ppl: ", " ".join([str(round(v["base_ppl"] / v["num_tokens"], 5)) for _, v in results.items()]))
    print("lora ppl: ", " ".join([str(round(v["lora_ppl"] / v["num_tokens"], 5)) for _, v in results.items()]))

    print("PPL by context length:")
    results = []
    for i in range(start, end, chunk_size):
        left, right = i, i + chunk_size
        k = f"[{left}k,{right}k)"
        results[k] = {"base_ppl": 0, "lora_ppl": 0, "num_tokens": 0}
        for r_idx in range(0, len(records)):
            base_ppl, lora_ppl, num_tokens = get_file_ppl(fp=records[r_idx]["fp"], ranges=(left * 1024, right * 1024))
            results[k]["base_ppl"] += base_ppl
            results[k]["lora_ppl"] += lora_ppl
            results[k]["num_tokens"] += num_tokens
    print("Ranges: ", " ".join([k for k in records]))
    print("base ppl: ", " ".join([str(round(v["base_ppl"] / v["num_tokens"], 5)) for _, v in results.items()]))
    print("lora ppl: ", " ".join([str(round(v["lora_ppl"] / v["num_tokens"], 5)) for _, v in results.items()]))

    eval_results = []
    for r in records:
        eval_results.extend(load_jsonline(fp=r["fp"]))
    for e_idx in range(0, len(eval_results)):
        eval_results[e_idx]["lora_prediction"] = eval_results[e_idx]["lora_prediction"].replace("<|im_end|>", "").strip()
        eval_results[e_idx]["base_prediction"] = eval_results[e_idx]["base_prediction"].replace("<|im_end|>", "").strip()
        eval_results[e_idx]["truth"] = eval_results[e_idx]["truth"].strip()

    print("Global PPL:")
    base_ppl, lora_ppl, num_tokens =0, 0, 0
    for r in eval_results:
        base_ppl += r["base_ppl"]
        lora_ppl += r["lora_ppl"]
        num_tokens += r["num_tokens"]
    print("base ppl: ", round(base_ppl / num_tokens, 5))
    print("lora ppl: ", round(lora_ppl / num_tokens, 5))

    print("Global BLEU:")
    bleu = evaluate.load("bleu")

    lora_predictions = []
    base_predictions = []
    references = []
    for r in eval_results:
        lora_predictions.append(r["lora_prediction"])
        base_predictions.append(r["base_prediction"])
        references.append(r["truth"])

    base_bleu = bleu.compute(predictions=base_predictions, references=references)
    lora_bleu = bleu.compute(predictions=lora_predictions, references=references)
    print("base bleu: ", base_bleu)
    print("lora bleu: ", lora_bleu)

    print("Global Comet:")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    lora_data = []
    base_data = []
    for r in eval_results:
        src = get_zh_text_from_prompt(prompt=r["prompt"])
        base_data.append({"src": src, "mt": r["base_prediction"], "ref": r["truth"]})
        lora_data.append({"src": src, "mt": r["lora_prediction"], "ref": r["truth"]})
    base_output = model.predict(samples=base_data, batch_size=128, gpus=1)
    lora_output = model.predict(samples=lora_data, batch_size=128, gpus=1)

    print("base comet: ", base_output.system_score)
    print("lora comet: ", lora_output.system_score)



