# Inference-Time Training: With Greater Text Comes Greater Necessity

Long-text generation, such as novel writing or discourse-level translation with extremely long context, poses significant challenges to language models. Existing methods primarily focus on extending the model’s context window  through strategies such as length extrapolation. However, these methods require daunting hardware resources in both training and inference.

Our proposed method, Temp-Lora, offers an alternative idea. Rather than relying on the KV cache to store all context information, Temp-Lora embeds this information directly into the model’s parameters. During the long-text generation process, we employ a temporary Lora module, which is progressively trained using the previously-generated text. This method efficiently preserves contextual knowledge, and the module is subsequently discarded after generation to prevent a permanent impact on the model parameters.

Extensive experiments conducted on the PG19 language-modeling benchmark and the GuoFeng discourse-level translation benchmark demonstrate the efficacy of Temp-Lora. Our findings reveal that: 1) Temp-Lora significantly improves generation quality on long texts, evidenced by a 1.68 perplexity decrease and a 6.6 BLEU increase on GuoFeng, 2) TempLora is compatible and complementary to most existing long-text generation methods, and 3) Temp-Lora can significantly reduce computation cost

# Running

1. Write the path to the `configs/deepspeed_zero2. json` file to the `deepspeed_config_file` field of `accelerate_default_config.yaml` as follows:
``` 
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: <the path of deepspeed_zeros.json>
  zero3_init_flag: false
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 1
use_cpu: false
```

2. Complete the shell file corresponding to the training task

To experiment on the `PG-19` dataset, you need to complete the 'scripts/llama2.sh' file, for example:

```
ACCELERATE_CONFIG="<the path of configs/accelerate_default_config.yaml>"
DS_CONFIG="<the path of configs/deepspeed_zero2.json >"
SAVE_DIR="<saving results>"

MODEL_NAME="togethercomputer/LLaMA-2-7B-32K"

mkdir -p $SAVE_DIR
accelerate launch --config_file $ACCELERATE_CONFIG trainer/acc_pg19_trainer.py --model_name $MODEL_NAME \
  --train_fp "<such as  data/pg19_test/12204.txt>" \
  --eval_fp "<same as train_fp>" \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --learning_rate 5e-5 \
  --weight_decay 0 \
  --optim torch_adaw_fused \
  --lr_scheduler_type constant_with_warmup \
  --warmup_steps 2 \
  --output_dir $SAVE_DIR \
  --training_input_length 3072 \
  --stride_size 1024 \
  --eval_input_length 3072 \
  --num_train_epochs 2 \
  --gradient_checkpointing "false" \
  --use_flash_attention_2
```

for `GuoFeng`, you need to edit `scripts/01_6B_chat.sh`

```
ACCELERATE_CONFIG="<the path of configs/accelerate_default_config.yaml>"
DS_CONFIG="<the path of configs/deepspeed_zero2.json>"
SAVE_DIR="<>"

mkdir -p $SAVE_DIR

accelerate launch --config_file $ACCELERATE_CONFIG trainer/acc_guo_feng_trainer.py \
  --model_name "01-ai/Yi-6B-Chat" \
  --train_fp <such as data/cache_guofeng/102-bgwzsl.train>" \
  --eval_fp <such as data/cache_guofeng/102-bgwzsl.eval>" \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --learning_rate 5e-5 \
  --weight_decay 0 \
  --optim torch_adaw_fused \
  --lr_scheduler_type constant_with_warmup \
  --warmup_steps 2 \
  --output_dir $SAVE_DIR \
  --num_train_epochs 2 \
  --gradient_checkpointing false \
  --training_input_length 1 \   // 1024
  --eval_input_length 1         // 1024
  ```

1. run 

```
bash scripts/llama2.sh 
```

4. evaluate results

```
python3 eval_results/pg19.py --help
python3 eval_results/guo_feng.py --help
```

## Example and KV Cache Reuse

We provide a complete example of `generate chunk base inputs -> train chunk -> generate ...` and implement KV Cache Reuse. Using KV Cache Reuse can reduce training time, the example is in `trainer/acc_complete_example_trainer.py`.


## Environment 

We build the running environment based on [nvidia/cuda:11.8.0-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/11.8.0-cudnn8-devel-ubuntu22.04/images/sha256-3fbac875b9fd3059d554226246c1ade676fad88aee1ac2cb2deaa6e77e4606a5). The required python packages are detailed in the `requirements.txt` file. Several important packages are as follows:

```bash
use docker: nvidia/cuda:11.8.0-devel-ubuntu22.04
python version: 3.10.12
torch 2.2.1+cu118
torchaudio  2.2.1+cu118
torchvision 0.17.1+cu118
transformers 4.38.2
transformers-stream-generator 0.0.4
flash-attn 2.5.6
xformers 0.0.24+cu118
peft 0.9.0
accelerate 0.27.2
deepspeed 0.13.4
deepspeed-kernels 0.0.1.dev1698255861
datasets 2.18.0
```