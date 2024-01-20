# Inference-Time Training: With Greater Text Comes Greater Necessity

Long-text generation, such as novel writing or discourse-level translation with extremely long context, poses significant challenges to language models. Existing methods primarily focus on extending the model’s context window  through strategies such as length extrapolation. However, these methods require daunting hardware resources in both training and inference.

Our proposed method, Temp-Lora, offers an alternative idea. Rather than relying on the KV cache to store all context information, Temp-Lora embeds this information directly into the model’s parameters. During the long-text generation process, we employ a temporary Lora module, which is progressively trained using the previously-generated text. This method efficiently preserves contextual knowledge, and the module is subsequently discarded after generation to prevent a permanent impact on the model parameters.

Extensive experiments conducted on the PG19 language-modeling benchmark and the GuoFeng discourse-level translation benchmark demonstrate the efficacy of Temp-Lora. Our findings reveal that: 1) Temp-Lora significantly improves generation quality on long texts, evidenced by a 1.68 perplexity decrease and a 6.6 BLEU increase on GuoFeng, 2) TempLora is compatible and complementary to most existing long-text generation methods, and 3) Temp-Lora can significantly reduce computation cost

# 代码运行

1. 将`configs/deepspeed_zero2.json`文件的路径写入 `accelerate_default_config.yaml` 的 `deepspeed_config_file` 字段，如下所示：

``` 
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: <deepspeed_zeros.json 文件的绝对路径>
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

2. 填写训练任务对应的shell文件

若要在 `PG-19` 数据集上进行实验，请填写 `scripts/llama2.sh` 文件，补全训练所用的配置文件、数据集等信息。

```
ACCELERATE_CONFIG="<configs/accelerate_default_config.yaml 文件地址>"
DS_CONFIG="<configs/deepspeed_zero2.json 文件地址>"
SAVE_DIR="<模型结果保存路径>"

MODEL_NAME="togethercomputer/LLaMA-2-7B-32K"

mkdir -p $SAVE_DIR
accelerate launch --config_file $ACCELERATE_CONFIG trainer/acc_pg19_trainer.py --model_name $MODEL_NAME \
  --train_fp "<训练所用的文件，如 data/pg19_test/12204.txt>" \
  --eval_fp "<与 train_fp 相同即可>" \
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

对应`GuoFeng`数据集则是`01_6B_chat.sh`

```
ACCELERATE_CONFIG="<configs/accelerate_default_config.yaml 文件地址>"
DS_CONFIG="<configs/deepspeed_zero2.json 文件地址>"
SAVE_DIR="<模型结果保存路径>"

mkdir -p $SAVE_DIR

accelerate launch --config_file $ACCELERATE_CONFIG trainer/acc_guo_feng_trainer.py \
  --model_name "01-ai/Yi-6B-Chat" \
  --train_fp "<训练所用的文件，如 data/cache_guofeng/102-bgwzsl.train>" \
  --eval_fp "<测试所用的文件，如 data/cache_guofeng/102-bgwzsl.eval>" \
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

3. 执行相关的sheel文件

```
bash scripts/llama2.sh 
```

4. 实验结果分析

```
python3 eval_results/pg19.py --help
python3 eval_results/guo_feng.py --help
```

## KV Cache Reuse

使用KV Cache Reuse可以减少训练时间，示例见`trainer/acc_complete_example_trainer.py`。
