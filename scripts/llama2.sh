ACCELERATE_CONFIG=""
SAVE_DIR=""

MODEL_NAME="togethercomputer/LLaMA-2-7B-32K"

mkdir -p $SAVE_DIR
accelerate launch --config_file $ACCELERATE_CONFIG trainer/acc_pg19_trainer.py --model_name $MODEL_NAME \
  --train_fp "" \
  --eval_fp "" \
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
  --eval_input_length 31744 \
  --num_train_epochs 2 \
  --gradient_checkpointing "false" 