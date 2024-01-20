ACCELERATE_CONFIG=""
SAVE_DIR=""

mkdir -p $SAVE_DIR

accelerate launch --config_file $ACCELERATE_CONFIG trainer/acc_guo_feng_trainer.py \
  --model_name "01-ai/Yi-6B-Chat" \
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
  --num_train_epochs 2 \
  --gradient_checkpointing "false" \
  --training_input_length 1 \
  --eval_input_length 1