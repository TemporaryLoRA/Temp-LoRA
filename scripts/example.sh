ACCELERATE_CONFIG=""
SAVE_DIR=""

MODEL_NAME="togethercomputer/LLaMA-2-7B-32K"

mkdir -p $SAVE_DIR
accelerate launch --config_file $ACCELERATE_CONFIG trainer/acc_complete_example_trainer.py --model_name $MODEL_NAME \
  --training_input_length 1024 \
  --eval_input_length 1024 \
  --kv_reuse_times 7 \
  --stride_size 1024 \
  --gradient_checkpointing "false" \
  --output_dir ""