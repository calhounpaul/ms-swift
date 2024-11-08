#MODEL_TYPE=llama3_2-1b-instruct
#MODEL_TYPE=llama3_2-1b
MODEL_TYPE=llama3_2-3b
#MODEL_TYPE=llama3_1-8b
#MODEL_TYPE=qwen2_5-0_5b
#MODEL_TYPE=qwen2_5-32b

FT_TYPE=lora

nproc_per_node=2
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    /ms-swift/examples/pytorch/llm/llm_sft.py \
    --model_type $MODEL_TYPE \
    --sft_type $FT_TYPE \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --dataset "/ms-swift/docker/workdir/datasets/swift_train.json" \
    --num_train_epochs 1 \
    --truncation_strategy truncation_left \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --quantization_bit 8 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing false \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 4 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --save_total_limit 2 \
    --logging_steps 10

#    --deepspeed default-zero2 \
#    --eval_steps 100 \
#    --save_steps 100 \
