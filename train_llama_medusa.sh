export PYTHONPATH=$PYTHONPATH:./ # 加入当前目录到PYTHONPATH
#tangledlabs/tangled-llama-pints-1.5b-v0.2-instruct
#model_name_or_path=tangledlabs/tangled-llama-pints-1.5b-v0.2-instruct

# 设置随机端口（避免冲突）
export MASTER_PORT=$((29500 + RANDOM % 100))
export MASTER_ADDR=localhost
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

model_name_or_path="/home/hkx/data/work/hf_data_and_model/models/MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA"

#    --per_device_eval_batch_size 1 \
#--data_path data/AdvertiseGenChatML/dev_min.jsonl \
#--eval_data_path data/AdvertiseGenChatML/dev_min.jsonl \

torchrun --standalone --nproc_per_node=1 medusa/train/train_medusa.py  \
    --model_name_or_path $model_name_or_path  \
    --data_path data/AdvertiseGenChatML/dev.jsonl \
    --bf16 True \
    --output_dir llama_medusa_output2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 256 \
    --lazy_preprocess True \
    --medusa_num_heads 2 \
    --medusa_num_layers 1
