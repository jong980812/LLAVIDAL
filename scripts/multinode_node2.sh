#!/bin/bash
#SBATCH -J LLAVIDAL
#SBATCH -p batch
#SBATCH --gres=gpu:8              # 노드당 GPU 8개
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=40G
#SBATCH --time=14-00:00:0
#SBATCH -o logs/16GPUS_zero2_LLAVIADAL_%x_%j.out
#SBATCH -e logs/16GPUS_zero2_LLAVIADAL_%x_%j.err

# set -euo pipefail
mkdir -p logs
# Master node & port 설정
export HF_HOME="/data/dataset/LLaVA-Video-100K-Subset/",
export PYTHONPATH="./:$PYTHONPATH"
export WANDB_PROJECT="LLAVIDAL"
export WANDB_API_KEY="b7d30964c576402ad464f5497787a77e381d8e39"
export WANDB_NAME="LLAVIDAL_videe-text_3epochs_multinode16gpus_zero2"



export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# NCCL 통신용 네트워크 인터페이스 설정
socket_ifname=$(cat /etc/hosts | grep $(hostname) | grep -Eo 'en\w+')
export NCCL_SOCKET_IFNAME=$socket_ifname

# DeepSpeed 실행
# deepspeed \
#   --num_nodes $NNODES \
#   --num_gpus $GPUS_PER_NODE \
#   --master_addr $MASTER_ADDR \
#   --master_port $MASTER_PORT \


export MASTER_ADDR=vll3   
export MASTER_PORT=30000
export NNODES=2
export NODE_RANK=1
export GPUS_PER_NODE=8
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
#--num_nodes $NNODES --num_gpus $GPUS_PER_NODE --node_rank $NODE_RANK --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  llavidal/train/train_mem.py \
  --deepspeed scripts/zero2.json \
  --version v1 \
  --dataloader_num_workers 12 \
  --dataloader_pin_memory False \
  --tune_mm_mlp_adapter True \
  --mm_use_vid_start_end \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy no \
  --report_to wandb \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 100 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --output_dir ./work_dirs/${WANDB_NAME} \
  --model_name_or_path mmaaz60/LLaVA-7B-Lightening-v1-1 \
  --data_path /data/dataset/ADL-X/instruction_data/NTU_QA-for-training.json \
  --video_folder /local_datasets/ADL-X/data/vidlab_datasets/NTU_combination_video_features \
  --object_folder /local_datasets/ADL-X/data/users/rchakra6/concatenated_objects \
  --pose_folder /local_datasets/ADL-X/data/users/rchakra6/concatenated_poses
