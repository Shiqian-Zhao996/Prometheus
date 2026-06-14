export MODEL_NAME="ShuttleDiffusion"
export DATASET="DALLEPrompt"

CUDA_VISIBLE_DEVICES=0 accelerate launch main.py \
  --oracle=$MODEL_NAME \
  --dataset=$DATASET \
  --max_budget=200
