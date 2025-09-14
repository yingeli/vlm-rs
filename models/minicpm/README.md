export MODEL_NAME="MiniCPM-V-4_5"

git clone https://huggingface.co/openbmb/${MODEL_NAME} hf/${MODEL_NAME}

python3 /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir=hf/${MODEL_NAME} \
    --output_dir=checkpoints/${MODEL_NAME}/bf16-1gpu
