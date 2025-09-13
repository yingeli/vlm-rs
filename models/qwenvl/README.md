export MODEL_NAME="Qwen2-VL-2B-Instruct"
git clone https://huggingface.co/Qwen/${MODEL_NAME} hf/${MODEL_NAME}

python3 /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir=hf/${MODEL_NAME} \
    --output_dir=checkpoints/${MODEL_NAME}/bf16-1gpu

trtllm-build --checkpoint_dir checkpoints/${MODEL_NAME}/bf16-1gpu \
    --output_dir engines/${MODEL_NAME}/bf16-1gpu/llm \
    --max_batch_size=4 \
    --max_input_len=2048 \
    --max_seq_len=3072 \
    --max_multimodal_len=1296 #(max_batch_size) * 324 (num_visual_features), this's for image_shape=[504,504]