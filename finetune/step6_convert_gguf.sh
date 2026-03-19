#!/bin/bash
# Step 6: 合并 LoRA 权重并转换为 GGUF 格式
# 使得微调后的模型可以直接替换原有的 llama.cpp 模型

set -e
BASE_DIR="/home/zsy/workspace/250606/code/work/260221_gd"
FINETUNE_DIR="$BASE_DIR/finetune"
LF_DIR="$BASE_DIR/LLaMA-Factory"
MERGED_DIR="$FINETUNE_DIR/output/qwen3-1.7b-k12-merged"
GGUF_DIR="$FINETUNE_DIR/output/gguf"

echo "=============================="
echo " Step 6: LoRA 合并 + GGUF 转换"
echo "=============================="

# 6.1 合并 LoRA 权重到 HF 格式
echo ""
echo "[6.1] 合并 LoRA 权重..."
cd "$LF_DIR"
python src/llamafactory/train/tuner.py \
    --config "$FINETUNE_DIR/qwen3_k12_merge.yaml" \
    --stage sft \
    --do_train false \
    --export_dir "$MERGED_DIR" \
    || llamafactory-cli export "$FINETUNE_DIR/qwen3_k12_merge.yaml"

echo "合并完成: $MERGED_DIR"

# 6.2 安装 llama.cpp（用于 GGUF 转换）
echo ""
echo "[6.2] 检查 llama.cpp..."
LLAMA_CPP_DIR="$BASE_DIR/llama.cpp"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    pip install -r "$LLAMA_CPP_DIR/requirements.txt" --break-system-packages
fi

# 6.3 转换为 GGUF
echo ""
echo "[6.3] 转换为 GGUF 格式..."
mkdir -p "$GGUF_DIR"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_DIR/qwen3-1.7b-k12-f16.gguf" \
    --outtype f16

echo "F16 GGUF 已生成: $GGUF_DIR/qwen3-1.7b-k12-f16.gguf"

# 6.4 量化为 Q8_0（与原模型格式一致）
echo ""
echo "[6.4] 量化为 Q8_0..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$GGUF_DIR/qwen3-1.7b-k12-f16.gguf" \
    "$GGUF_DIR/qwen3-1.7b-k12-Q8_0.gguf" \
    Q8_0

echo ""
echo "=============================="
echo " 转换完成！"
echo " GGUF 路径: $GGUF_DIR/qwen3-1.7b-k12-Q8_0.gguf"
echo ""
echo " 要使用微调模型，在 config.py 中修改："
echo " QWEN_GGUF_PATH = '$GGUF_DIR/qwen3-1.7b-k12-Q8_0.gguf'"
echo "=============================="
