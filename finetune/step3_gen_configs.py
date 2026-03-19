"""
Step 3: 生成 LLaMA-Factory 所需配置文件
"""
import json
import os

BASE_DIR    = '/home/zsy/workspace/250606/code/work/260221_gd'
FINETUNE_DIR= os.path.join(BASE_DIR, 'finetune')
LF_DIR      = os.path.join(BASE_DIR, 'LLaMA-Factory')
MODEL_PATH  = os.path.join(BASE_DIR, 'data/models/qwen3-1.7b-q4_k_m.gguf')

# ==================== 1. dataset_info.json ====================
# 注册自定义数据集到 LLaMA-Factory
dataset_info = {
    "k12_train": {
        "file_name": os.path.join(FINETUNE_DIR, "train.json"),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    },
    "k12_val": {
        "file_name": os.path.join(FINETUNE_DIR, "val.json"),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
}

dataset_info_path = os.path.join(LF_DIR, 'data', 'dataset_info.json')
# 读取已有的，追加进去
if os.path.exists(dataset_info_path):
    with open(dataset_info_path, 'r') as f:
        existing = json.load(f)
    existing.update(dataset_info)
    with open(dataset_info_path, 'w') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
else:
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
print(f"dataset_info.json 已更新: {dataset_info_path}")

# ==================== 2. 训练配置 YAML ====================
# Qwen3-1.7B HuggingFace 格式（非 GGUF）需要单独下载
# 这里配置指向 HF 格式模型路径
HF_MODEL_PATH = os.path.join(BASE_DIR, 'data/models/Qwen3-1.7B-HF')

train_yaml = f"""### 模型配置
model_name_or_path: Qwen/Qwen3-1.7B  # 首次运行自动下载，或改为本地路径: {HF_MODEL_PATH}
trust_remote_code: true

### 微调方法
stage: sft                    # Supervised Fine-Tuning
do_train: true
finetuning_type: lora         # 使用 LoRA

### LoRA 超参数
lora_rank: 16                 # LoRA 秩，8GB显存下建议16
lora_alpha: 32                # 通常为 rank 的2倍
lora_dropout: 0.1
lora_target: q_proj,v_proj,k_proj,o_proj   # 对注意力层做 LoRA

### 8-bit 量化
quantization_bit: 8           # 8-bit 量化降低显存占用
quantization_method: bitsandbytes

### 数据集
dataset: k12_train
val_size: 0.0                 # 已单独划分验证集
max_samples: 50000            # 最多使用样本数
cutoff_len: 1024              # 最大序列长度
preprocessing_num_workers: 4

### 训练超参数
per_device_train_batch_size: 2     # 8GB显存下batch_size=2
gradient_accumulation_steps: 8    # 等效batch=16
num_train_epochs: 3
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
max_grad_norm: 1.0

### 优化器
optim: adamw_torch
weight_decay: 0.01
adam_beta1: 0.9
adam_beta2: 0.999

### 保存与评估
output_dir: {FINETUNE_DIR}/output/qwen3-1.7b-k12-lora
logging_dir: {FINETUNE_DIR}/output/logs
save_strategy: epoch
eval_strategy: no
logging_steps: 50
save_total_limit: 3
load_best_model_at_end: false

### 其他
bf16: false                   # RTX 4060 支持 bf16，可改为 true 提速
fp16: true
dataloader_num_workers: 2
report_to: none               # 不上报到 wandb（改为 wandb 可开启可视化）
"""

train_yaml_path = os.path.join(FINETUNE_DIR, 'qwen3_k12_lora.yaml')
with open(train_yaml_path, 'w', encoding='utf-8') as f:
    f.write(train_yaml)
print(f"训练配置已保存: {train_yaml_path}")

# ==================== 3. 评估配置 YAML ====================
eval_yaml = f"""model_name_or_path: Qwen/Qwen3-1.7B
adapter_name_or_path: {FINETUNE_DIR}/output/qwen3-1.7b-k12-lora
trust_remote_code: true
stage: sft
do_predict: true
finetuning_type: lora
quantization_bit: 8
quantization_method: bitsandbytes
dataset: k12_val
cutoff_len: 1024
per_device_eval_batch_size: 2
predict_with_generate: true
max_new_tokens: 512
output_dir: {FINETUNE_DIR}/output/eval_results
"""

eval_yaml_path = os.path.join(FINETUNE_DIR, 'qwen3_k12_eval.yaml')
with open(eval_yaml_path, 'w', encoding='utf-8') as f:
    f.write(eval_yaml)
print(f"评估配置已保存: {eval_yaml_path}")

# ==================== 4. 合并 LoRA 权重配置 ====================
merge_yaml = f"""model_name_or_path: Qwen/Qwen3-1.7B
adapter_name_or_path: {FINETUNE_DIR}/output/qwen3-1.7b-k12-lora
trust_remote_code: true
finetuning_type: lora
export_dir: {FINETUNE_DIR}/output/qwen3-1.7b-k12-merged
export_size: 2
export_legacy_format: false
"""

merge_yaml_path = os.path.join(FINETUNE_DIR, 'qwen3_k12_merge.yaml')
with open(merge_yaml_path, 'w', encoding='utf-8') as f:
    f.write(merge_yaml)
print(f"合并配置已保存: {merge_yaml_path}")

os.makedirs(os.path.join(FINETUNE_DIR, 'output'), exist_ok=True)
print("\n所有配置文件生成完成")
print(f"\n目录结构:")
print(f"  {FINETUNE_DIR}/")
print(f"  ├── cleaned_data.json      # 清洗后原始数据")
print(f"  ├── train.json             # 训练集 (Alpaca格式)")
print(f"  ├── val.json               # 验证集")
print(f"  ├── test.json              # 测试集")
print(f"  ├── test_with_meta.json    # 测试集含任务标签")
print(f"  ├── qwen3_k12_lora.yaml    # 训练配置")
print(f"  ├── qwen3_k12_eval.yaml    # 评估配置")
print(f"  ├── qwen3_k12_merge.yaml   # 合并配置")
print(f"  └── output/                # 训练输出目录")
