"""
全局配置文件
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ============ 模型路径 ============
# InternVL3.5-4B：可填 HuggingFace model id 或本地绝对路径
INTERNVL_MODEL_PATH = os.getenv(
    "INTERNVL_MODEL_PATH",
    "/home/zsy/workspace/250606/code/work/260221_gd/data/models/InternVL3_5-2B"          # 替换为本地路径如 "/data/models/InternVL3-5-4B"
)

# Qwen3-1.7B GGUF 文件路径
QWEN_GGUF_PATH = os.getenv(
    "QWEN_GGUF_PATH",
    "/home/zsy/workspace/250606/code/work/260221_gd/data/models/qwen3-1.7b-q4_k_m.gguf/Qwen3-1.7B-Q8_0.gguf"

)

# ============ 数据路径 ============
QUESTION_BANK_PATH = str(BASE_DIR / "data" / "question_bank.json")
FAISS_INDEX_PATH = str(BASE_DIR / "data" / "faiss_index.bin")
FAISS_META_PATH = str(BASE_DIR / "data" / "faiss_meta.json")
SQLITE_DB_PATH = str(BASE_DIR / "data" / "app.db")
UPLOAD_DIR = str(BASE_DIR / "data" / "uploads")

# ============ 向量模型 ============
# 用于题目文本向量化的句子编码模型（轻量级，CPU 可运行）
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# ============ 服务参数 ============
MAX_RETRIEVE = 5        # RAG 检索最多返回题目数
LLM_MAX_TOKENS = 512   # LLM 单次最大生成 token 数
INTERNVL_MAX_NEW_TOKENS = 256

# ============ 设备 ============
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
