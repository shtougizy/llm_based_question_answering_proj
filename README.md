# 拍照搜题辅助学习系统

基于 InternVL3.5-2B + Qwen3-1.7B + FAISS 的 K12 学习辅助系统。

## 功能
- 拍照识题（多模态）
- 题库检索（FAISS 向量检索）
- AI 解题（RAG + LLM）
- 错题本 + KMeans 聚类薄弱知识点分析
- 个性化练习推荐

## 环境要求
- Python 3.10+
- CUDA 12.x（可选）

## 安装
```bash
# 安装 PyTorch（CUDA 版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 安装 llama-cpp-python（CUDA 版）
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

## 模型配置
下载以下模型并在 config.py 中配置路径：
- InternVL3_5-2B（多模态识别）
- Qwen3-1.7B-Q8_0.gguf（文本推理）
- BAAI/bge-small-zh-v1.5（向量检索，自动下载）

## 启动
```bash
uvicorn backend.main:app --reload --port 8000
```

访问 http://localhost:8000
