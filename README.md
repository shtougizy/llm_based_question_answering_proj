# 拍照搜题辅助学习系统

基于多模态大模型 + RAG 检索增强生成 + KMeans 聚类分析的 K12 智能学习辅助系统。

---

## 项目架构

```
260221_gd/
├── config.py                      # 全局配置（模型路径、设备、数据库路径等）
├── requirements.txt               # Python 依赖
│
├── backend/
│   ├── main.py                    # FastAPI 后端主程序，所有 HTTP 接口
│   └── init_db.py                 # 数据库初始化脚本
│
├── core/
│   ├── multimodal.py              # 多模态识别模块（InternVL3.5-2B）
│   ├── retrieval.py               # 向量检索模块（FAISS + BGE）
│   ├── llm.py                     # 语言模型推理模块（Qwen3-1.7B-GGUF）
│   ├── database.py                # 数据库 ORM 与操作（SQLAlchemy + SQLite）
│   └── analysis.py                # KMeans 聚类分析与练习推荐模块
│
├── frontend/
│   └── templates/
│       ├── index.html             # 首页（拍照/文字搜题）
│       ├── history.html           # 历史记录页
│       └── wrong_book.html        # 错题本页
│
└── data/
    ├── faiss_index.bin            # FAISS 向量索引（题库）
    ├── faiss_meta.json            # 题库元数据（题目、答案、知识点等）
    ├── question_bank.json         # 原始题库 JSON
    └── app.db                     # SQLite 数据库（运行时生成，不提交）
```

---

## 技术栈

| 模块 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 多模态识别 | InternVL3.5-2B（图文理解） |
| 语言模型 | Qwen3-1.7B-Q8_0（GGUF 量化，llama-cpp-python） |
| 向量检索 | FAISS + BGE-small-zh-v1.5（sentence-transformers） |
| 数据库 | SQLite + SQLAlchemy |
| 聚类分析 | KMeans（纯 numpy 实现） |
| 前端 | 原生 HTML + CSS + JavaScript |

---

## 核心功能

### 1. 拍照 / 文字搜题
- 上传题目图片，InternVL 自动识别题目文字
- **智能判断**：若图片含图表、几何图、坐标轴等，由多模态模型直接解题；纯文字题目交给 Qwen 解题
- FAISS 向量检索匹配题库中相似题目（RAG）
- Qwen3 结合检索结果生成详细解析

### 2. 历史记录
- 展示所有解题记录，支持多维筛选（学科 / 题型 / 难度 / 来源 / 关键词）
- 顶部统计卡片（总题数、错题数、正确率）
- 展开查看题目全文、题库原题对比、AI 解析、知识点标签
- 一键加入错题本

### 3. 错题本
- **错题列表**：支持学科 / 题型 / 难度 / 知识点 / 关键词筛选
- **KMeans 聚类分析**：对错题知识点进行向量化 + KMeans 聚类，识别薄弱知识群，标注严重程度（高 / 中 / 低）
- **练习推荐**：基于聚类结果从题库中推荐针对性练习题，按知识点匹配度和难度梯度排序
- **AI 报告**：Qwen 生成个性化错题分析与学习建议

### 4. 模型思考过程展示
- 解题结果页展示可折叠的模型思考过程，方便用户查看或跳过

### 5. 程序题可视化（实验性）
- 勾选选项后，对代码 / 算法类题目生成 HTML 可视化演示

---

## 数据流

```
用户上传图片
    │
    ▼
InternVL3.5-2B
    ├── 含图表 → 多模态直接解题 → 返回答案
    └── 纯文字 → 提取题目文本
                    │
                    ▼
             FAISS 向量检索
                    │
                    ▼
          Qwen3-1.7B (RAG 解答)
                    │
                    ▼
             保存至 SQLite
                    │
                    ▼
              返回给前端展示
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/search/image` | 图片搜题 |
| POST | `/api/search/text` | 文字搜题 |
| POST | `/api/mark-wrong` | 标记错题 |
| GET | `/api/history` | 获取解题历史 |
| GET | `/api/wrong-book` | 获取错题本 |
| GET | `/api/wrong-report` | 生成 AI 错题报告 |
| GET | `/api/knowledge-stats` | 知识点薄弱统计 |
| GET | `/api/cluster-analysis` | KMeans 聚类分析 |
| GET | `/api/practice-plan` | 生成个性化练习计划 |

---

## 环境要求

- Python 3.10+
- CUDA 12.x（推荐，CPU 可运行但较慢）
- 内存 8GB+，显存 6GB+（推荐）

---

## 安装与启动

### 1. 安装 PyTorch（CUDA 版）

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装 llama-cpp-python（CUDA 版）

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型

在 `config.py` 中配置以下模型路径：

| 模型 | 用途 | 来源 |
|------|------|------|
| InternVL3_5-2B | 图片识别与多模态解题 | HuggingFace: OpenGVLab/InternVL3_5-2B |
| Qwen3-1.7B-Q8_0.gguf | 文本推理与解题 | HuggingFace: Qwen/Qwen3-1.7B-GGUF |
| BAAI/bge-small-zh-v1.5 | 向量检索 | 首次运行自动下载 |

### 5. 启动服务

```bash
uvicorn backend.main:app --reload --port 8000
```

访问 [http://localhost:8000](http://localhost:8000)

---

## 修复 InternVL 模型加载问题

若遇到 `RuntimeError: Tensor.item() cannot be called on meta tensors`，需修改模型源码：

**`modeling_intern_vit.py` 第 312 行：**
```python
# 原代码
dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
# 改为
dpr = torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device='cpu').tolist()
```

修改后删除 transformers 缓存：
```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/InternVL3_5*
```

---

## 开发进度

- [x] 多模态图片识别（InternVL）
- [x] 向量检索题库（FAISS + BGE）
- [x] RAG 解题（Qwen3 + llama-cpp）
- [x] 历史记录与多维筛选
- [x] 错题本管理
- [x] KMeans 聚类薄弱知识点分析
- [x] 个性化练习推荐
- [x] 图表题多模态直接解题
- [x] 模型思考过程折叠展示
- [ ] 模型微调
- [ ] 语音讲题（TTS）
- [ ] 微信小程序端
- [ ] 知识图谱构建
