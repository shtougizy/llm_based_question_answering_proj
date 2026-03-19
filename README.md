# 拍照搜题辅助学习系统

基于多模态大模型 + RAG 检索增强生成 + KMeans 聚类分析的 K12 智能学习辅助系统。

本项目最后更新时间：2026.3.19

---

## 项目架构

```
260221_gd/
├── config.py                      # 全局配置（模型路径、设备、数据库路径等）
├── requirements.txt               # Python 依赖
├── migrate_questions.py           # 题库数据迁移脚本（JSON → SQLite）
│
├── backend/
│   ├── main.py                    # FastAPI 后端主程序，所有 HTTP 接口
│   └── init_db.py                 # 数据库初始化脚本
│
├── core/
│   ├── multimodal.py              # 多模态识别模块（InternVL3.5-2B）
│   ├── retrieval.py               # 向量检索模块（FAISS + BGE，支持DB/JSON双模式）
│   ├── llm.py                     # 语言模型推理模块（Qwen3-1.7B-GGUF，微调后）
│   ├── database.py                # 数据库 ORM 与操作（SQLAlchemy + SQLite，6张表）
│   ├── analysis.py                # KMeans 聚类分析与练习推荐模块
│   └── auth.py                    # JWT 用户认证、密码哈希、短信验证码
│
├── frontend/
│   └── templates/
│       ├── index.html             # 首页（拍照/文字搜题，含登录注册 Modal）
│       ├── history.html           # 历史记录页
│       ├── wrong_book.html        # 错题本页（4个Tab）
│       ├── login.html             # 登录页
│       └── register.html          # 注册页
│
├── finetune/
│   ├── step1_data_analysis.py     # 数据清洗与质量分析
│   ├── step2_build_dataset.py     # 构造多任务指令数据集
│   ├── step3_gen_configs.py       # 生成 LLaMA-Factory 训练配置
│   ├── step4_baseline_eval.py     # 微调前/后评估脚本
│   ├── step5_post_eval.py         # 微调后对比评估
│   ├── step6_convert_gguf.sh      # LoRA 合并 + GGUF 转换脚本
│   ├── qwen3_k12_lora.yaml        # 训练配置
│   ├── qwen3_k12_merge.yaml       # LoRA 合并配置
│   └── output/
│       ├── qwen3-1.7b-k12-Q8_0.gguf   # 微调后量化模型（当前使用）
│       ├── baseline_results.json       # 基线评估结果
│       └── final_report.json           # 微调对比报告
│
└── data/
    ├── faiss_index.bin            # FAISS 向量索引（题库）
    ├── faiss_meta.json            # 题库元数据（保留用于索引位置映射）
    ├── question_bank.json         # 原始题库 JSON
    ├── app.db                     # SQLite 主数据库（运行时生成，不提交）
    └── models/
        ├── InternVL3_5-2B/        # 多模态模型
        └── Qwen3-1.7B-HF/         # HuggingFace 格式模型（微调用）
```

---

## 技术栈

| 模块 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 多模态识别 | InternVL3.5-2B（图文理解） |
| 语言模型 | Qwen3-1.7B-Q8_0（GGUF 量化，llama-cpp-python，K12微调） |
| 向量检索 | FAISS + BGE-small-zh-v1.5（sentence-transformers） |
| 数据库 | SQLite + SQLAlchemy（6张表） |
| 聚类分析 | KMeans（纯 numpy 实现，含 KMeans++ 初始化） |
| 用户认证 | JWT（python-jose）+ bcrypt + 短信验证码 |
| 模型微调 | LLaMA-Factory + QLoRA（int4量化） |
| 前端 | 原生 HTML + CSS + JavaScript |

---

## 核心功能

### 1. 拍照 / 文字搜题
- 上传题目图片，InternVL 自动识别题目文字
- **智能判断**：若图片含图表、几何图、坐标轴等，由多模态模型直接解题；纯文字题目交给 Qwen 解题（一次调用，不增加额外延迟）
- FAISS 向量检索匹配题库中相似题目（RAG）
- Qwen3 结合检索结果生成详细解析，双阶段调用（解题 + 元数据提取）

### 2. 历史记录
- 展示所有解题记录，支持多维筛选（学科 / 题型 / 难度 / 来源 / 关键词）
- 顶部统计卡片（总题数、错题数、正确率）
- 展开查看题目全文、题库原题对比、AI 解析、知识点标签
- 一键加入错题本

### 3. 错题本（4个Tab）
- **错题列表**：支持学科 / 题型 / 难度 / 知识点 / 关键词筛选
- **KMeans 聚类分析**：对错题知识点进行 one-hot 向量化 + KMeans++ 聚类，识别薄弱知识群，标注严重程度（高 / 中 / 低），展示各知识点出错频率热图
- **练习推荐**：基于聚类结果从题库推荐针对性练习题（知识点匹配优先 + 难度梯度）；题库无匹配时调用 LLM 生成题目；支持按单个知识点一键生成专项练习
- **AI 报告**：Qwen 生成个性化错题分析与学习建议

### 4. 用户系统
- 手机号 + 用户名 + 密码注册（含短信验证码，开发模式下验证码显示在页面）
- 用户名 + 密码登录，JWT 保持登录态（7天有效）
- 游客模式（sessionStorage 存储，关闭浏览器即失效，24小时后服务端自动清理）
- 登录/注册 Modal 弹窗（初次进入网站 & 游客首次搜题后触发）

### 5. 模型思考过程展示
- 解题结果页展示可折叠的模型思考过程，方便用户查看或跳过

### 6. 程序题可视化
- 勾选选项后，对代码 / 算法类题目生成交互式 HTML 动画演示
- 内置模板：01背包（DP表格动画）、排序（柱状图动画）、二分查找、斐波那契、通用步骤展示

---

## 数据库设计

系统使用 SQLite，共 6 张表：

| 表名 | 说明 |
|------|------|
| questions | 题库（20820条，从JSON迁移） |
| question_knowledges | 知识点索引（按知识点查题） |
| users | 用户基本信息 |
| user_auth | 用户认证（密码/手机/角色/登录时间） |
| solve_records | 解题记录（含思考过程、可视化HTML） |
| knowledge_stats | 知识点错误统计（支持聚类分析） |

---

## 数据流

```
用户上传图片
    │
    ▼
InternVL3.5-2B（一次调用）
    ├── 含图表 ──→ 多模态直接解题 ──→ 跳过LLM
    │
    └── 纯文字 ──→ 提取题目文本
                        │
                        ▼
                  FAISS 向量检索
                        │
                        ▼
              Qwen3-1.7B（RAG 解答）
              ├── 阶段1：生成解答（800 tokens）
              └── 阶段2：提取元数据（学科/题型/知识点）
                        │
                        ▼
                   保存至 SQLite
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
        返回前端展示        （可选）程序题可视化
                                   │
                              内置HTML模板
                              直接渲染，不调用LLM
```

---

## 模型微调

### 数据准备
- 原始题库 20820 条，清洗后保留 **20571 条**（98.8%）
- 过滤条件：无内容、无答案、无解析、内容过短/过长、重复
- 构造三类指令数据共 **48344 条**：

| 任务 | 说明 | 数量 |
|------|------|------|
| Task A 解题 | 给题目 → 输出答案+解析 | 20571 |
| Task B 知识点提取 | 给题目 → 输出涉及知识点 | 20571 |
| Task C 错题分析 | 给题目+错误答案 → 分析错因+正确解析 | 7202 |

- 数据集划分：训练集 85% / 验证集 7% / 测试集 8%
- 实际训练取 1/10 子集（4109条）用于快速验证

### 训练配置

| 参数 | 值 |
|------|------|
| 基础模型 | Qwen3-1.7B（HuggingFace 格式） |
| 微调方法 | QLoRA（int4 量化） |
| LoRA rank / alpha | 8 / 16 |
| 学习率 | 3e-4（cosine 调度，warmup 5%） |
| Batch size | 1（梯度累积 ×16，等效 batch=16） |
| 最大序列长度 | 512 |
| 训练轮数 | 2 epochs |
| 训练时长 | **约 47 分钟**（RTX 4060 8GB） |
| 微调框架 | LLaMA-Factory（finetune conda 环境，Python 3.11） |

### 微调效果

| 指标 | 微调前 | 微调后 | 提升 |
|------|--------|--------|------|
| 平均 ROUGE-L | 0.2262 | 0.4265 | **+88.6%** |
| 解题答案准确率 | 16.0% | 58.0% | **+262.5%** |
| 最终训练 Loss | — | 1.268 | — |

### 模型转换流程

```
Qwen3-1.7B（HF格式）
    ↓ LLaMA-Factory QLoRA 微调（47分钟）
HF格式 + LoRA adapter（3,211,264 可训练参数，占比 0.16%）
    ↓ llamafactory-cli export 合并 LoRA
HF格式完整模型（merged，3份 safetensors）
    ↓ llama.cpp convert_hf_to_gguf.py
F16 GGUF（3.8GB）
    ↓ llama-quantize Q8_0
Q8_0 GGUF（2.1GB，当前使用）
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
| GET | `/api/practice-by-knowledge` | 按知识点生成专项练习 |
| POST | `/api/auth/send-sms` | 发送短信验证码 |
| POST | `/api/auth/register` | 用户注册 |
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/auth/guest` | 游客登录 |
| GET | `/api/auth/me` | 获取当前用户信息 |

---

## 环境要求

- Python 3.10+（主项目）/ Python 3.11+（微调环境）
- CUDA 12.x（推荐，CPU 可运行但较慢）
- 内存 16GB+，显存 6GB+（推荐 8GB）

---

## 安装与启动

### 1. 安装 PyTorch（CUDA 版）

```bash
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 安装 llama-cpp-python（CUDA 版）

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
pip install python-jose[cryptography] passlib[bcrypt] bcrypt==4.0.1
```

### 4. 下载模型

在 `config.py` 中配置以下模型路径：

| 模型 | 用途 | 来源 |
|------|------|------|
| InternVL3_5-2B | 图片识别与多模态解题 | HuggingFace: OpenGVLab/InternVL3_5-2B |
| qwen3-1.7b-k12-Q8_0.gguf | 文本推理与解题（K12微调版） | 本地微调生成 / 原版：Qwen/Qwen3-1.7B-GGUF |
| BAAI/bge-small-zh-v1.5 | 向量检索 | 首次运行自动下载 |

### 5. 初始化数据库与迁移题库

```bash
# 初始化数据库表结构
python3 backend/init_db.py

# 将 faiss_meta.json 题库迁移至 SQLite（首次运行）
python3 migrate_questions.py
```

### 6. 启动服务

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

可选：通过 Cloudflare Tunnel 在公网访问

```bash
cloudflared tunnel --url http://localhost:8000
```

复制命令行中给出的随机网址即可在任意互联网上访问。

---

## 微调流程复现

```bash
# 需要先创建 finetune 环境（Python 3.11）
conda create -n finetune python=3.11 -y
conda activate finetune
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes
cd LLaMA-Factory && pip install -e ".[torch,metrics]"

# 1. 数据清洗
python3 finetune/step1_data_analysis.py

# 2. 构造指令数据集（Task A/B/C 三类任务）
python3 finetune/step2_build_dataset.py

# 3. 生成训练配置并注册数据集
python3 finetune/step3_gen_configs.py

# 4. 微调前基线评估
conda activate study_ai
python3 finetune/step4_baseline_eval.py

# 5. 开始微调（约 47 分钟）
conda activate finetune
cd LLaMA-Factory
llamafactory-cli train ../finetune/qwen3_k12_lora.yaml

# 6. 合并 LoRA 权重
llamafactory-cli export ../finetune/qwen3_k12_merge.yaml

# 7. 转换为 GGUF 并量化
python3 llama.cpp/convert_hf_to_gguf.py finetune/output/qwen3-1.7b-k12-merged \
    --outfile finetune/output/qwen3-1.7b-k12-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize \
    finetune/output/qwen3-1.7b-k12-f16.gguf \
    finetune/output/qwen3-1.7b-k12-Q8_0.gguf Q8_0

# 8. 微调后评估对比
conda activate study_ai
python3 finetune/step4_baseline_eval.py
```

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
- [x] 题库迁移至 SQLite 关系数据库
- [x] 历史记录与多维筛选
- [x] 错题本管理
- [x] KMeans 聚类薄弱知识点分析
- [x] 个性化练习推荐（题库匹配 + LLM 生成兜底）
- [x] 图表题多模态直接解题
- [x] 模型思考过程折叠展示
- [x] 程序题可视化（内置交互动画模板）
- [x] 用户注册 / 登录 / 游客系统（JWT + 手机验证码）
- [x] Qwen3-1.7B K12 领域 QLoRA 微调（ROUGE-L +88.6%，准确率 +262.5%）
- [ ] 语音讲题（TTS）
- [ ] 微信小程序端
- [ ] 知识图谱构建
