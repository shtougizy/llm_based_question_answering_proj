"""
题库向量检索模块
- 使用 sentence-transformers 对题目文本编码
- 使用 FAISS 进行近似最近邻检索
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH, FAISS_META_PATH, EMBED_MODEL,
    QUESTION_BANK_PATH, MAX_RETRIEVE
)

logger = logging.getLogger(__name__)

_encoder = None
_faiss_index = None
_question_meta: List[Dict] = []


def _load_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载向量编码模型: {EMBED_MODEL}")
        _encoder = SentenceTransformer(EMBED_MODEL)
    return _encoder


def _load_index():
    """加载已构建好的 FAISS 索引和元数据"""
    global _faiss_index, _question_meta

    if _faiss_index is not None:
        return

    import faiss

    if not Path(FAISS_INDEX_PATH).exists():
        logger.warning("FAISS 索引文件不存在，请先运行 init_db.py 构建索引")
        return

    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        _question_meta = json.load(f)

    logger.info(f"FAISS 索引加载完成，共 {_faiss_index.ntotal} 条题目")


def build_index(question_bank_path: str = QUESTION_BANK_PATH):
    """
    从 JSON 题库构建 FAISS 向量索引，并保存到磁盘。
    题库格式：每行一个 JSON 对象（或一个 JSON 数组）
    """
    import faiss

    # 读取题库
    questions = _load_question_bank(question_bank_path)
    if not questions:
        logger.error("题库为空，无法构建索引")
        return

    # 提取题目文本用于编码
    texts = []
    for q in questions:
        content = q.get("ques_content", "")
        subject = q.get("subject", "")
        texts.append(f"{subject} {content}")

    encoder = _load_encoder()
    logger.info(f"正在对 {len(texts)} 条题目编码...")
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 构建 FAISS 内积索引（归一化后等价余弦相似度）
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 保存索引和元数据
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    logger.info(f"FAISS 索引构建完成，保存至 {FAISS_INDEX_PATH}")


def retrieve(query: str, top_k: int = MAX_RETRIEVE) -> List[Dict[str, Any]]:
    """
    根据查询文本检索最相似的题目

    Args:
        query: 查询文本（用户输入的题目）
        top_k: 返回最多多少条

    Returns:
        相似题目列表，每项包含题目信息和相似度分数
    """
    _load_index()

    if _faiss_index is None or len(_question_meta) == 0:
        return []

    encoder = _load_encoder()
    query_vec = encoder.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    scores, indices = _faiss_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_question_meta):
            continue
        item = dict(_question_meta[idx])
        item["similarity"] = float(score)
        results.append(item)

    return results


def _load_question_bank(path: str) -> List[Dict]:
    """加载 JSON 题库，支持 JSON 数组或每行一个 JSON 对象（JSONL）"""
    questions = []
    path_obj = Path(path)

    if not path_obj.exists():
        logger.error(f"题库文件不存在: {path}")
        return []

    content = path_obj.read_text(encoding="utf-8").strip()

    # 尝试解析为 JSON 数组
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # 尝试 JSONL 格式
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return questions
