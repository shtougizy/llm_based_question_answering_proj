"""
retrieval.py 更新版
检索流程：
  1. FAISS 向量检索 → 返回 source_id 列表
  2. 从 SQLite questions 表查完整数据（替代原来读 faiss_meta.json）
  3. 降级：若 questions 表为空，回退读 faiss_meta.json
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAISS_INDEX_PATH, FAISS_META_PATH, EMBED_MODEL as EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_index       = None
_meta        = None   # 降级用的内存缓存
_embedder    = None
_use_db      = None   # 是否使用 DB 模式（惰性检测）


def _load_embedder():
    global _embedder
    if _embedder is not None:
        return
    from sentence_transformers import SentenceTransformer
    logger.info(f"加载向量编码模型: {EMBEDDING_MODEL}")
    _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    logger.info("向量编码模型加载完成")


def _load_index():
    global _index
    if _index is not None:
        return
    import faiss
    logger.info("加载 FAISS 索引...")
    _index = faiss.read_index(FAISS_INDEX_PATH)
    logger.info(f"FAISS 索引加载完成，共 {_index.ntotal} 条题目")


def _load_meta_fallback():
    """降级：把 faiss_meta.json 加载到内存"""
    global _meta
    if _meta is not None:
        return
    logger.warning("questions 表为空，回退使用 faiss_meta.json")
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 建立 source_id → item 的映射
    _meta = {item["id"]: item for item in raw}
    logger.info(f"faiss_meta.json 加载完成，共 {len(_meta)} 条")


def _check_use_db() -> bool:
    """检测 questions 表是否有数据"""
    global _use_db
    if _use_db is not None:
        return _use_db
    try:
        from core.database import SessionLocal, Question
        with SessionLocal() as db:
            count = db.query(Question).count()
            _use_db = count > 0
            logger.info(f"questions 表共 {count} 条，{'使用 DB 模式' if _use_db else '回退 JSON 模式'}")
    except Exception as e:
        logger.warning(f"DB 检测失败：{e}，回退 JSON 模式")
        _use_db = False
    return _use_db


def _fetch_from_db(source_ids: List[str]) -> List[Dict]:
    from core.database import SessionLocal, Question
    with SessionLocal() as db:
        qs = db.query(Question).filter(Question.source_id.in_(source_ids)).all()
        mapping = {}
        for q in qs:
            mapping[q.source_id] = {
                "id": q.source_id,
                "subject": q.subject or "",
                "ques_type": q.ques_type or "",
                "ques_difficulty": q.ques_difficulty or "",
                "ques_content": q.ques_content or "",
                "ques_answer": q.ques_answer or [],
                "ques_analyze": q.ques_analyze or "",
                "ques_knowledges": q.ques_knowledges or [],
                "rag_search_text": q.rag_search_text or "",
                "display_data": {
                    "subject": q.subject or "",
                    "ques_type": q.ques_type or "",
                    "ques_difficulty": q.ques_difficulty or "",
                    "ques_content": q.ques_content or "",
                    "ques_answer": q.ques_answer or [],
                    "ques_analyze": q.ques_analyze or "",
                    "ques_knowledges": q.ques_knowledges or [],
                }
            }
        return [mapping[sid] for sid in source_ids if sid in mapping]


def _fetch_from_json(source_ids: List[str]) -> List[Dict]:
    _load_meta_fallback()
    result = []
    for sid in source_ids:
        item = _meta.get(sid)
        if not item:
            continue
        dd = item.get("display_data") or {}
        meta = item.get("metadata") or {}
        result.append({
            "id": sid,
            "subject": dd.get("subject") or meta.get("subject", ""),
            "ques_type": dd.get("ques_type") or meta.get("ques_type", ""),
            "ques_difficulty": dd.get("ques_difficulty") or meta.get("difficulty", ""),
            "ques_content": dd.get("ques_content", ""),
            "ques_answer": dd.get("ques_answer") or [],
            "ques_analyze": dd.get("ques_analyze", ""),
            "ques_knowledges": dd.get("ques_knowledges") or meta.get("knowledges") or [],
            "rag_search_text": item.get("rag_search_text", ""),
            "display_data": dd,
        })
    return result


def retrieve(question_text: str, top_k: int = 5) -> List[Dict]:
    """
    向量检索题库，返回最相似的 top_k 道题
    兼容 DB 模式和 JSON 降级模式
    """
    _load_embedder()
    _load_index()

    vec = _embedder.encode([question_text], normalize_embeddings=True)

    import numpy as np
    vec = np.array(vec, dtype="float32")
    distances, indices = _index.search(vec, top_k)

    # 把 FAISS 索引位置映射回 source_id
    # faiss_meta.json 的顺序和 FAISS 索引对应
    if not hasattr(retrieve, "_id_list"):
        # 缓存 source_id 列表
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        retrieve._id_list = [item["id"] for item in raw]

    id_list = retrieve._id_list
    source_ids = []
    sims = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(id_list):
            source_ids.append(id_list[idx])
            sims.append(float(dist))

    # 从 DB 或 JSON 获取完整数据
    if _check_use_db():
        items = _fetch_from_db(source_ids)
    else:
        items = _fetch_from_json(source_ids)

    # 把相似度附加回去
    sim_map = {sid: sim for sid, sim in zip(source_ids, sims)}
    results = []
    for item in items:
        item = dict(item)
        item["similarity"] = sim_map.get(item["id"], 0.0)
        results.append(item)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
