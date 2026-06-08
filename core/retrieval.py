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


def _normalize_item(item: dict) -> dict:
    """标准化单个题库条目：展开嵌套字段、统一类型、反序列化 JSON 字符串"""
    dd = item.get("display_data") or {}
    meta = item.get("metadata") or {}

    # 展平 display_data
    for k, v in dd.items():
        if k not in item:
            item[k] = v
    # 展平 metadata
    if "subject" not in item:
        item["subject"] = meta.get("subject", "")
    if "ques_difficulty" not in item:
        item["ques_difficulty"] = meta.get("difficulty", "一般")

    # 通用 JSON 字符串反序列化：适用于从 DB 读取的 Text 列
    def _parse_json_string(val):
        if isinstance(val, str) and val.strip().startswith('['):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        return val

    # ques_answer：最终统一为逗号分隔字符串
    qa = _parse_json_string(item.get("ques_answer", ""))
    if isinstance(qa, list):
        item["ques_answer"] = ", ".join(str(x) for x in qa)
    elif not isinstance(qa, str):
        item["ques_answer"] = str(qa)

    # ques_knowledges：最终统一为 list
    qk = _parse_json_string(item.get("ques_knowledges", []))
    if isinstance(qk, list):
        item["ques_knowledges"] = qk
    elif isinstance(qk, str) and qk:
        item["ques_knowledges"] = [k.strip() for k in qk.replace("、", ",").split(",") if k.strip()]
    else:
        item["ques_knowledges"] = []

    # ques_content 是 dict 时提取文本
    if isinstance(item.get("ques_content"), dict):
        qc = item["ques_content"]
        item["ques_content"] = (
            qc.get("ques_content") or
            qc.get("题目内容") or
            qc.get("题目") or
            qc.get("question") or
            qc.get("content") or
            qc.get("text") or
            str(qc)
        )

    # 同样标准化 display_data 内的字段（如果有的话）
    if "display_data" in item and isinstance(item["display_data"], dict):
        dd = item["display_data"]
        dd_qa = _parse_json_string(dd.get("ques_answer", ""))
        if isinstance(dd_qa, list):
            dd["ques_answer"] = ", ".join(str(x) for x in dd_qa)
        dd_qk = _parse_json_string(dd.get("ques_knowledges", []))
        if isinstance(dd_qk, list):
            dd["ques_knowledges"] = dd_qk
        if isinstance(dd.get("ques_content"), dict):
            qc = dd["ques_content"]
            dd["ques_content"] = (
                qc.get("ques_content") or qc.get("题目内容") or
                qc.get("题目") or str(qc)
            )

    # 标准化完成后删除冗余的内部字段，减小 API 响应体积
    item.pop("display_data", None)
    item.pop("rag_search_text", None)
    item.pop("metadata", None)
    return item


def _fetch_from_db(source_ids: List[str]) -> List[Dict]:
    from core.database import SessionLocal, Question
    with SessionLocal() as db:
        qs = db.query(Question).filter(Question.source_id.in_(source_ids)).all()
        mapping = {}
        for q in qs:
            item = {
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
            _normalize_item(item)
            mapping[q.source_id] = item
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
        normalized = {
            "id": sid,
            "metadata": meta,
            "display_data": dd,
            "subject": dd.get("subject") or meta.get("subject", ""),
            "ques_type": dd.get("ques_type") or meta.get("ques_type", ""),
            "ques_difficulty": dd.get("ques_difficulty") or meta.get("difficulty", ""),
            "ques_content": dd.get("ques_content", ""),
            "ques_answer": dd.get("ques_answer") or [],
            "ques_analyze": dd.get("ques_analyze", ""),
            "ques_knowledges": dd.get("ques_knowledges") or meta.get("knowledges") or [],
            "rag_search_text": item.get("rag_search_text", ""),
        }
        _normalize_item(normalized)
        result.append(normalized)
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
