"""
知识点聚类分析模块
使用 KMeans 对用户错题的知识点进行聚类，识别薄弱知识群，
并根据聚类结果从题库中推荐针对性练习题。
支持知识图谱增强：通过前置依赖展开生成结构化学习路径。
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from core.llm import generate_practice_questions

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAISS_META_PATH

logger = logging.getLogger(__name__)


# ==================== 知识点向量化 ====================

def _build_knowledge_vocab(all_knowledges: List[str]) -> Dict[str, int]:
    """构建知识点词汇表（知识点 → 索引）"""
    unique = sorted(set(all_knowledges))
    return {k: i for i, k in enumerate(unique)}


def _encode_wrong_record(
    record: Dict,
    vocab: Dict[str, int]
) -> np.ndarray:
    """
    将一条错题记录编码为知识点 one-hot 向量。
    同时考虑学科、题型、难度等特征。
    """
    dim = len(vocab)
    vec = np.zeros(dim, dtype=np.float32)

    knowledges = record.get("knowledges") or []
    for k in knowledges:
        if k in vocab:
            vec[vocab[k]] = 1.0

    return vec


# ==================== KMeans 聚类 ====================

def _kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    纯 numpy 实现的 KMeans，避免额外依赖。
    返回 (labels, centroids)
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]

    if n <= k:
        # 样本数不足时，直接一个样本一个簇
        return np.arange(n), X.copy()

    # KMeans++ 初始化
    centers_idx = [rng.randint(0, n)]
    for _ in range(k - 1):
        dists = np.array([
            min(np.sum((X[i] - X[c]) ** 2) for c in centers_idx)
            for i in range(n)
        ])
        # probs = dists / dists.sum()
        total = dists.sum()
        if total == 0:
            # 所有剩余点与已有中心距离均为 0（特征完全相同），剩余中心随机选
            centers_idx.extend(rng.choice(n, size=k - len(centers_idx), replace=False).tolist())
            break
        probs = dists / total
        centers_idx.append(rng.choice(n, p=probs))

    centroids = X[centers_idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # 分配
        dists = np.array([
            np.sum((X - c) ** 2, axis=1) for c in centroids
        ])  # shape: (k, n)
        new_labels = np.argmin(dists, axis=0)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        # 更新质心
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)

    return labels, centroids


def cluster_weak_knowledge_points(
    wrong_records: List[Dict],
    n_clusters: int = None
) -> List[Dict[str, Any]]:
    """
    对用户的错题进行 KMeans 聚类，识别薄弱知识群。

    Args:
        wrong_records: 用户错题列表（来自数据库）
        n_clusters: 聚类数量，默认自动决定（min(错题数//3, 5)）

    Returns:
        聚类结果列表，每项包含：
        {
            "cluster_id": int,
            "label": str,               # 聚类主题标签（最高频知识点）
            "knowledge_points": list,   # 该簇包含的知识点
            "wrong_count": int,         # 错题数
            "severity": str,            # 严重程度：高/中/低
            "records": list,            # 该簇的错题记录
            "subjects": list,           # 涉及学科
        }
    """
    if not wrong_records:
        return []

    # 收集所有知识点
    all_knowledges = []
    for r in wrong_records:
        all_knowledges.extend(r.get("knowledges") or [])

    if not all_knowledges:
        logger.warning("错题中无知识点信息，无法进行聚类")
        return _fallback_frequency_analysis(wrong_records)

    vocab = _build_knowledge_vocab(all_knowledges)

    if len(vocab) == 0:
        return []

    # 编码
    X = np.array([_encode_wrong_record(r, vocab) for r in wrong_records])

    # 去掉全零向量（无知识点的记录）
    nonzero_mask = X.sum(axis=1) > 0
    X_valid = X[nonzero_mask]
    records_valid = [r for r, m in zip(wrong_records, nonzero_mask) if m]

    if len(X_valid) == 0:
        return _fallback_frequency_analysis(wrong_records)

    # 确定 k
    if n_clusters is None:
        k = max(2, min(len(X_valid) // 2, 5))
    else:
        k = max(1, min(n_clusters, len(X_valid)))

    logger.info(f"KMeans 聚类：{len(X_valid)} 条错题，k={k}")

    labels, centroids = _kmeans(X_valid, k)

    # 整理聚类结果
    clusters = []
    vocab_inv = {v: k for k, v in vocab.items()}
    total_wrong = len(records_valid)

    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_records = [r for r, m in zip(records_valid, mask) if m]

        if not cluster_records:
            continue

        # 统计该簇的知识点频率
        kn_counter = Counter()
        subject_counter = Counter()
        for r in cluster_records:
            kns = r.get("knowledges") or []
            kn_counter.update(kns)
            if r.get("subject"):
                subject_counter[r["subject"]] += 1

        top_kns = [kn for kn, _ in kn_counter.most_common(8)]
        top_subjects = [s for s, _ in subject_counter.most_common(3)]

        # 严重程度
        ratio = len(cluster_records) / total_wrong
        if ratio >= 0.4 or len(cluster_records) >= 5:
            severity = "高"
        elif ratio >= 0.2 or len(cluster_records) >= 3:
            severity = "中"
        else:
            severity = "低"

        # 簇标签：取最高频知识点
        label = top_kns[0] if top_kns else f"知识群{cluster_id+1}"

        clusters.append({
            "cluster_id": cluster_id,
            "label": label,
            "knowledge_points": top_kns,
            "knowledge_freq": dict(kn_counter.most_common(10)),
            "wrong_count": len(cluster_records),
            "severity": severity,
            "records": cluster_records,
            "subjects": top_subjects,
        })

    # 按错题数降序
    clusters.sort(key=lambda c: c["wrong_count"], reverse=True)
    return clusters


def _fallback_frequency_analysis(wrong_records: List[Dict]) -> List[Dict]:
    """无知识点时的降级处理：按学科分组"""
    groups = defaultdict(list)
    for r in wrong_records:
        subject = r.get("subject") or "未分类"
        groups[subject].append(r)

    result = []
    for i, (subject, records) in enumerate(sorted(groups.items(), key=lambda x: -len(x[1]))):
        result.append({
            "cluster_id": i,
            "label": subject,
            "knowledge_points": [],
            "knowledge_freq": {},
            "wrong_count": len(records),
            "severity": "高" if len(records) >= 5 else "中" if len(records) >= 2 else "低",
            "records": records,
            "subjects": [subject],
        })
    return result


# ==================== 练习题推荐 ====================

def _load_question_bank() -> List[Dict]:
    """从 FAISS meta 文件加载题库，并将嵌套字段展平到顶层"""
    if not Path(FAISS_META_PATH).exists():
        logger.warning("题库 meta 文件不存在")
        return []
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 将 display_data 子字段展平到顶层，兼容下游直接 .get("ques_content") 等用法
    for item in raw:
        dd = item.get("display_data") or {}
        for k, v in dd.items():
            if k not in item:
                item[k] = v
        # 同时展平 metadata 中的 subject / difficulty 等
        meta = item.get("metadata") or {}
        if "subject" not in item:
            item["subject"] = meta.get("subject", "")
        if "ques_difficulty" not in item:
            item["ques_difficulty"] = meta.get("difficulty", "一般")
        # 确保 ques_answer 为字符串（题库中为列表）
        if isinstance(item.get("ques_answer"), list):
            item["ques_answer"] = ", ".join(str(x) for x in item["ques_answer"])
        # 处理 ques_content 为 dict 的异常情况（现代文阅读等复杂题型）
        if isinstance(item.get("ques_content"), dict):
            qc = item["ques_content"]
            # 尝试从嵌套 dict 中提取文本
            item["ques_content"] = (
                qc.get("ques_content") or
                qc.get("题目内容") or
                qc.get("题目") or
                qc.get("question") or
                qc.get("content") or
                qc.get("text") or
                json.dumps(qc, ensure_ascii=False)
            )
        # 删除内部字段，减小 API 响应体积
        item.pop("display_data", None)
        item.pop("rag_search_text", None)
        item.pop("metadata", None)
    return raw


def recommend_practice_questions(
    cluster: Dict[str, Any],
    n_questions: int = 5,
    exclude_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """
    根据聚类结果，从题库中推荐针对性练习题。

    策略：
    1. 优先选取知识点完全匹配的题目
    2. 其次选取部分知识点匹配的题目
    3. 按难度梯度排序（先易后难）
    4. 排除已做过的题目

    Args:
        cluster: 聚类结果字典
        n_questions: 推荐题目数量
        exclude_ids: 需要排除的题目索引列表

    Returns:
        推荐题目列表
    """
    question_bank = _load_question_bank()
    if not question_bank:
        return []

    target_kns = set(cluster.get("knowledge_points") or [])
    target_subjects = set(cluster.get("subjects") or [])
    exclude_ids = set(exclude_ids or [])

    # 已做错过的题目文本，用于去重
    done_texts = set()
    for r in cluster.get("records") or []:
        # 取前30个字符作为匹配键
        done_texts.add(r.get("question_text", "")[:30])

    difficulty_order = {"简单": 0, "一般": 1, "中等": 1, "较难": 2, "困难": 2}

    scored = []
    for idx, q in enumerate(question_bank):
        if idx in exclude_ids:
            continue

        q_text = q.get("ques_content", "")[:30]
        if q_text in done_texts:
            continue

        q_kns = set(q.get("ques_knowledges") or [])
        q_subject = q.get("subject", "")

        # 计算匹配分
        kn_overlap = len(target_kns & q_kns)
        subject_match = 1 if q_subject in target_subjects else 0
        difficulty = difficulty_order.get(q.get("ques_difficulty", "一般"), 1)

        if kn_overlap == 0 and subject_match == 0:
            continue

        # 综合分：知识点匹配优先，其次学科匹配，难度作为次要排序
        score = kn_overlap * 10 + subject_match * 2

        scored.append((score, difficulty, idx, q))

    if not scored:
        # 退路：只按学科匹配
        for idx, q in enumerate(question_bank):
            if idx in exclude_ids:
                continue
            if q.get("subject", "") in target_subjects:
                diff = difficulty_order.get(q.get("ques_difficulty", "一般"), 1)
                scored.append((1, diff, idx, q))

    # 排序：分数高 → 难度低（先易后难）→ 取前 n
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = scored[:n_questions]

    result = []
    for score, diff, idx, q in selected:
        item = dict(q)
        item["_match_score"] = score
        item["_bank_idx"] = idx
        result.append(item)

    return result


def generate_cluster_practice_plan(
    wrong_records: List[Dict],
    questions_per_cluster: int = 3,
    use_kg: bool = True,
) -> List[Dict[str, Any]]:
    """
    完整的练习计划生成入口：
    1. 对错题进行 KMeans 聚类
    2. 为每个聚类推荐练习题
    3. 如果知识图谱可用，为每个聚类生成结构化学习路径
    4. 返回完整的练习计划

    Args:
        wrong_records: 用户所有错题
        questions_per_cluster: 每个知识群推荐的练习题数
        use_kg: 是否启用知识图谱增强学习路径

    Returns:
        练习计划列表，每项包含聚类信息和推荐题目。
        如果 KG 可用且能展开依赖链，还会包含 learning_path 字段。
    """
    clusters = cluster_weak_knowledge_points(wrong_records)

    # 惰性加载 KG（避免循环导入）
    kg = {}
    if use_kg:
        try:
            from core.kg import build_adjacency, compute_learning_path, recommend_by_learning_path
            kg = build_adjacency()
        except Exception as e:
            logger.warning(f"知识图谱加载失败，使用纯聚类模式: {e}")
            kg = {}

    plan = []
    for cluster in clusters:
        practice_questions = recommend_practice_questions(
            cluster,
            n_questions=questions_per_cluster,
        )
        if not practice_questions:
            logger.info(f"题库无匹配，LLM 生成「{cluster['label']}」练习题")
            practice_questions = generate_practice_questions(
                cluster["label"], questions_per_cluster
            )

        plan_entry = {
            "cluster_id": cluster["cluster_id"],
            "label": cluster["label"],
            "knowledge_points": cluster["knowledge_points"],
            "wrong_count": cluster["wrong_count"],
            "severity": cluster["severity"],
            "subjects": cluster["subjects"],
            "practice_questions": practice_questions,
            "has_learning_path": False,
        }

        # KG 增强：为聚类知识点生成学习路径
        if kg and cluster["knowledge_points"]:
            try:
                learning_path = compute_learning_path(
                    cluster["knowledge_points"], kg, max_depth=2
                )
                # 只有当学习路径有多个阶段（确实展开了依赖链）时才启用
                if len(learning_path) > 1:
                    bank = _load_question_bank()
                    path_with_questions = recommend_by_learning_path(
                        learning_path, bank,
                        wrong_records=cluster.get("records", []),
                    )
                    plan_entry["learning_path"] = path_with_questions
                    plan_entry["has_learning_path"] = True
            except Exception as e:
                logger.warning(f"学习路径生成失败（cluster={cluster['cluster_id']}）: {e}")

        plan.append(plan_entry)

    return plan
