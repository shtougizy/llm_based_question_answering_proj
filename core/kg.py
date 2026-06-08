"""
轻量知识图谱服务模块
提供知识点前置依赖查询、学习路径生成、按路径推荐练习题等功能。

数据来源：kg_relations 表（由 scripts/extract_kg_relations.py 离线抽取）
"""
import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# 模块级缓存：邻接表 {知识点: [前置知识列表]}
_adjacency_cache: Optional[Dict[str, List[str]]] = None


def build_adjacency() -> Dict[str, List[str]]:
    """
    从 kg_relations 表加载全部关系，构建邻接表。
    首次调用会查询数据库并缓存，后续调用直接返回缓存。

    Returns:
        {knowledge_point: [prerequisite_list]} 格式的邻接表
    """
    global _adjacency_cache
    if _adjacency_cache is not None:
        return _adjacency_cache

    from core.database import get_all_kg_relations
    relations = get_all_kg_relations()
    adj = defaultdict(list)
    for rel in relations:
        adj[rel["knowledge_point"]].append(rel["prerequisite"])
    _adjacency_cache = dict(adj)
    logger.info(f"知识图谱邻接表已构建: {len(_adjacency_cache)} 个知识点, "
                f"{len(relations)} 条关系")
    return _adjacency_cache


def reload_adjacency():
    """强制重新加载邻接表（抽取新关系后使用）"""
    global _adjacency_cache
    _adjacency_cache = None
    return build_adjacency()


def get_prerequisites_of(knowledge: str, kg: Dict[str, List[str]] = None) -> List[str]:
    """查单个知识点的直接前置依赖"""
    if kg is None:
        kg = build_adjacency()
    return kg.get(knowledge, [])


def expand_weak_points(
    weak_points: List[str],
    kg: Dict[str, List[str]] = None,
    max_depth: int = 2,
) -> Dict[str, List[str]]:
    """
    从用户薄弱知识点出发，BFS 向上游展开前置依赖链。

    Args:
        weak_points: 用户薄弱知识点列表
        kg: 邻接表（None 则自动加载）
        max_depth: 最大展开深度

    Returns:
        {知识点: [该知识点的直接前置依赖列表]}，包含原始薄弱点和展开的前置知识点
    """
    if kg is None:
        kg = build_adjacency()

    if not kg:
        return {kp: [] for kp in weak_points}

    # BFS：从薄弱点出发往上走（查前置），记录每个知识点的前置
    discovered = {}  # 知识点 → 其直接前置列表
    queue = deque(weak_points)
    depths = {kp: 0 for kp in weak_points}

    while queue:
        current = queue.popleft()
        current_depth = depths.get(current, 0)

        if current in discovered:
            continue

        prereqs = kg.get(current, [])
        discovered[current] = prereqs

        if current_depth < max_depth:
            for pre in prereqs:
                if pre not in discovered and pre not in depths:
                    depths[pre] = current_depth + 1
                    queue.append(pre)

    return discovered


def _topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """
    Kahn 算法拓扑排序。

    graph: {节点: [前置节点列表]}，边从 prerequisite 指向 knowledge_point。
           即：学 B 之前需要学 A，则图中 B → A 有边（B 依赖 A）。

    Returns:
        按学习顺序排列的知识点列表（基础 → 进阶）
    """
    # 收集所有节点
    all_nodes = set(graph.keys())
    for prereqs in graph.values():
        all_nodes.update(prereqs)

    # 构建正向邻接（前置→后继）和入度表
    # 在 graph 中，边的方向是 knowledge_point → prerequisite
    # 学习路径则是 prerequisite → knowledge_point（先学前置，再学后继）
    forward_adj = defaultdict(list)  # prerequisite → [可以用它做前置的知识点]
    in_degree = {node: 0 for node in all_nodes}

    for node, prereqs in graph.items():
        for pre in prereqs:
            forward_adj[pre].append(node)
            in_degree[node] = in_degree.get(node, 0) + 1
            if pre not in in_degree:
                in_degree[pre] = in_degree.get(pre, 0)

    # Kahn: 从入度为 0 的节点开始
    queue = deque([n for n in all_nodes if in_degree.get(n, 0) == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for successor in forward_adj.get(node, []):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    # 如果有环或未处理的节点，追加到末尾
    remaining = [n for n in all_nodes if n not in set(result)]
    result.extend(remaining)

    return result


def _classify_stages(
    sorted_kps: List[str],
    weak_points: Set[str],
) -> List[Dict]:
    """
    将拓扑排序后的知识点列表划分为三个阶段。

    阶段划分规则：
    - 阶段1 "基础巩固"：拓扑序前 1/3，且不是用户直接薄弱点的基础知识
    - 阶段2 "过渡提升"：拓扑序中间 1/3
    - 阶段3 "核心突破"：用户原始薄弱点（拓扑序最后的部分）
    """
    if not sorted_kps:
        return []

    n = len(sorted_kps)
    weak_set = set(weak_points)

    # 分离：基础知识点（非薄弱点）和薄弱点
    foundation = [kp for kp in sorted_kps if kp not in weak_set]
    core = [kp for kp in sorted_kps if kp in weak_set]

    stages = []

    if foundation:
        # 基础知识点再按位置分为两段
        fn = len(foundation)
        if fn <= 2:
            stages.append({
                "stage": 1, "label": "基础巩固",
                "knowledge_points": foundation,
                "question_count": max(2, len(foundation)),
            })
        else:
            mid = fn // 2
            stages.append({
                "stage": 1, "label": "基础巩固",
                "knowledge_points": foundation[:mid],
                "question_count": max(2, len(foundation[:mid])),
            })
            stages.append({
                "stage": 2, "label": "过渡提升",
                "knowledge_points": foundation[mid:],
                "question_count": max(2, len(foundation[mid:])),
            })

    # 核心薄弱点
    stage_num = len(stages) + 1
    if foundation:
        label = "核心突破"
    else:
        label = "专项练习"

    stages.append({
        "stage": stage_num, "label": label,
        "knowledge_points": core,
        "question_count": max(3, len(core) * 2),
    })

    return stages


def compute_learning_path(
    weak_points: List[str],
    kg: Dict[str, List[str]] = None,
    max_depth: int = 2,
) -> List[Dict]:
    """
    生成结构化的学习路径。

    算法流程：
    1. BFS 展开前置依赖
    2. 对有向图做拓扑排序
    3. 将排序结果分为 2-3 个学习阶段

    Args:
        weak_points: 用户薄弱知识点列表
        kg: 邻接表
        max_depth: 最大展开深度

    Returns:
        学习路径列表，每个阶段包含：
        {"stage": N, "label": str, "knowledge_points": [...], "question_count": N}
    """
    if kg is None:
        kg = build_adjacency()

    if not kg or not weak_points:
        # KG 不可用，返回单阶段（仅薄弱点本身）
        return [{
            "stage": 1, "label": "专项练习",
            "knowledge_points": list(weak_points),
            "question_count": max(3, len(weak_points) * 2),
        }]

    # 1. 展开
    expanded = expand_weak_points(weak_points, kg, max_depth)

    # 2. 拓扑排序
    sorted_kps = _topological_sort(expanded)

    # 3. 分阶段
    stages = _classify_stages(sorted_kps, set(weak_points))

    logger.info(f"学习路径生成: {len(weak_points)} 薄弱点 → "
                f"展开 {len(expanded)} 知识点 → {len(stages)} 个阶段")
    return stages


def recommend_by_learning_path(
    learning_path: List[Dict],
    question_bank: List[Dict],
    wrong_records: List[Dict] = None,
    questions_per_stage: int = None,
) -> List[Dict]:
    """
    为学习路径的每个阶段推荐练习题。

    复用 analysis.py 中的 recommend_practice_questions()，
    为每个阶段的知识点集合分别推荐题目。

    Args:
        learning_path: compute_learning_path() 的输出
        question_bank: 题库列表
        wrong_records: 用户错题记录（用于去重）
        questions_per_stage: 每个阶段的题目数（None 则用 learning_path 中的 question_count）

    Returns:
        带题目的学习路径，每个阶段增加 "questions" 字段
    """
    from core.analysis import recommend_practice_questions

    wrong_records = wrong_records or []
    result = []

    for stage in learning_path:
        n = questions_per_stage or stage.get("question_count", 3)
        # 构建该阶段的虚拟聚类对象，供 recommend_practice_questions 使用
        fake_cluster = {
            "label": stage["label"],
            "knowledge_points": stage["knowledge_points"],
            "knowledge_freq": {kp: 1 for kp in stage["knowledge_points"]},
            "wrong_count": len(wrong_records),
            "severity": "中",
            "records": wrong_records,
            "subjects": [],
        }

        stage_questions = recommend_practice_questions(fake_cluster, n_questions=n)

        # 如果题库中匹配不够，用 LLM 补充
        if len(stage_questions) < max(1, n // 2):
            from core.llm import generate_practice_questions
            main_kp = stage["knowledge_points"][0] if stage["knowledge_points"] else ""
            if main_kp:
                logger.info(f"阶段{stage['stage']} 题库不足，LLM 补充「{main_kp}」")
                llm_qs = generate_practice_questions(main_kp, max(1, n - len(stage_questions)))
                stage_questions.extend(llm_qs)

        result.append({
            **stage,
            "questions": stage_questions,
        })

    return result
