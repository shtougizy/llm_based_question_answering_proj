"""
知识图谱关系抽取脚本（离线，一次性运行）
从题库中提取所有知识点，使用 Qwen3-1.7B 批量抽取前置依赖关系，写入 kg_relations 表。
"""
import json
import logging
import re
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FAISS_META_PATH
from core.database import init_db, insert_kg_relation, get_all_kg_relations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_all_knowledge_points() -> list:
    """从题库 faiss_meta.json 中提取所有不重复的知识点"""
    if not Path(FAISS_META_PATH).exists():
        logger.error(f"题库文件不存在: {FAISS_META_PATH}")
        return []

    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        bank = json.load(f)

    all_kps = set()
    for q in bank:
        kps = q.get("ques_knowledges") or q.get("metadata", {}).get("knowledges") or []
        for k in kps:
            k = k.strip()
            if k and len(k) >= 2:
                all_kps.add(k)

    logger.info(f"从 {len(bank)} 道题目中提取到 {len(all_kps)} 个不重复知识点")
    return sorted(all_kps)


def extract_prerequisites_for(knowledge: str, _llm) -> list:
    """
    使用 Qwen3 模型，针对单个知识点提取其前置依赖。
    返回前置知识点列表。
    """
    prompt = (
        f"对于知识点「{knowledge}」，请列出学生在学习它之前必须掌握的直接前置知识点。\n"
        "只列直接的前置知识，不要列间接的（前置的前置）。\n"
        "如果该知识点没有明显的前置依赖，返回空列表。\n\n"
        "只输出一行JSON，不要解释、不要Markdown标记、不要多余文字：\n"
        '{"prerequisites": ["前置知识1", "前置知识2"]}'
    )

    resp = _llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是K12教育专家，熟悉各学科知识点的先修依赖关系。"
                    "你的唯一任务是输出JSON，禁止输出任何其他内容（包括分析、解释、思考过程、Markdown格式）。"
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.1,
    )

    raw = resp["choices"][0]["message"]["content"].strip()

    # 去除 <think> / 思考标签（支持多种格式）
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'<\|think\|>.*?<\|/think\|>', '', raw, flags=re.DOTALL).strip()

    # 去除 markdown 代码块标记
    for fence in ['```json', '```JSON', '```']:
        raw = raw.replace(fence, '')

    # 去除常见的模型回复前缀
    # 只保留从第一个 { 到最后一个 } 之间的内容
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        logger.warning(f"未找到JSON对象: {raw[:120]}")
        return []

    raw_json = raw[start:end + 1]

    # 尝试解析 JSON
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # 日志记录失败详情（仅前几条，避免刷屏）
        if not hasattr(extract_prerequisites_for, '_parse_fail_count'):
            extract_prerequisites_for._parse_fail_count = 0
        extract_prerequisites_for._parse_fail_count += 1
        if extract_prerequisites_for._parse_fail_count <= 5:
            logger.warning(f"JSON解析失败 (第{extract_prerequisites_for._parse_fail_count}次): "
                           f"raw={raw[:150]}")
        return []

    prereqs = data.get("prerequisites", [])
    if not isinstance(prereqs, list):
        prereqs = []

    # 清洗：去除空白、单字、不包含中文的知识点
    result = []
    for p in prereqs:
        p = p.strip()
        if not p or len(p) < 2:
            continue
        # 必须有中文字符（排除纯英文/数字/符号）
        if not re.search(r'[一-鿿]', p):
            continue
        result.append(p)

    return result


def main():
    # 1. 初始化数据库（确保 kg_relations 表存在）
    init_db()
    logger.info("数据库已初始化")

    # 2. 检查是否已有抽取结果
    existing = get_all_kg_relations()
    if existing:
        logger.info(f"kg_relations 表中已有 {len(existing)} 条关系记录")
        resp = input("是否清空重新抽取？(y/N): ").strip().lower()
        if resp == 'y':
            from core.database import engine
            from sqlalchemy import text
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM kg_relations"))
            logger.info("已清空 kg_relations 表")
        else:
            logger.info("保留现有数据，跳过抽取")
            return

    # 3. 收集知识点
    all_kps = collect_all_knowledge_points()
    if not all_kps:
        logger.error("无知识点可抽取，退出")
        return

    # 4. 加载 LLM
    logger.info("加载 Qwen3 模型...")
    from core.llm import _load_llm
    _load_llm()
    from core.llm import _llm
    logger.info("模型加载完成")

    # 5. 批量抽取
    total_relations = 0
    kps_with_prereqs = 0
    kps_without_prereqs = 0
    prereq_counter = Counter()  # 统计被依赖次数

    for i, kp in enumerate(all_kps):
        logger.info(f"[{i+1}/{len(all_kps)}] 抽取: {kp}")
        try:
            prereqs = extract_prerequisites_for(kp, _llm)
            if prereqs:
                kps_with_prereqs += 1
                for pre in prereqs:
                    insert_kg_relation(kp, pre)
                    prereq_counter[pre] += 1
                    total_relations += 1
                logger.info(f"  → {len(prereqs)} 个前置: {', '.join(prereqs)}")
            else:
                kps_without_prereqs += 1
                logger.info(f"  → 无前置依赖")
        except Exception as e:
            logger.error(f"  → 抽取失败: {e}")
            kps_without_prereqs += 1

    # 6. 打印统计
    logger.info("=" * 60)
    logger.info("抽取完成！统计：")
    logger.info(f"  总知识点数:     {len(all_kps)}")
    logger.info(f"  有前置依赖:     {kps_with_prereqs}")
    logger.info(f"  无前置依赖:     {kps_without_prereqs}")
    logger.info(f"  关系总数:       {total_relations}")
    logger.info(f"  被依赖最多的知识点 Top 10:")
    for kp, count in prereq_counter.most_common(10):
        logger.info(f"    {count:3d} 次 ← {kp}")


if __name__ == "__main__":
    main()
