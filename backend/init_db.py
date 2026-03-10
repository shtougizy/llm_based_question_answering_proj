"""
初始化脚本：创建数据库表 + 构建 FAISS 向量索引
使用方式：python backend/init_db.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== 初始化数据库 ===")
    from core.database import init_db
    init_db()

    logger.info("=== 构建 FAISS 向量索引 ===")
    from config import QUESTION_BANK_PATH
    from core.retrieval import build_index
    from pathlib import Path as P

    if not P(QUESTION_BANK_PATH).exists():
        logger.warning(
            f"题库文件不存在: {QUESTION_BANK_PATH}\n"
            "请将 JSON 题库文件放到 data/question_bank.json 后重新运行"
        )
        logger.info("将创建示例题库文件用于测试...")
        _create_sample_bank(QUESTION_BANK_PATH)

    build_index(QUESTION_BANK_PATH)
    logger.info("=== 初始化完成 ===")


def _create_sample_bank(path: str):
    """创建一个示例题库（用于测试）"""
    import json
    sample = [
        {
            "subject": "初中数学",
            "ques_type": "填空题",
            "ques_difficulty": "一般",
            "ques_content": "题目内容: 在数轴上距离原点3个单位长度的点是__ ．",
            "ques_answer": ["3或-3"],
            "ques_analyze": "在数轴上，距离原点3个单位长度的点有两个：3和-3。",
            "ques_knowledges": ["数轴上两点之间的距离", "绝对值的其他应用"]
        },
        {
            "subject": "初中数学",
            "ques_type": "选择题",
            "ques_difficulty": "简单",
            "ques_content": "题目内容: 下列各数中，最小的整数是（  ）A.-3  B.-1  C.0  D.1",
            "ques_answer": ["A"],
            "ques_analyze": "负数小于0，负数中绝对值越大的数越小，所以-3最小。",
            "ques_knowledges": ["整数大小比较", "负数的比较"]
        },
        {
            "subject": "计算机",
            "ques_type": "编程题",
            "ques_difficulty": "中等",
            "ques_content": "题目内容: 请用 Python 实现冒泡排序，并输出每次交换过程。",
            "ques_answer": ["def bubble_sort(arr): ..."],
            "ques_analyze": "冒泡排序通过相邻元素比较和交换，每轮将最大元素移到末尾。",
            "ques_knowledges": ["排序算法", "冒泡排序", "Python基础"]
        },
        {
            "subject": "初中物理",
            "ques_type": "计算题",
            "ques_difficulty": "一般",
            "ques_content": "题目内容: 一个物体做匀速直线运动，速度为10m/s，经过5s后，位移是多少？",
            "ques_answer": ["50m"],
            "ques_analyze": "匀速直线运动：位移 = 速度 × 时间 = 10 × 5 = 50m",
            "ques_knowledges": ["匀速直线运动", "速度公式"]
        },
        {
            "subject": "高中数学",
            "ques_type": "解答题",
            "ques_difficulty": "较难",
            "ques_content": "题目内容: 求函数 f(x) = x² - 4x + 3 的最小值及最小值点。",
            "ques_answer": ["最小值为-1，在x=2处取得"],
            "ques_analyze": "配方法：f(x) = (x-2)² - 1，顶点为(2,-1)，最小值为-1。",
            "ques_knowledges": ["二次函数", "配方法", "顶点公式"]
        }
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    logger.info(f"示例题库已创建: {path}（共{len(sample)}题）")


if __name__ == "__main__":
    main()
