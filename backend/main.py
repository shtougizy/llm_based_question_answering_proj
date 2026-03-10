"""
FastAPI 后端主入口
提供 RESTful API 接口，同时提供简单的 HTML 前端页面
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import UPLOAD_DIR
from core.database import (
    init_db, get_or_create_user, save_solve_record,
    mark_as_wrong, get_wrong_questions, get_solve_history,
    get_knowledge_stats, delete_records, unmark_wrong
)
from core.retrieval import retrieve
from core.multimodal import extract_question_from_image
from core.llm import answer_question, generate_wrong_answer_report, generate_visualization_html
from core.analysis import generate_cluster_practice_plan, cluster_weak_knowledge_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 确保上传目录存在
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="拍照搜题辅助学习系统", version="1.0.0")

# 静态文件 & 模板
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))

# 默认用户 ID（演示用，实际可加入登录系统）
DEFAULT_USER = "default"


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("系统启动完成")


# ==================== 前端页面路由 ====================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/wrong-book", response_class=HTMLResponse)
async def wrong_book_page(request: Request):
    return templates.TemplateResponse("wrong_book.html", {"request": request})


# ==================== API 接口 ====================

class TextSearchRequest(BaseModel):
    question_text: str
    username: str = DEFAULT_USER
    need_visualization: bool = False


class MarkWrongRequest(BaseModel):
    record_id: int
    username: str = DEFAULT_USER


# @app.post("/api/search/image")
# async def search_by_image(
#     file: UploadFile = File(...),
#     username: str = Form(DEFAULT_USER),
#     need_visualization: bool = Form(False),
# ):
#     """
#     接口1：上传题目图片 → 多模态识别 → 检索 + LLM 解答
#     """
#     # 保存上传图片
#     ext = Path(file.filename).suffix or ".jpg"
#     filename = f"{uuid.uuid4().hex}{ext}"
#     save_path = str(Path(UPLOAD_DIR) / filename)
#
#     with open(save_path, "wb") as f:
#         f.write(await file.read())
#
#     try:
#         # Step 1: 多模态图片识别
#         logger.info(f"识别图片: {filename}")
#         question_text = extract_question_from_image(save_path)
#
#         if not question_text.strip():
#             raise HTTPException(status_code=400, detail="图片中未识别到题目文字")
#
#         # Step 2-4: 检索 + 解答 + 保存
#         return await _solve_and_save(
#             question_text=question_text,
#             username=username,
#             image_path=save_path,
#             need_visualization=need_visualization,
#         )
#
#     except Exception as e:
#         logger.exception("图片搜题失败")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    username: str = Form(DEFAULT_USER),
    need_visualization: bool = Form(False),
):
    """
    接口1：上传题目图片 → 多模态识别 → 检索 + LLM 解答
    若图片含图表则多模态直接解题，否则交给 LLM
    """
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = str(Path(UPLOAD_DIR) / filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        # Step 1: 多模态识别（一次调用，同时判断是否含图表）
        logger.info(f"识别图片: {filename}")
        vl_result = extract_question_from_image(save_path)

        question_text = vl_result["question_text"]
        if not question_text.strip():
            raise HTTPException(status_code=400, detail="图片中未识别到题目文字")

        if vl_result.get("has_figure"):
            logger.info("图片含图表，多模态直接解题")
        else:
            logger.info("纯文字题目，交由 LLM 解题")

        # Step 2-4: 检索 + 解答 + 保存
        return await _solve_and_save(
            question_text=question_text,
            username=username,
            image_path=save_path,
            need_visualization=need_visualization,
            vl_answer=vl_result.get("vl_answer"),
            has_figure=vl_result.get("has_figure", False),
        )

    except Exception as e:
        logger.exception("图片搜题失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/text")
async def search_by_text(req: TextSearchRequest):
    """
    接口2：文字输入题目 → 检索 + LLM 解答
    """
    if not req.question_text.strip():
        raise HTTPException(status_code=400, detail="题目文字不能为空")

    return await _solve_and_save(
        question_text=req.question_text,
        username=req.username,
        image_path=None,
        need_visualization=req.need_visualization,
    )


async def _solve_and_save(
    question_text: str,
    username: str,
    image_path: Optional[str],
    need_visualization: bool,
    vl_answer: Optional[str] = None,  # ← 新增参数
    has_figure: bool = False,  # ← 新增参数
) -> dict:
    """公共逻辑：检索 → LLM 解答 → 保存记录 → 返回结果"""
    user = get_or_create_user(username)

    # Step 2: 向量检索题库
    logger.info(f"检索题库: {question_text[:50]}...")
    retrieved = retrieve(question_text)

    # 最佳匹配题目
    best_match = retrieved[0] if retrieved else None
    similarity = best_match["similarity"] if best_match else 0.0

    # Step 3: LLM 解答（RAG）
    if has_figure and vl_answer:
        logger.info("图片含图表，使用多模态直接解答")
        llm_result = {
            "llm_answer": vl_answer,
            "thinking": "",
            "subject": "",
            "ques_type": "",
            "ques_difficulty": "一般",
            "knowledges": [],
        }
    else:
        llm_result = answer_question(question_text, retrieved)

    # llm_result = answer_question(question_text, retrieved)
    llm_answer = llm_result["llm_answer"]
    llm_thinking = llm_result.get("thinking", "")   # ← 新增

    # 修复可视化：判断条件放宽，只要勾选了就尝试生成
    viz_html = None
    is_program_question = _is_program_question(question_text, best_match)
    if need_visualization and is_program_question:
        logger.info("生成程序题可视化 HTML")
        viz_html = generate_visualization_html(question_text, llm_answer)

    record = save_solve_record(
        user_id=user.id,
        question_text=question_text,
        llm_answer=llm_answer,
        llm_thinking=llm_thinking,
        matched_question=best_match,
        similarity_score=similarity,
        image_path=image_path,
        visualization_html=viz_html,
        subject=best_match.get("subject", "") if best_match else llm_result.get("subject", ""),
        ques_type=best_match.get("ques_type", "") if best_match else llm_result.get("ques_type", ""),
        ques_difficulty=best_match.get("ques_difficulty", "") if best_match else llm_result.get("ques_difficulty", ""),
        knowledges=list(set(
            (best_match.get("ques_knowledges") or []) +
            (llm_result.get("knowledges") or [])
        )) if best_match else (llm_result.get("knowledges") or []),
    )

    # record = save_solve_record(
    #     user_id=user.id,
    #     question_text=question_text,
    #     llm_answer=llm_answer,
    #     llm_thinking=llm_thinking,                  # ← 新增
    #     matched_question=best_match,
    #     similarity_score=similarity,
    #     image_path=image_path,
    #     visualization_html=viz_html,
    #     subject=best_match.get("subject", "") if best_match else llm_result.get("subject", ""),
    #     ques_type=best_match.get("ques_type", "") if best_match else llm_result.get("ques_type", ""),
    #     ques_difficulty=best_match.get("ques_difficulty", "") if best_match else llm_result.get("ques_difficulty", ""),
    #     knowledges=list(set(
    #         (best_match.get("ques_knowledges") or []) +
    #         (llm_result.get("knowledges") or [])
    #     )) if best_match else (llm_result.get("knowledges") or []),
    # )

    # # Step 4: 程序题可视化（可选）

    return {
        "record_id": record.id,
        "question_text": question_text,
        "matched_from_bank": best_match is not None and similarity > 0.7,
        "similarity": round(similarity, 4),
        "matched_question": best_match,
        "llm_answer": llm_answer,
        "llm_thinking": llm_thinking,
        "knowledges": record.knowledges or [],
        "visualization_html": viz_html,
        "retrieved_references": retrieved[:3],
        "is_program_question": is_program_question,
        "answered_by_vl": has_figure and bool(vl_answer),  # 前端可据此显示"图表解析"标签
    }

    # return {
    #     "record_id": record.id,
    #     "question_text": question_text,
    #     "matched_from_bank": best_match is not None and similarity > 0.7,
    #     "similarity": round(similarity, 4),
    #     "matched_question": best_match,
    #     "llm_answer": llm_answer,
    #     "llm_thinking": llm_thinking,  # ← 新增，返回给前端
    #     "knowledges": record.knowledges or [],  # ← 加这行，让前端能拿到知识点
    #     "visualization_html": viz_html,
    #     "retrieved_references": retrieved[:3],
    #     "is_program_question": is_program_question,
    # }


def _is_program_question(text: str, matched: Optional[dict]) -> bool:
    """判断是否为程序题"""
    program_keywords = [
        '背包', '动态规划', '算法', '代码', '程序', '编程',
        '排序', '查找', '递归', '复杂度', '数据结构', '链表',
        '二叉树', '图论', 'python', 'java', 'c++', 'javascript',
        'dp', 'bfs', 'dfs', '贪心', '分治'
    ]
    text_lower = text.lower()
    for kw in program_keywords:
        if kw in text_lower:
            return True
    if matched:
        subject = matched.get("subject", "").lower()
        if "计算机" in subject or "编程" in subject:
            return True
    return False


class DeleteRequest(BaseModel):
    record_ids: List[int]
    username: str = "default"

class UnmarkWrongRequest(BaseModel):
    record_id: int
    username: str = "default"

@app.post("/api/mark-wrong")
async def mark_wrong(req: MarkWrongRequest):
    """接口3：标记错题"""
    user = get_or_create_user(req.username)
    mark_as_wrong(req.record_id, user.id)
    return {"success": True, "message": "已加入错题本"}


@app.get("/api/history")
async def get_history(username: str = DEFAULT_USER, limit: int = 20):
    """接口4：获取解题历史"""
    user = get_or_create_user(username)
    records = get_solve_history(user.id, limit)
    return {"records": records, "total": len(records)}


@app.get("/api/wrong-book")
async def get_wrong_book(username: str = DEFAULT_USER):
    """接口5：获取错题本"""
    user = get_or_create_user(username)
    wrong_questions = get_wrong_questions(user.id)
    return {"wrong_questions": wrong_questions, "total": len(wrong_questions)}


@app.get("/api/wrong-report")
async def get_wrong_report(username: str = DEFAULT_USER):
    """接口6：生成错题分析报告"""
    user = get_or_create_user(username)
    wrong_questions = get_wrong_questions(user.id)

    if not wrong_questions:
        return {"report": "暂无错题记录，请先做题并标记错题。", "knowledge_stats": []}

    report = generate_wrong_answer_report(wrong_questions, user.id)
    stats = get_knowledge_stats(user.id)

    return {
        "report": report,
        "knowledge_stats": stats,
        "total_wrong": len(wrong_questions),
    }


@app.get("/api/knowledge-stats")
async def get_stats(username: str = DEFAULT_USER):
    """接口7：获取知识点薄弱统计"""
    user = get_or_create_user(username)
    stats = get_knowledge_stats(user.id)
    return {"stats": stats}



@app.get("/api/cluster-analysis")
async def get_cluster_analysis(username: str = DEFAULT_USER, n_clusters: int = None):
    """接口8：KMeans 聚类薄弱知识点分析"""
    user = get_or_create_user(username)
    wrong_questions = get_wrong_questions(user.id)

    if not wrong_questions:
        return {"clusters": [], "total_wrong": 0, "message": "暂无错题记录"}

    clusters = cluster_weak_knowledge_points(wrong_questions, n_clusters)

    # 精简返回，不含完整records
    result = []
    for c in clusters:
        result.append({
            "cluster_id": c["cluster_id"],
            "label": c["label"],
            "knowledge_points": c["knowledge_points"],
            "knowledge_freq": c["knowledge_freq"],
            "wrong_count": c["wrong_count"],
            "severity": c["severity"],
            "subjects": c["subjects"],
        })

    return {
        "clusters": result,
        "total_wrong": len(wrong_questions),
        "n_clusters": len(result),
    }


@app.get("/api/practice-plan")
async def get_practice_plan(username: str = DEFAULT_USER, questions_per_cluster: int = 3):
    """接口9：基于聚类生成个性化练习计划"""
    user = get_or_create_user(username)
    wrong_questions = get_wrong_questions(user.id)

    if not wrong_questions:
        return {"plan": [], "message": "暂无错题记录，请先做题并标记错题。"}

    plan = generate_cluster_practice_plan(wrong_questions, questions_per_cluster)

    return {
        "plan": plan,
        "total_clusters": len(plan),
        "total_questions": sum(len(p["practice_questions"]) for p in plan),
    }


@app.post("/api/delete-records")
async def delete_records_api(req: DeleteRequest):
    """接口10：批量删除解题记录"""
    user = get_or_create_user(req.username)
    deleted = delete_records(req.record_ids, user.id)
    return {"success": True, "deleted": deleted}


@app.post("/api/unmark-wrong")
async def unmark_wrong_api(req: UnmarkWrongRequest):
    """接口11：从错题本移除（取消错题标记）"""
    user = get_or_create_user(req.username)
    unmark_wrong(req.record_id, user.id)
    return {"success": True}


@app.get("/health")
async def health():
    return {"status": "ok"}
