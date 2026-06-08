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
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel  # 如果还没有的话

import asyncio
import uuid

app = FastAPI()

# 添加这个配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 调试时可以允许所有，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 确保上传目录存在
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="拍照搜题辅助学习系统", version="1.0.0")

from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse as StarletteJSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """安全处理文件上传场景的校验错误，避免二进制数据导致 UnicodeDecodeError"""
    safe_errors = []
    for err in exc.errors():
        safe_errors.append({
            "loc": [str(loc) for loc in err.get("loc", [])],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        })
    return StarletteJSONResponse(
        status_code=422,
        content={"detail": safe_errors},
    )

# 静态文件 & 模板
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))

# 默认用户 ID（演示用，实际可加入登录系统）
DEFAULT_USER = "default"


# @app.on_event("startup")
# async def startup():
#     init_db()
#     logger.info("系统启动完成")
@app.on_event("startup")
async def startup():
    _tts_warmup()  # 后台预热 TTS worker
    import asyncio
    import asyncio
    asyncio.create_task(cleanup_guest_users())
    loop = asyncio.get_event_loop()
    # 后台线程预加载，不阻塞服务启动
    loop.run_in_executor(None, _preload_models)

def _preload_models():
    from core.multimodal import _load_model
    from core.llm import _load_llm
    logger.info("预加载模型...")
    _load_llm()       # Qwen 先加载（快）
    _load_model()     # InternVL 后加载（慢）
    # 预热知识图谱邻接表
    try:
        from core.kg import build_adjacency
        build_adjacency()
        logger.info("知识图谱邻接表已预热")
    except Exception as e:
        logger.warning(f"知识图谱预热失败（将使用纯聚类模式）: {e}")
    logger.info("模型预加载完成")


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


# 任务存储（简单内存dict，重启后丢失）
_tasks = {}
_viz_store = {}  # 临时存储可视化HTML，key为uuid



@app.post("/api/search/image/async")
async def search_image_async(
    file: UploadFile = File(...),
    username: str = Form(default=DEFAULT_USER),
    need_visualization: bool = Form(default=False),
):
    """异步图片搜题：立即返回 task_id，前端轮询 /api/task/{task_id}"""
    # 保存图片
    ext = os.path.splitext(file.filename)[-1] or '.jpg'
    fname = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, fname)
    content = await file.read()
    with open(save_path, 'wb') as f:
        f.write(content)

    task_id = uuid.uuid4().hex
    _tasks[task_id] = {"status": "pending", "result": None, "error": None}

    # 后台执行（用线程池避免阻塞事件循环）
    import concurrent.futures
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def run_task_sync():
        """同步版本，在线程池里运行"""
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        try:
            vl_result = extract_question_from_image(save_path)
            question_text = vl_result.get("question_text", "")
            has_figure = vl_result.get("has_figure", False)
            vl_answer = vl_result.get("vl_answer")
            result = loop.run_until_complete(_solve_and_save(
                question_text=question_text,
                username=username,
                image_path=fname,
                need_visualization=need_visualization,
                vl_answer=vl_answer,
                has_figure=has_figure,
            ))
            _tasks[task_id]["status"] = "done"
            _tasks[task_id]["result"] = result
        except Exception as e:
            import traceback
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["error"] = str(e)
            logger.error(f"异步任务失败: {e}\n{traceback.format_exc()}")
        finally:
            loop.close()

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, run_task_sync)
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/task/{task_id}")
async def get_task_result(task_id: str):
    """轮询任务结果"""
    task = _tasks.get(task_id)
    if not task:
        return JSONResponse({"status": "error", "error": "任务不存在"}, status_code=404)
    if task["status"] == "error":
        return JSONResponse({"status": "error", "error": task["error"]}, status_code=500)
    return task


# @app.post("/api/search/image")    # ← 已废弃，不可取消注释！会被 Python 挂到下一个 def 上
# async def search_by_image(
#     file: UploadFile = File(...),
#     username: str = Form(DEFAULT_USER),
#     need_visualization: bool = Form(False),
# ):
#     """
#     接口1：上传题目图片 → 多模态识别 → 检索 + LLM 解答
#     若图片含图表则多模态直接解题，否则交给 LLM
#     """
#     ext = Path(file.filename).suffix or ".jpg"
#     filename = f"{uuid.uuid4().hex}{ext}"
#     save_path = str(Path(UPLOAD_DIR) / filename)
#
#     with open(save_path, "wb") as f:
#         f.write(await file.read())
#
#     try:
#         # Step 1: 多模态识别（一次调用，同时判断是否含图表）
#         logger.info(f"识别图片: {filename}")
#         vl_result = extract_question_from_image(save_path)
#
#         question_text = vl_result["question_text"]
#         if not question_text.strip():
#             raise HTTPException(status_code=400, detail="图片中未识别到题目文字")
#
#         if vl_result.get("has_figure"):
#             logger.info("图片含图表，多模态直接解题")
#         else:
#             logger.info("纯文字题目，交由 LLM 解题")
#
#         # Step 2-4: 检索 + 解答 + 保存
#         return await _solve_and_save(
#             question_text=question_text,
#             username=username,
#             image_path=save_path,
#             need_visualization=need_visualization,
#             vl_answer=vl_result.get("vl_answer"),
#             has_figure=vl_result.get("has_figure", False),
#         )
#
#     except Exception as e:
#         logger.exception("图片搜题失败")
#         raise HTTPException(status_code=500, detail=str(e))
#



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

    # 入口校验：题目文本为空或无效时直接返回
    if not question_text or not question_text.strip() or question_text.strip() == "无":
        return {
            "record_id": -1,
            "question_text": question_text or "",
            "matched_from_bank": False,
            "matched_question": None,
            "similarity": 0.0,
            "llm_answer": "未能从图片中识别到题目文字，请重新拍摄清晰的题目图片。",
            "llm_thinking": "",
            "knowledges": [],
            "visualization_html": None,
            "retrieved_references": [],
            "is_program_question": False,
            "answered_by_vl": False,
        }

    # Step 2: 向量检索题库
    logger.info(f"检索题库: {question_text[:50]}...")
    retrieved = retrieve(question_text)

    # 最佳匹配题目
    # best_match = retrieved[0] if retrieved else None
    # similarity = best_match["similarity"] if best_match else 0.0
    best_match = retrieved[0] if retrieved else None
    similarity = best_match["similarity"] if best_match else 0.0
    matched = best_match if similarity >= 0.75 else None

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
    # 修复可视化：判断条件放宽，只要勾选了就尝试生成
    viz_html = None
    is_program_question = _is_program_question(question_text, best_match)
    if need_visualization and is_program_question:
        logger.info("生成程序题可视化 HTML")
        viz_html = generate_visualization_html(question_text, llm_answer)
        # 确保是合法 HTML 字符串
        if not isinstance(viz_html, str) or len(viz_html.strip()) < 50:
            viz_html = None

    # 安全合并知识点，确保最终是 list
    # bank_kns = (best_match.get("ques_knowledges") or []) if matched else []
    # llm_kns = llm_result.get("knowledges") or []
    # if not isinstance(bank_kns, list): bank_kns = []
    # if not isinstance(llm_kns, list): llm_kns = []
    # merged_knowledges = list(set(bank_kns + llm_kns))
    #
    # record = save_solve_record(
    #     user_id=user.id,
    #     question_text=question_text,
    #     llm_answer=llm_answer,
    #     llm_thinking=llm_thinking,
    #     matched_question=best_match,
    #     similarity_score=similarity,
    #     image_path=image_path,
    #     visualization_html=viz_html,
    #     subject=matched.get("subject", "") if matched else llm_result.get("subject", ""),
    #     ques_type=matched.get("ques_type", "") if matched else llm_result.get("ques_type", ""),
    #     ques_difficulty=matched.get("ques_difficulty", "") if matched else llm_result.get("ques_difficulty", ""),
    #     # subject=best_match.get("subject", "") if best_match and similarity >= 0.75 else llm_result.get("subject", ""),
    #     # ques_type=best_match.get("ques_type", "") if best_match and similarity >= 0.75 else llm_result.get("ques_type", ""),
    #     # ques_difficulty=best_match.get("ques_difficulty", "") if best_match and similarity >= 0.75 else llm_result.get("ques_difficulty", ""),
    #     knowledges=merged_knowledges,  # ← 用安全合并后的结果
    # )

    bank_kns = (best_match.get("ques_knowledges") or []) if best_match else []
    # 反序列化 JSON 字符串（DB 中 Text 列存的是 JSON 字符串）
    if isinstance(bank_kns, str) and bank_kns.strip().startswith('['):
        try:
            bank_kns = _json.loads(bank_kns)
        except (_json.JSONDecodeError, TypeError):
            bank_kns = []
    if not isinstance(bank_kns, list):
        bank_kns = []

    record = save_solve_record(
        user_id=user.id,
        question_text=question_text,
        llm_answer=llm_answer,
        llm_thinking=llm_thinking,
        matched_question=best_match,
        similarity_score=similarity,
        image_path=image_path,
        visualization_html=viz_html,
        subject=best_match.get("subject", "") if best_match else "",
        ques_type=best_match.get("ques_type", "") if best_match else "",
        ques_difficulty=best_match.get("ques_difficulty", "") if best_match else "一般",
        knowledges=bank_kns,
    )

    # 兜底：如果 record.knowledges 是 JSON 字符串，反序列化为 list
    record_knowledges = record.knowledges
    if isinstance(record_knowledges, str):
        try:
            record_knowledges = _json.loads(record_knowledges)
        except (_json.JSONDecodeError, TypeError):
            record_knowledges = []
    if not isinstance(record_knowledges, list):
        record_knowledges = []

    return {
        "record_id": record.id,
        "question_text": question_text,
        "matched_from_bank": matched is not None,  # 用 matched 而不是 best_match
        "matched_question": matched,
        # "matched_from_bank": best_match is not None and similarity > 0.7,
        "similarity": round(similarity, 4),
        # "matched_question": best_match,
        "llm_answer": llm_answer,
        "llm_thinking": llm_thinking,
        "knowledges": record_knowledges,
        "visualization_html": viz_html,
        "retrieved_references": retrieved[:3],
        "is_program_question": is_program_question,
        "answered_by_vl": has_figure and bool(vl_answer),  # 前端可据此显示"图表解析"标签
    }



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
    """接口9：基于聚类生成个性化练习计划（支持知识图谱增强学习路径）"""
    user = get_or_create_user(username)
    wrong_questions = get_wrong_questions(user.id)

    if not wrong_questions:
        return {"plan": [], "message": "暂无错题记录，请先做题并标记错题。"}

    plan = generate_cluster_practice_plan(wrong_questions, questions_per_cluster)

    # 检查是否启用了 KG 增强
    kg_enabled = any(p.get("has_learning_path") for p in plan)

    return {
        "plan": plan,
        "total_clusters": len(plan),
        "total_questions": sum(len(p["practice_questions"]) for p in plan),
        "kg_enabled": kg_enabled,
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

# ==================== 认证依赖 ====================
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    from core.auth import decode_token
    payload = decode_token(credentials.credentials)
    return payload

def require_login(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="请先登录")
    from core.auth import decode_token
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")
    if payload.get("is_guest"):
        raise HTTPException(status_code=403, detail="游客无权限，请注册登录")
    return payload


# ==================== 注册接口 ====================

class RegisterRequest(BaseModel):
    username: str
    password: str
    phone: str
    sms_code: str

@app.post("/api/auth/send-sms")
async def send_sms_code(phone: str):
    import re
    if not re.match(r"^1[3-9]\d{9}$", phone):
        raise HTTPException(status_code=400, detail="手机号格式不正确")
    from core.auth import generate_sms_code, send_sms
    code = generate_sms_code(phone)
    send_sms(phone, code)
    return {"message": "验证码已发送", "dev_code": code}



@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    from core.auth import verify_sms_code, hash_password, create_access_token
    from core.database import SessionLocal, User, UserAuth

    if not verify_sms_code(req.phone, req.sms_code):
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    # 先计算好密码哈希（可能抛异常），再开始写库
    hashed = hash_password(req.password)

    with SessionLocal() as db:
        if db.query(UserAuth).filter_by(username=req.username).first():
            raise HTTPException(status_code=400, detail="用户名已存在")
        if db.query(UserAuth).filter_by(email=req.phone).first():
            raise HTTPException(status_code=400, detail="该手机号已注册")

        # 在同一个事务里写 user 和 auth，任何一个失败都整体回滚
        try:
            user = User(username=req.username)
            db.add(user)
            db.flush()  # 获取 user.id，但不提交

            auth = UserAuth(
                user_id=user.id,
                username=req.username,
                email=req.phone,
                hashed_password=hashed,
                role="student",
                is_active=True,
            )
            db.add(auth)
            db.commit()
            db.refresh(user)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"注册失败：{str(e)}")

        token = create_access_token(user.id, req.username, role="student")
        return {
            "token": token,
            "user": {"id": user.id, "username": req.username, "role": "student"},
            "message": "注册成功",
        }

# ==================== 登录接口 ====================

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    from core.auth import verify_password, create_access_token
    from core.database import SessionLocal, User, UserAuth
    from datetime import datetime
    with SessionLocal() as db:
        auth = db.query(UserAuth).filter_by(username=req.username).first()
        if not auth or not auth.hashed_password:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        if not auth.is_active:
            raise HTTPException(status_code=403, detail="账号已被禁用")
        from core.auth import verify_password
        if not verify_password(req.password, auth.hashed_password):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        auth.last_login = datetime.utcnow()
        db.commit()
        token = create_access_token(auth.user_id, auth.username, role=auth.role)
        return {"token": token, "user": {"id": auth.user_id, "username": auth.username, "role": auth.role}, "message": "登录成功"}


# ==================== 游客接口 ====================

@app.post("/api/auth/guest")
async def guest_login():
    import uuid
    from core.auth import create_access_token
    from core.database import SessionLocal, User, UserAuth
    guest_id = f"guest_{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        user = User(username=guest_id)
        db.add(user); db.commit(); db.refresh(user)
        db.add(UserAuth(user_id=user.id, username=guest_id, role="guest", is_active=True))
        db.commit()
        token = create_access_token(user.id, guest_id, role="guest", is_guest=True, expires_minutes=1440)
        return {"token": token, "user": {"id": user.id, "username": guest_id, "role": "guest"}}


@app.get("/api/auth/me")
async def get_me(payload: dict = Depends(get_current_user)):
    if not payload:
        raise HTTPException(status_code=401, detail="未登录")
    from core.database import SessionLocal, UserAuth
    with SessionLocal() as db:
        auth = db.query(UserAuth).filter_by(user_id=int(payload["sub"])).first()
        return {
            "id": int(payload["sub"]),
            "username": payload.get("username"),
            "role": payload.get("role"),
            "is_guest": payload.get("is_guest", False),
            "phone": auth.email if auth else "",
        }


# ==================== 游客清理（在 startup 事件里调用）====================

async def cleanup_guest_users():
    import asyncio
    from datetime import datetime, timedelta
    from core.database import SessionLocal, User, UserAuth, SolveRecord, KnowledgeStat
    while True:
        await asyncio.sleep(3600)
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            with SessionLocal() as db:
                guests = db.query(UserAuth).filter(
                    UserAuth.role == "guest",
                    UserAuth.created_at < cutoff,
                ).all()
                for g in guests:
                    uid = g.user_id
                    db.query(KnowledgeStat).filter_by(user_id=uid).delete()
                    db.query(SolveRecord).filter_by(user_id=uid).delete()
                    db.query(UserAuth).filter_by(user_id=uid).delete()
                    db.query(User).filter_by(id=uid).delete()
                db.commit()
                if guests:
                    logger.info(f"清理了 {len(guests)} 个过期游客用户")
        except Exception as e:
            logger.error(f"游客清理失败: {e}")


# ==================== 页面路由 ====================

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# ==================== 微信登录 ====================
import httpx as _httpx

@app.post("/api/auth/wechat-login")
async def wechat_login(code: str):
    from config import WECHAT_APPID, WECHAT_SECRET
    from core.database import get_or_create_user_by_openid
    from core.auth import create_access_token

    async with _httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.weixin.qq.com/sns/jscode2session",
            params={
                "appid": WECHAT_APPID,
                "secret": WECHAT_SECRET,
                "js_code": code,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
    data = resp.json()
    openid = data.get("openid")
    if not openid:
        raise HTTPException(status_code=400, detail=f"微信登录失败：{data.get('errmsg','未知错误')}")

    user, is_new = get_or_create_user_by_openid(openid)

    # 直接用返回的基础数据，不再访问 ORM 对象属性
    token = create_access_token(user.id, user.username, role="student")
    return {
        "token": token,
        "user": {"id": user.id, "username": user.username, "role": "student"},
        "is_new": is_new,
    }
# @app.post("/api/auth/wechat-login")
# async def wechat_login(code: str):
#     """微信小程序登录：用 code 换取 openid，返回 JWT"""
#     from config import WECHAT_APPID, WECHAT_SECRET
#     from core.database import get_or_create_user_by_openid
#     from core.auth import create_access_token
#
#     # 1. 用 code 换 openid
#     async with _httpx.AsyncClient() as client:
#         resp = await client.get(
#             "https://api.weixin.qq.com/sns/jscode2session",
#             params={
#                 "appid": WECHAT_APPID,
#                 "secret": WECHAT_SECRET,
#                 "js_code": code,
#                 "grant_type": "authorization_code",
#             },
#             timeout=10,
#         )
#     data = resp.json()
#     openid = data.get("openid")
#     if not openid:
#         raise HTTPException(status_code=400, detail=f"微信登录失败：{data.get('errmsg','未知错误')}")
#
#     # 2. 查找或创建用户
#     user, is_new = get_or_create_user_by_openid(openid)
#
#     # 3. 签发 JWT
#     token = create_access_token(user.id, user.username, role="student")
#     return {
#         "token": token,
#         "user": {"id": user.id, "username": user.username, "role": "student"},
#         "is_new": is_new,
#     }


@app.post("/api/viz/store")
async def store_viz(request: Request):
    """存储可视化HTML，返回访问key"""
    import uuid as _uuid
    body = await request.json()
    html = body.get("html", "")
    key = _uuid.uuid4().hex
    _viz_store[key] = html
    return {"key": key}

@app.get("/api/viz/{key}")
async def get_viz(key: str):
    """通过key获取可视化HTML页面"""
    from fastapi.responses import HTMLResponse
    html = _viz_store.get(key, "<h1>已过期或不存在</h1>")
    return HTMLResponse(content=html)


from core.tts import synthesize as _tts_synthesize, warmup as _tts_warmup
import base64 as _base64

@app.post("/api/tts")
async def text_to_speech(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "text不能为空")
    try:
        wav_bytes = await asyncio.get_event_loop().run_in_executor(
            None, _tts_synthesize, text
        )
        audio_b64 = _base64.b64encode(wav_bytes).decode()
        return {"audio_base64": audio_b64, "format": "wav"}
    except Exception as e:
        raise HTTPException(502, f"TTS合成失败：{str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}

# @app.get("/api/practice-by-knowledge")
# async def practice_by_knowledge(
#     knowledge: str,
#     username: str = DEFAULT_USER,
#     n: int = 3,
# ):
#     """按单个知识点从题库推荐练习题"""
#     user = get_or_create_user(username)
#     wrong_questions = get_wrong_questions(user.id)
#
#     # 构造一个只包含该知识点的虚拟 cluster
#     from core.analysis import recommend_practice_questions
#     cluster = {
#         "label": knowledge,
#         "knowledge_points": [knowledge],
#         "knowledge_freq": {knowledge: 1},
#         "wrong_count": 0,
#         "severity": "中",
#         "records": [],
#         "subjects": [],
#     }
#     questions = recommend_practice_questions(cluster, n_questions=n)
#
#     return {
#         "knowledge": knowledge,
#         "questions": questions,
#         "total": len(questions),
#     }

@app.get("/api/practice-by-knowledge")
async def practice_by_knowledge(
    knowledge: str,
    username: str = DEFAULT_USER,
    n: int = 3,
):
    from core.analysis import recommend_practice_questions
    from core.llm import generate_practice_questions

    user = get_or_create_user(username)

    cluster = {
        "label": knowledge,
        "knowledge_points": [knowledge],
        "knowledge_freq": {knowledge: 1},
        "wrong_count": 0,
        "severity": "中",
        "records": [],
        "subjects": [],
    }
    questions = recommend_practice_questions(cluster, n_questions=n)

    # 题库没有匹配，改用 LLM 生成
    if not questions:
        logger.info(f"题库无匹配，LLM 生成「{knowledge}」练习题")
        questions = generate_practice_questions(knowledge, n)
        ai_generated = True
    else:
        ai_generated = False

    return {
        "knowledge": knowledge,
        "questions": questions,
        "total": len(questions),
        "ai_generated": ai_generated,
    }


from fastapi.responses import StreamingResponse
import json as _json

@app.get("/api/practice-by-knowledge-stream")
async def practice_by_knowledge_stream(
    knowledge: str,
    username: str = DEFAULT_USER,
    n: int = 3,
):
    """按知识点流式生成练习题，每生成一题推送一次"""
    from core.analysis import recommend_practice_questions
    from core.llm import generate_one_practice_question

    async def generate():
        # 先尝试题库
        cluster = {
            "label": knowledge,
            "knowledge_points": [knowledge],
            "knowledge_freq": {knowledge: 1},
            "wrong_count": 0, "severity": "中",
            "records": [], "subjects": [],
        }
        bank_questions = recommend_practice_questions(cluster, n_questions=n)

        if bank_questions:
            for q in bank_questions:
                data = _json.dumps(q, ensure_ascii=False)
                yield f"data: {data}\n\n"
        else:
            # 题库无匹配，逐题调用 LLM 生成
            for i in range(n):
                q = generate_one_practice_question(knowledge)
                if q:
                    data = _json.dumps(q, ensure_ascii=False)
                    yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
