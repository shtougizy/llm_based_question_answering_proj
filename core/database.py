"""
database.py 完整替换版
新增：
  - questions 题库表
  - user_auth 用户认证表（为登录注册准备）
  - uploaded_images 上传图片表
保留：
  - users / solve_records / knowledge_stats（原有表结构不变）
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Float, Boolean, DateTime, ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

Base = declarative_base()
engine = create_engine(
    f"sqlite:///{SQLITE_DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)


def _parse_json_list(val) -> list:
    """把 JSON 字符串或 list 统一转为 list"""
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        import json as _json
        result = _json.loads(val)
        return result if isinstance(result, list) else [result]
    except Exception:
        return [val]

# ==================== 题库表 ====================

class Question(Base):
    """题目表（从 faiss_meta.json 迁移）"""
    __tablename__ = "questions"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    source_id   = Column(String(128), unique=True, nullable=False, index=True)  # 原 JSON id
    subject     = Column(String(32), index=True)
    ques_type   = Column(String(32), index=True)
    ques_difficulty = Column(String(16), index=True)
    ques_content    = Column(Text, nullable=False)
    ques_answer     = Column(Text)          # JSON 字符串，list[str]
    ques_analyze    = Column(Text)
    ques_knowledges = Column(Text)          # JSON 字符串，list[str]
    rag_search_text = Column(Text)          # 用于全文检索辅助
    created_at  = Column(DateTime, default=datetime.utcnow)


class QuestionKnowledge(Base):
    """知识点索引表（方便按知识点查题）"""
    __tablename__ = "question_knowledges"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, index=True)
    knowledge   = Column(String(128), nullable=False, index=True)
    subject     = Column(String(32))


# ==================== 用户认证表 ====================

class UserAuth(Base):
    """
    用户认证表（为登录注册准备）
    与 users 表一对一关联
    """
    __tablename__ = "user_auth"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    email           = Column(String(128), unique=True, index=True)
    hashed_password = Column(String(256))           # bcrypt hash，暂留空
    role            = Column(String(16), default="student")  # student / teacher / admin
    is_active       = Column(Boolean, default=True)
    last_login      = Column(DateTime)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    openid          = Column(String(128), unique=True, index=True)


# ==================== 上传图片表 ====================

class UploadedImage(Base):
    """上传图片记录表"""
    __tablename__ = "uploaded_images"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename    = Column(String(256), nullable=False)
    filepath    = Column(String(512), nullable=False)
    file_size   = Column(Integer)                   # bytes
    mime_type   = Column(String(64))
    record_id   = Column(Integer, ForeignKey("solve_records.id"), index=True)  # 关联解题记录
    created_at  = Column(DateTime, default=datetime.utcnow)


# ==================== 原有表（保持不变）====================

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    records         = relationship("SolveRecord", back_populates="user")
    knowledge_stats = relationship("KnowledgeStat", back_populates="user")


class SolveRecord(Base):
    __tablename__ = "solve_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_path      = Column(String(256))
    question_text   = Column(Text, nullable=False)
    subject         = Column(String(32))
    ques_type       = Column(String(32))
    ques_difficulty = Column(String(16))
    matched_question    = Column(JSON)
    similarity_score    = Column(Float)
    llm_answer          = Column(Text)
    llm_thinking        = Column(Text, default="")
    is_wrong            = Column(Boolean, default=False)
    user_answer         = Column(Text)
    knowledges          = Column(JSON)
    visualization_html  = Column(Text)
    created_at          = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="records")


class KnowledgeStat(Base):
    __tablename__ = "knowledge_stats"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    knowledge   = Column(String(128), nullable=False)
    wrong_count = Column(Integer, default=1)
    subject     = Column(String(32))

    user = relationship("User", back_populates="knowledge_stats")


# ==================== 数据库初始化 ====================

def init_db():
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    logger.info(f"数据库初始化完成: {SQLITE_DB_PATH}")
    with SessionLocal() as db:
        if not db.query(User).filter_by(username="default").first():
            user = User(username="default")
            db.add(user)
            db.commit()
            db.refresh(user)
            # 为 default 用户创建 auth 记录
            db.add(UserAuth(user_id=user.id, username="default", role="student"))
            db.commit()


# ==================== 题库操作 ====================

def get_question_by_source_id(source_id: str) -> Optional[Dict]:
    with SessionLocal() as db:
        q = db.query(Question).filter_by(source_id=source_id).first()
        return _question_to_dict(q) if q else None


def get_questions_by_knowledge(knowledge: str, limit: int = 10) -> List[Dict]:
    with SessionLocal() as db:
        qks = db.query(QuestionKnowledge).filter_by(knowledge=knowledge).limit(limit).all()
        result = []
        for qk in qks:
            q = db.query(Question).get(qk.question_id)
            if q:
                result.append(_question_to_dict(q))
        return result


def get_questions_by_subject(subject: str, limit: int = 20) -> List[Dict]:
    with SessionLocal() as db:
        qs = db.query(Question).filter_by(subject=subject).limit(limit).all()
        return [_question_to_dict(q) for q in qs]


def search_questions_by_ids(source_ids: List[str]) -> List[Dict]:
    """根据 source_id 列表批量查题（检索后还原完整题目数据）"""
    with SessionLocal() as db:
        qs = db.query(Question).filter(Question.source_id.in_(source_ids)).all()
        mapping = {q.source_id: _question_to_dict(q) for q in qs}
        return [mapping[sid] for sid in source_ids if sid in mapping]


def _question_to_dict(q: Question) -> Dict:
    return {
        "id": q.source_id,
        "subject": q.subject or "",
        "ques_type": q.ques_type or "",
        "ques_difficulty": q.ques_difficulty or "",
        "ques_content": q.ques_content or "",
        "ques_answer": _parse_json_list(q.ques_answer),
        "ques_analyze": q.ques_analyze or "",
        "ques_knowledges": _parse_json_list(q.ques_knowledges),
        "rag_search_text": q.rag_search_text or "",
        # 兼容旧代码的 display_data 格式
        "display_data": {
            "subject": q.subject or "",
            "ques_type": q.ques_type or "",
            "ques_difficulty": q.ques_difficulty or "",
            "ques_content": q.ques_content or "",
            "ques_answer": _parse_json_list(q.ques_answer),
            "ques_analyze": q.ques_analyze or "",
            "ques_knowledges": _parse_json_list(q.ques_knowledges),
        }
    }


# ==================== 用户管理操作 ====================

def get_or_create_user(username: str) -> User:
    with SessionLocal() as db:
        user = db.query(User).filter_by(username=username).first()
        if not user:
            user = User(username=username)
            db.add(user)
            db.commit()
            db.refresh(user)
            db.add(UserAuth(user_id=user.id, username=username, role="student"))
            db.commit()
        return user


def get_all_users() -> List[Dict]:
    with SessionLocal() as db:
        users = db.query(User).all()
        result = []
        for u in users:
            auth = db.query(UserAuth).filter_by(user_id=u.id).first()
            result.append({
                "id": u.id,
                "username": u.username,
                "email": auth.email if auth else "",
                "role": auth.role if auth else "student",
                "is_active": auth.is_active if auth else True,
                "created_at": u.created_at.isoformat() if u.created_at else "",
                "last_login": auth.last_login.isoformat() if auth and auth.last_login else "",
            })
        return result


def create_user(username: str, email: str = "", role: str = "student") -> Optional[User]:
    with SessionLocal() as db:
        if db.query(User).filter_by(username=username).first():
            return None  # 已存在
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(UserAuth(user_id=user.id, username=username, email=email or None, role=role))
        db.commit()
        return user


def update_user_last_login(user_id: int):
    with SessionLocal() as db:
        auth = db.query(UserAuth).filter_by(user_id=user_id).first()
        if auth:
            auth.last_login = datetime.utcnow()
            db.commit()


# ==================== 图片管理操作 ====================

def save_uploaded_image(
    user_id: int,
    filename: str,
    filepath: str,
    file_size: int = 0,
    mime_type: str = "",
    record_id: int = None,
) -> UploadedImage:
    with SessionLocal() as db:
        img = UploadedImage(
            user_id=user_id,
            filename=filename,
            filepath=filepath,
            file_size=file_size,
            mime_type=mime_type,
            record_id=record_id,
        )
        db.add(img)
        db.commit()
        db.refresh(img)
        return img


def get_user_images(user_id: int, limit: int = 20) -> List[Dict]:
    with SessionLocal() as db:
        imgs = db.query(UploadedImage).filter_by(user_id=user_id)\
                 .order_by(UploadedImage.created_at.desc()).limit(limit).all()
        return [{
            "id": i.id,
            "filename": i.filename,
            "filepath": i.filepath,
            "file_size": i.file_size,
            "record_id": i.record_id,
            "created_at": i.created_at.isoformat() if i.created_at else "",
        } for i in imgs]


# ==================== 原有操作函数（保持不变）====================

def save_solve_record(
    user_id: int,
    question_text: str,
    llm_answer: str,
    matched_question: Optional[Dict] = None,
    similarity_score: float = 0.0,
    image_path: Optional[str] = None,
    is_wrong: bool = False,
    knowledges: Optional[List[str]] = None,
    visualization_html: Optional[str] = None,
    subject: str = "",
    ques_type: str = "",
    ques_difficulty: str = "",
    llm_thinking: str = "",
) -> SolveRecord:
    knowledges = knowledges or []
    if matched_question:
        subject = subject or matched_question.get("subject", "")
        ques_type = ques_type or matched_question.get("ques_type", "")
        ques_difficulty = ques_difficulty or matched_question.get("ques_difficulty", "")
        if not knowledges:
            knowledges = matched_question.get("ques_knowledges", [])

    record = SolveRecord(
        user_id=user_id,
        question_text=question_text,
        llm_answer=llm_answer,
        llm_thinking=llm_thinking,
        matched_question=matched_question,
        similarity_score=similarity_score,
        image_path=image_path,
        subject=subject,
        ques_type=ques_type,
        ques_difficulty=ques_difficulty,
        is_wrong=is_wrong,
        knowledges=knowledges,
        visualization_html=visualization_html,
    )
    with SessionLocal() as db:
        db.add(record)
        db.commit()
        db.refresh(record)
        if is_wrong and knowledges:
            _update_knowledge_stats(db, user_id, knowledges, subject)
        return record


def _update_knowledge_stats(db: Session, user_id: int, knowledges: List[str], subject: str):
    for k in knowledges:
        stat = db.query(KnowledgeStat).filter_by(user_id=user_id, knowledge=k).first()
        if stat:
            stat.wrong_count += 1
        else:
            db.add(KnowledgeStat(user_id=user_id, knowledge=k, subject=subject))
    db.commit()


def mark_as_wrong(record_id: int, user_id: int):
    with SessionLocal() as db:
        record = db.query(SolveRecord).filter_by(id=record_id, user_id=user_id).first()
        if record and not record.is_wrong:
            record.is_wrong = True
            db.commit()
            knowledges = record.knowledges or []
            if knowledges:
                _update_knowledge_stats(db, user_id, knowledges, record.subject)


def get_wrong_questions(user_id: int, limit: int = 50) -> List[Dict]:
    with SessionLocal() as db:
        records = db.query(SolveRecord)\
            .filter_by(user_id=user_id, is_wrong=True)\
            .order_by(SolveRecord.created_at.desc())\
            .limit(limit).all()
        return [_record_to_dict(r) for r in records]


def get_solve_history(user_id: int, limit: int = 20) -> List[Dict]:
    with SessionLocal() as db:
        records = db.query(SolveRecord)\
            .filter_by(user_id=user_id)\
            .order_by(SolveRecord.created_at.desc())\
            .limit(limit).all()
        return [_record_to_dict(r) for r in records]


def get_knowledge_stats(user_id: int) -> List[Dict]:
    with SessionLocal() as db:
        stats = db.query(KnowledgeStat)\
            .filter_by(user_id=user_id)\
            .order_by(KnowledgeStat.wrong_count.desc()).all()
        return [{"knowledge": s.knowledge, "wrong_count": s.wrong_count, "subject": s.subject}
                for s in stats]


def _record_to_dict(record: SolveRecord) -> Dict:
    return {
        "id": record.id,
        "question_text": record.question_text,
        "subject": record.subject,
        "ques_type": record.ques_type,
        "ques_difficulty": record.ques_difficulty,
        "llm_answer": record.llm_answer,
        "llm_thinking": record.llm_thinking or "",
        "matched_question": record.matched_question,
        "similarity_score": record.similarity_score,
        "is_wrong": record.is_wrong,
        "knowledges": record.knowledges or [],
        "visualization_html": record.visualization_html,
        "created_at": record.created_at.isoformat() if record.created_at else "",
    }


def delete_records(record_ids: list, user_id: int) -> int:
    """删除解题记录，返回删除数量"""
    with SessionLocal() as db:
        deleted = 0
        for rid in record_ids:
            record = db.query(SolveRecord).filter_by(id=rid, user_id=user_id).first()
            if record:
                db.delete(record)
                deleted += 1
        db.commit()
        return deleted


def unmark_wrong(record_id: int, user_id: int):
    """取消错题标记"""
    with SessionLocal() as db:
        record = db.query(SolveRecord).filter_by(id=record_id, user_id=user_id).first()
        if record and record.is_wrong:
            record.is_wrong = False
            db.commit()
            # 更新知识点统计（减少计数）
            knowledges = record.knowledges or []
            for k in knowledges:
                stat = db.query(KnowledgeStat).filter_by(user_id=user_id, knowledge=k).first()
                if stat:
                    stat.wrong_count = max(0, stat.wrong_count - 1)
                    if stat.wrong_count == 0:
                        db.delete(stat)
            db.commit()

def get_or_create_user_by_openid(openid: str) -> tuple:
    with SessionLocal() as db:
        auth = db.query(UserAuth).filter_by(openid=openid).first()
        if auth:
            user = db.query(User).filter_by(id=auth.user_id).first()
            # 在 session 关闭前把属性读出来
            user_id = user.id
            username = user.username
            # 构造一个简单对象返回，避免 DetachedInstanceError
            class UserInfo:
                pass
            u = UserInfo()
            u.id = user_id
            u.username = username
            return u, False

        username = f"wx_{openid[-8:]}"
        i = 0
        base = username
        while db.query(User).filter_by(username=username).first():
            i += 1
            username = f"{base}_{i}"

        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
        user_id = user.id
        username = user.username

        auth = UserAuth(
            user_id=user_id,
            username=username,
            openid=openid,
            role="student",
            is_active=True,
        )
        db.add(auth)
        db.commit()

        class UserInfo:
            pass
        u = UserInfo()
        u.id = user_id
        u.username = username
        return u, True

# def get_or_create_user_by_openid(openid: str) -> tuple:
#     """通过微信 openid 查找或创建用户，返回 (user, is_new)"""
#     with SessionLocal() as db:
#         auth = db.query(UserAuth).filter_by(openid=openid).first()
#         if auth:
#             user = db.query(User).get(auth.user_id)
#             return user, False
#
#         # 新用户：用 openid 后8位生成用户名
#         username = f"wx_{openid[-8:]}"
#         # 避免重名
#         i = 0
#         base = username
#         while db.query(User).filter_by(username=username).first():
#             i += 1
#             username = f"{base}_{i}"
#
#         user = User(username=username)
#         db.add(user)
#         db.commit()
#         db.refresh(user)
#
#         auth = UserAuth(
#             user_id=user.id,
#             username=username,
#             openid=openid,
#             role="student",
#             is_active=True,
#         )
#         db.add(auth)
#         db.commit()
#         return user, True


def get_user_by_openid(openid: str):
    """通过 openid 查找用户"""
    with SessionLocal() as db:
        auth = db.query(UserAuth).filter_by(openid=openid).first()
        if not auth:
            return None
        return db.query(User).get(auth.user_id)
