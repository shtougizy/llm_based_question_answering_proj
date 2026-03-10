"""
数据库层：SQLAlchemy + SQLite
- 用户表
- 解题记录表（含错题标记）
- 知识点错误统计表
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Float, Boolean, DateTime, ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

Base = declarative_base()
engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


# ==================== 数据库模型 ====================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    records = relationship("SolveRecord", back_populates="user")
    knowledge_stats = relationship("KnowledgeStat", back_populates="user")


class SolveRecord(Base):
    """解题记录表"""
    __tablename__ = "solve_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    image_path = Column(String(256))              # 上传图片路径（可为空，如文字输入）
    question_text = Column(Text, nullable=False)  # 提取/输入的题目文本
    subject = Column(String(32))                  # 学科
    ques_type = Column(String(32))                # 题型
    ques_difficulty = Column(String(16))          # 难度

    # 匹配到的题库题目（JSON）
    matched_question = Column(JSON)
    similarity_score = Column(Float)

    # LLM 生成的解答
    llm_answer = Column(Text)
    llm_thinking = Column(Text, default="")  # ← 新增

    # 用户标记
    is_wrong = Column(Boolean, default=False)     # 是否错题
    user_answer = Column(Text)                    # 用户自己的答案（可选）

    # 知识点（JSON 数组）
    knowledges = Column(JSON)

    # 可视化 HTML（程序题）
    visualization_html = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="records")


class KnowledgeStat(Base):
    """知识点错误统计表"""
    __tablename__ = "knowledge_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    knowledge = Column(String(128), nullable=False)
    wrong_count = Column(Integer, default=1)
    subject = Column(String(32))

    user = relationship("User", back_populates="knowledge_stats")


# ==================== 数据库操作函数 ====================

def init_db():
    """初始化数据库（创建所有表）"""
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    logger.info(f"数据库初始化完成: {SQLITE_DB_PATH}")

    # 创建默认用户
    with SessionLocal() as db:
        if not db.query(User).filter_by(username="default").first():
            db.add(User(username="default"))
            db.commit()


def get_or_create_user(username: str) -> User:
    with SessionLocal() as db:
        user = db.query(User).filter_by(username=username).first()
        if not user:
            user = User(username=username)
            db.add(user)
            db.commit()
            db.refresh(user)
        return user


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
    llm_thinking: str = "",
    subject: str = "",
    ques_type: str = "",
    ques_difficulty: str = "",
) -> SolveRecord:
    """保存一条解题记录"""
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

        # 如果标记为错题，更新知识点统计
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
    """标记某条解题记录为错题"""
    with SessionLocal() as db:
        record = db.query(SolveRecord).filter_by(id=record_id, user_id=user_id).first()
        if record and not record.is_wrong:
            record.is_wrong = True
            db.commit()
            knowledges = record.knowledges or []
            if knowledges:
                _update_knowledge_stats(db, user_id, knowledges, record.subject)


def get_wrong_questions(user_id: int, limit: int = 50) -> List[Dict]:
    """获取用户错题列表"""
    with SessionLocal() as db:
        records = (
            db.query(SolveRecord)
            .filter_by(user_id=user_id, is_wrong=True)
            .order_by(SolveRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_record_to_dict(r) for r in records]


def get_solve_history(user_id: int, limit: int = 20) -> List[Dict]:
    """获取用户解题历史"""
    with SessionLocal() as db:
        records = (
            db.query(SolveRecord)
            .filter_by(user_id=user_id)
            .order_by(SolveRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_record_to_dict(r) for r in records]


def get_knowledge_stats(user_id: int) -> List[Dict]:
    """获取用户知识点薄弱统计"""
    with SessionLocal() as db:
        stats = (
            db.query(KnowledgeStat)
            .filter_by(user_id=user_id)
            .order_by(KnowledgeStat.wrong_count.desc())
            .all()
        )
        return [{"knowledge": s.knowledge, "wrong_count": s.wrong_count, "subject": s.subject} for s in stats]


def _record_to_dict(record: SolveRecord) -> Dict:
    return {
        "id": record.id,
        "question_text": record.question_text,
        "subject": record.subject,
        "ques_type": record.ques_type,
        "ques_difficulty": record.ques_difficulty,
        "llm_answer": record.llm_answer,
        "llm_thinking": record.llm_thinking or "",  # ← 新增
        "matched_question": record.matched_question,
        "similarity_score": record.similarity_score,
        "is_wrong": record.is_wrong,
        "knowledges": record.knowledges or [],
        "visualization_html": record.visualization_html,
        "created_at": record.created_at.isoformat() if record.created_at else "",
    }


def delete_records(record_ids: List[int], user_id: int) -> int:
    """批量删除解题记录（只能删自己的）"""
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
    """从错题本移除（取消 is_wrong 标记）"""
    with SessionLocal() as db:
        record = db.query(SolveRecord).filter_by(id=record_id, user_id=user_id).first()
        if record and record.is_wrong:
            record.is_wrong = False
            db.commit()
            # 同步减少知识点统计
            knowledges = record.knowledges or []
            for k in knowledges:
                stat = db.query(KnowledgeStat).filter_by(user_id=user_id, knowledge=k).first()
                if stat:
                    stat.wrong_count = max(0, stat.wrong_count - 1)
                    if stat.wrong_count == 0:
                        db.delete(stat)
            db.commit()
