"""
数据迁移脚本：把 faiss_meta.json 导入 SQLite 的 questions 表
使用原生 SQL 插入，避免 SQLAlchemy JSON 列类型问题
"""
import json, sys, logging, sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import FAISS_META_PATH, SQLITE_DB_PATH
from core.database import Base, engine

def ensure_list(val):
    if val is None: return []
    if isinstance(val, list): return val
    if isinstance(val, str):
        try:
            r = json.loads(val)
            return r if isinstance(r, list) else [r]
        except: return [val]
    return [str(val)]

def _insert_batch(cur, conn, batch):
    """批量插入，失败时逐条插入并跳过有问题的记录"""
    SQL = ("INSERT OR IGNORE INTO questions "
           "(source_id,subject,ques_type,ques_difficulty,ques_content,"
           "ques_answer,ques_analyze,ques_knowledges,rag_search_text,created_at) "
           "VALUES (?,?,?,?,?,?,?,?,?,?)")
    try:
        cur.executemany(SQL, batch)
        conn.commit()
    except Exception:
        conn.rollback()
        for row in batch:
            # 确保所有字段都是字符串或 None
            safe_row = tuple(
                str(v) if v is not None and not isinstance(v, (str, int, float, bytes)) else v
                for v in row
            )
            try:
                cur.execute(SQL, safe_row)
            except Exception as e2:
                logger.warning(f"跳过问题记录 {safe_row[0]}: {e2}")
        conn.commit()

def migrate():
    Base.metadata.create_all(engine)
    logger.info("表结构已创建/更新")

    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"读取 faiss_meta.json，共 {len(data)} 条")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT source_id FROM questions")
    existing = set(row[0] for row in cur.fetchall())
    logger.info(f"数据库中已有 {len(existing)} 条，将跳过")

    now = datetime.utcnow().isoformat()
    batch, kn_batch = [], []
    inserted = skipped = 0

    for item in data:
        source_id = item.get("id", "")
        if source_id in existing:
            skipped += 1
            continue

        dd   = item.get("display_data") or {}
        meta = item.get("metadata") or {}

        subject      = dd.get("subject") or meta.get("subject", "") or ""
        ques_type    = dd.get("ques_type") or meta.get("ques_type", "") or ""
        ques_diff    = dd.get("ques_difficulty") or meta.get("difficulty", "") or ""
        ques_content = dd.get("ques_content", "") or ""
        ques_answer  = ensure_list(dd.get("ques_answer"))
        ques_analyze = dd.get("ques_analyze", "") or ""
        ques_kns     = ensure_list(dd.get("ques_knowledges") or meta.get("knowledges"))
        rag_text     = item.get("rag_search_text", "") or ""

        batch.append((
            source_id, subject, ques_type, ques_diff, ques_content,
            json.dumps(ques_answer, ensure_ascii=False),
            ques_analyze,
            json.dumps(ques_kns, ensure_ascii=False),
            rag_text, now,
        ))
        for kn in ques_kns:
            if kn: kn_batch.append((source_id, kn, subject))
        inserted += 1

        if len(batch) >= 500:
            _insert_batch(cur, conn, batch)
            logger.info(f"已导入 {inserted} 条...")
            batch = []

    if batch:
        _insert_batch(cur, conn, batch)

    logger.info(f"题目导入完成，共 {inserted} 条，跳过 {skipped} 条")
    logger.info("开始建立知识点索引...")

    kn_rows = []
    for source_id, kn, subject in kn_batch:
        cur.execute("SELECT id FROM questions WHERE source_id=?", (source_id,))
        row = cur.fetchone()
        if row:
            kn_rows.append((row[0], kn, subject))
        if len(kn_rows) >= 500:
            cur.executemany(
                "INSERT INTO question_knowledges (question_id,knowledge,subject) VALUES (?,?,?)",
                kn_rows)
            conn.commit()
            kn_rows = []
    if kn_rows:
        cur.executemany(
            "INSERT INTO question_knowledges (question_id,knowledge,subject) VALUES (?,?,?)",
            kn_rows)
        conn.commit()

    conn.close()
    logger.info("✅ 迁移完成！")

    conn2 = sqlite3.connect(SQLITE_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute("SELECT COUNT(*) FROM questions")
    print(f"验证：questions 表共 {cur2.fetchone()[0]} 条")
    cur2.execute("SELECT COUNT(*) FROM question_knowledges")
    print(f"验证：question_knowledges 表共 {cur2.fetchone()[0]} 条")
    conn2.close()

if __name__ == "__main__":
    migrate()
