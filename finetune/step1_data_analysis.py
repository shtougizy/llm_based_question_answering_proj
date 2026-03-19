"""
Step 1: 数据质量分析与清洗
- 统计数据质量问题
- 过滤低质量样本
- 输出清洗报告
"""
import sqlite3
import json
import re
import os
from collections import Counter, defaultdict

DB_PATH = '/home/zsy/workspace/250606/code/work/260221_gd/data/app.db'
OUTPUT_DIR = '/home/zsy/workspace/250606/code/work/260221_gd/finetune'
os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT source_id, subject, ques_type, ques_difficulty, ques_content, ques_answer, ques_analyze, ques_knowledges FROM questions")
rows = cur.fetchall()
conn.close()

print(f"原始数据总量: {len(rows)}")

def parse_json(val):
    if not val: return []
    if isinstance(val, list): return val
    try:
        r = json.loads(val)
        return r if isinstance(r, list) else [r]
    except: return [val] if val else []

stats = {
    'total': len(rows),
    'no_content': 0,
    'no_answer': 0,
    'no_analyze': 0,
    'content_too_short': 0,
    'content_too_long': 0,
    'duplicate': 0,
    'kept': 0,
}

seen_contents = set()
cleaned = []

for row in rows:
    source_id, subject, ques_type, difficulty, content, answer_raw, analyze, kns_raw = row

    # 解析 JSON 字段
    answers = parse_json(answer_raw)
    kns = parse_json(kns_raw)

    # 过滤条件
    if not content or len(content.strip()) < 10:
        stats['no_content'] += 1; continue
    if not answers or all(not a for a in answers):
        stats['no_answer'] += 1; continue
    if not analyze or len(analyze.strip()) < 20:
        stats['no_analyze'] += 1; continue
    if len(content) < 20:
        stats['content_too_short'] += 1; continue
    if len(content) > 2000:
        stats['content_too_long'] += 1; continue

    # 去重（基于题目内容前100字）
    dedup_key = content[:100].strip()
    if dedup_key in seen_contents:
        stats['duplicate'] += 1; continue
    seen_contents.add(dedup_key)

    cleaned.append({
        'source_id': source_id,
        'subject': subject or '',
        'ques_type': ques_type or '',
        'difficulty': difficulty or '一般',
        'content': content.strip(),
        'answers': answers,
        'analyze': analyze.strip(),
        'knowledges': kns,
    })

stats['kept'] = len(cleaned)

print("\n=== 数据清洗报告 ===")
print(f"原始总量:     {stats['total']}")
print(f"无内容:       {stats['no_content']}")
print(f"无答案:       {stats['no_answer']}")
print(f"无解析:       {stats['no_analyze']}")
print(f"内容过短:     {stats['content_too_short']}")
print(f"内容过长:     {stats['content_too_long']}")
print(f"重复:         {stats['duplicate']}")
print(f"清洗后保留:   {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")

# 学科分布
subj_counter = Counter(d['subject'] for d in cleaned)
print("\n=== 清洗后学科分布 ===")
for subj, cnt in subj_counter.most_common():
    print(f"  {subj}: {cnt}")

# 题型分布
type_counter = Counter(d['ques_type'] for d in cleaned)
print("\n=== 清洗后题型分布 ===")
for t, cnt in type_counter.most_common():
    print(f"  {t}: {cnt}")

# 保存清洗后数据
out_path = os.path.join(OUTPUT_DIR, 'cleaned_data.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)
print(f"\n清洗后数据已保存至: {out_path}")
