"""
语言模型推理模块：使用 Qwen3-1.7B-GGUF 进行题目解答
支持 RAG（检索增强生成）
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import QWEN_GGUF_PATH, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

_llm = None

import re


# def _strip_thinking(text: str) -> str:
#     """去除思考过程内容"""
#     # 去除 <think> 块
#     text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
#
#     # 去除 "好的，我需要..." 这类思考过程（出现在答案前面）
#     # 找到第一个 { 的位置，之前的内容如果像思考过程就丢弃
#     json_start = text.find('{')
#     if json_start > 50:  # { 前面有较多文字，说明有前导思考内容
#         text = text[json_start:]
#
#     return text.strip()
def _strip_thinking(text: str) -> str:
    """去除 <think> 标签，返回 (thinking, answer) 两部分"""
    thinking = ""
    # 提取 <think> 块
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        text = text[think_match.end():].strip()
    return text, thinking


def _load_llm():
    """懒加载 Qwen GGUF 模型"""
    global _llm
    if _llm is not None:
        return

    from llama_cpp import Llama

    if not Path(QWEN_GGUF_PATH).exists():
        raise FileNotFoundError(
            f"GGUF 模型文件不存在: {QWEN_GGUF_PATH}\n"
            "请下载 Qwen/Qwen3-1.7B-GGUF 并配置 QWEN_GGUF_PATH"
        )

    logger.info(f"加载 Qwen GGUF 模型: {QWEN_GGUF_PATH}")
    _llm = Llama(
        model_path=QWEN_GGUF_PATH,
        n_ctx=8192,
        n_threads=4,
        n_gpu_layers=-1,  # 如果有 GPU 则全量卸载，没有 GPU 则自动回退 CPU
        verbose=False,
        logits_all=False,
    )
    logger.info("Qwen 模型加载完成")


def build_rag_prompt(
    question_text: str,
    retrieved_questions: List[Dict[str, Any]]
) -> str:
    """
    构建 RAG prompt：将检索到的相似题目作为参考上下文
    """
    context_parts = []
    for i, q in enumerate(retrieved_questions[:3], 1):
        content = q.get("ques_content", "")
        answer = q.get("ques_answer", [])
        analyze = q.get("ques_analyze", "")
        answer_str = "、".join(answer) if isinstance(answer, list) else str(answer)

        context_parts.append(
            f"【参考题目{i}】\n{content}\n"
            f"答案：{answer_str}\n"
            f"解析：{analyze}"
        )

    context = "\n\n".join(context_parts)

    if context:
        prompt = f"""你是一位专业的学习辅助助手，擅长解答 K12 教育及计算机领域的题目。

以下是一些相似题目供参考：
{context}

---

现在请解答以下题目：
{question_text}

请给出：
1. 答案
2. 详细解析步骤
3. 涉及的知识点
"""
    else:
        prompt = f"""你是一位专业的学习辅助助手，擅长解答 K12 教育及计算机领域的题目。

请解答以下题目：
{question_text}

请给出：
1. 答案
2. 详细解析步骤
3. 涉及的知识点
"""
    return prompt


def answer_question(question_text: str, retrieved: list) -> dict:
    _load_llm()

    writing_keywords = ['写作', '作文', '写一篇', '写一段', '写出', '写信', '日记', '短文', '写文章', '议论文', '说明文', '记叙文']
    is_writing = any(kw in question_text for kw in writing_keywords)
    max_tokens = 2048 if is_writing else 800
    writing_hint = "请写完整文章，不少于600字。" if is_writing else ""

    context = ""
    if retrieved:
        context = "\n参考题目：\n"
        for i, r in enumerate(retrieved[:3]):
            context += f"{i+1}. {r.get('ques_content', '')}\n"
            if r.get('ques_answer'):
                context += f"   答案：{r['ques_answer']}\n"

    # 第一步：解答题目
    resp1 = _llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "/no_think 你是专业教师，用中文解答题目，回答简洁清晰，不要输出思考过程。"},
            {"role": "user", "content": f"请解答以下题目。{writing_hint}\n\n题目：{question_text}{context}"}
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    raw = resp1["choices"][0]["message"]["content"].strip()

    # 去掉 <think> 块（即使是空的也去掉）
    import re
    thinking = ""
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        raw = raw[think_match.end():].strip()

    answer_text = raw if raw else "暂无解答"

    # 第二步：提取元数据
    resp2 = _llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "/no_think 只输出指定格式，不要其他内容。"},
            {"role": "user", "content": f"根据题目输出以下信息，每项占一行：\n学科：\n题型：\n难度：\n知识点：（逗号分隔）\n\n题目：{question_text[:150]}"}
        ],
        max_tokens=80,
        temperature=0.1,
    )
    meta_raw = resp2["choices"][0]["message"]["content"].strip()
    meta_raw = re.sub(r'<think>.*?</think>', '', meta_raw, flags=re.DOTALL).strip()

    subject, ques_type, ques_difficulty, knowledges = "", "", "一般", []
    for line in meta_raw.split('\n'):
        line = line.strip()
        if line.startswith('学科：'):
            subject = line[3:].strip()
        elif line.startswith('题型：'):
            ques_type = line[3:].strip()
        elif line.startswith('难度：'):
            ques_difficulty = line[3:].strip()
        elif line.startswith('知识点：'):
            kns = line[4:].strip()
            knowledges = [k.strip() for k in kns.replace('、', ',').split(',') if k.strip()]

    print(f"[DEBUG] subject={subject}, knowledges={knowledges}")

    return {
        "llm_answer": answer_text,
        "thinking": thinking,
        "subject": subject,
        "ques_type": ques_type,
        "ques_difficulty": ques_difficulty,
        "knowledges": knowledges,
    }

# def answer_question(question_text: str, retrieved: list) -> dict:
#     _load_llm()
#
#     writing_keywords = ['写作', '作文', '写一篇', '写一段', '写出', '写信', '日记', '短文', '写文章', '议论文', '说明文', '记叙文']
#     is_writing = any(kw in question_text for kw in writing_keywords)
#     max_tokens = 2048 if is_writing else 800
#
#     context = ""
#     if retrieved:
#         context = "\n\n参考题目：\n"
#         for i, r in enumerate(retrieved[:3]):
#             context += f"{i+1}. {r.get('ques_content', '')}\n"
#             if r.get('ques_answer'):
#                 context += f"   答案：{r['ques_answer']}\n"
#
#     # 第一步：让模型自然作答
#     prompt = f"""你是一位专业教师，请用中文解答以下题目。{'要求写完整文章，不少于600字。' if is_writing else ''}
#
# 题目：{question_text}
# {context}
# 请给出答案和解析："""
#
#     response = _llm.create_completion(
#         prompt=prompt,
#         max_tokens=max_tokens,
#         temperature=0.3,
#         stop=["</s>", "<|im_end|>", "\n\n\n\n"],
#         echo=False,
#     )
#     raw = response["choices"][0]["text"].strip()
#
#     # response = _llm(
#     #     prompt=prompt,
#     #     max_tokens=max_tokens,
#     #     temperature=0.3,
#     #     stop=["</s>"],
#     # )
#     #
#     # raw = response["choices"][0]["text"].strip()
#
#     # 分离思考过程（<think> 标签 或 JSON前的自然语言）
#     import re
#     thinking = ""
#     think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
#     if think_match:
#         thinking = think_match.group(1).strip()
#         raw = raw[think_match.end():].strip()
#
#     # JSON前的自然语言思考（"好的，我需要..."之类）
#     # 找到正文起始：跳过开头的思考段落
#     lines = raw.split('\n')
#     answer_lines = []
#     skip_thinking = True
#     for line in lines:
#         # 思考段落特征：以"好的"、"首先"、"接下来"、"现在"、"我需要"开头
#         thinking_starters = ['好的，', '首先，', '接下来，', '现在，', '我需要', '我来', '让我']
#         if skip_thinking and any(line.startswith(s) for s in thinking_starters):
#             thinking = (thinking + '\n' + line).strip()
#             continue
#         skip_thinking = False
#         answer_lines.append(line)
#
#     answer_text = '\n'.join(answer_lines).strip()
#     if not answer_text:
#         answer_text = raw  # 全部都是思考，直接用原文
#
#     # 第二步：用第二次 LLM 调用提取结构化元数据（轻量调用）
#     meta_prompt = f"""根据以下题目，只输出学科、题型、难度、知识点，格式如下，不要其他内容：
# 学科：数学
# 题型：填空题
# 难度：一般
# 知识点：知识点1,知识点2,知识点3
#
# 题目：{question_text[:200]}"""
#
#     meta_resp = _llm(
#         prompt=meta_prompt,
#         max_tokens=80,
#         temperature=0.1,
#         stop=["</s>", "\n\n"],
#     )
#     meta_raw = meta_resp["choices"][0]["text"].strip()
#
#     # 解析元数据
#     subject, ques_type, ques_difficulty, knowledges = "", "", "一般", []
#     for line in meta_raw.split('\n'):
#         line = line.strip()
#         if line.startswith('学科：'):
#             subject = line[3:].strip()
#         elif line.startswith('题型：'):
#             ques_type = line[3:].strip()
#         elif line.startswith('难度：'):
#             ques_difficulty = line[3:].strip()
#         elif line.startswith('知识点：'):
#             kns = line[4:].strip()
#             knowledges = [k.strip() for k in kns.replace('、', ',').split(',') if k.strip()]
#
#     print(f"[DEBUG] subject={subject}, knowledges={knowledges}")
#
#     return {
#         "llm_answer": answer_text,
#         "thinking": thinking,
#         "subject": subject,
#         "ques_type": ques_type,
#         "ques_difficulty": ques_difficulty,
#         "knowledges": knowledges,
#     }


def generate_practice_questions(knowledge: str, n: int = 3) -> list:
    """
    根据知识点，让 LLM 生成 n 道练习题
    返回格式与题库一致的列表
    """
    _load_llm()

    resp = _llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "/no_think 你是专业出题教师，根据知识点出题，格式严格按要求输出，不要多余内容。"},
            {"role": "user", "content": f"""请针对知识点「{knowledge}」出{n}道练习题。

每道题严格按如下格式输出，题目之间用"---"分隔：
题目：<题目内容>
答案：<答案>
解析：<解析>
难度：<简单/一般/较难>

---"""}
        ],
        max_tokens=1200,
        temperature=0.7,
    )

    raw = resp["choices"][0]["message"]["content"].strip()

    import re
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # 解析每道题
    questions = []
    blocks = re.split(r'\n---+\n?', raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q = {}
        for field, key in [('题目：', 'ques_content'), ('答案：', 'ques_answer'),
                           ('解析：', 'ques_analyze'), ('难度：', 'ques_difficulty')]:
            m = re.search(field + r'(.+?)(?=\n(?:题目|答案|解析|难度)：|$)', block, re.DOTALL)
            if m:
                q[key] = m.group(1).strip()
        if q.get('ques_content'):
            q['ques_knowledges'] = [knowledge]
            q['subject'] = ''
            q['ques_type'] = 'AI生成'
            q['_ai_generated'] = True
            questions.append(q)

    return questions[:n]


def generate_wrong_answer_report(
    wrong_questions: List[Dict[str, Any]],
    user_id: int
) -> str:
    """
    根据用户的错题记录生成个性化错题分析报告

    Args:
        wrong_questions: 用户错题列表
        user_id: 用户ID

    Returns:
        错题分析报告文本
    """
    _load_llm()

    if not wrong_questions:
        return "暂无错题记录。"

    # 汇总错题信息
    wrong_summary = []
    knowledge_counter: Dict[str, int] = {}

    for q in wrong_questions:
        content = q.get("ques_content", "")
        knowledges = q.get("ques_knowledges", [])
        subject = q.get("subject", "")
        wrong_summary.append(f"- {subject}：{content[:60]}...")
        for k in knowledges:
            knowledge_counter[k] = knowledge_counter.get(k, 0) + 1

    # 按频率排序薄弱知识点
    weak_points = sorted(knowledge_counter.items(), key=lambda x: x[1], reverse=True)
    weak_str = "、".join([f"{k}（{v}次）" for k, v in weak_points[:5]])

    wrong_str = "\n".join(wrong_summary[:10])

    prompt = f"""请根据以下错题记录，生成一份个性化的错题分析报告。

错题列表：
{wrong_str}

频繁出错的知识点：{weak_str}

请输出：
1. 错题整体分析（100字以内）
2. 主要薄弱知识点及建议
3. 针对性学习建议（3条）
"""

    messages = [
        {"role": "system", "content": "你是专业的学习分析师，用中文输出清晰简洁的学习报告。"},
        {"role": "user", "content": prompt}
    ]

    response = _llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.5,
    )

    # return response["choices"][0]["message"]["content"].strip()
    return _strip_thinking(response["choices"][0]["message"]["content"])

def generate_visualization_html(question_text: str, answer_text: str) -> str:
    """
    程序题可视化：使用预置 HTML 模板，不调用 LLM。
    根据题目类型选择对应模板填充数据。
    """
    q = question_text.lower()

    # 01背包 / 背包问题
    if '背包' in q:
        return _viz_knapsack()

    # 排序类
    if any(k in q for k in ['排序', '冒泡', '快速排序', '归并', '堆排序', '插入排序']):
        return _viz_sorting(question_text)

    # 二分查找
    if any(k in q for k in ['二分', '查找', '搜索']):
        return _viz_binary_search()

    # 斐波那契 / 递归
    if any(k in q for k in ['斐波那契', 'fibonacci', '递归']):
        return _viz_fibonacci()

    # 通用：显示解题步骤
    return _viz_general(question_text, answer_text)


def _viz_knapsack() -> str:
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<title>01背包可视化</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#f0f4f8;padding:20px;margin:0}
h2{color:#374151;font-size:1.1rem;margin-bottom:16px}
.controls{background:#fff;border-radius:12px;padding:16px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08)}
.controls label{font-size:0.85rem;color:#6b7280;margin-right:8px}
.controls input{border:1.5px solid #e5e7eb;border-radius:6px;padding:4px 8px;width:60px;font-size:0.85rem}
.btn{padding:8px 20px;border:none;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:0.85rem;font-weight:600;cursor:pointer;margin-top:10px}
.btn:hover{opacity:0.9}
table{border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)}
th{background:#667eea;color:#fff;padding:8px 12px;font-size:0.8rem}
td{border:1px solid #e5e7eb;padding:8px 12px;text-align:center;font-size:0.82rem;transition:background 0.3s}
td.highlight{background:#fef9c3}
td.updated{background:#d1fae5;font-weight:700}
.items-table{margin-bottom:16px}
.result{background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;padding:12px 16px;margin-top:14px;font-size:0.9rem;color:#166534;font-weight:600}
.step-info{font-size:0.82rem;color:#6b7280;margin:10px 0}
</style>
</head>
<body>
<h2>🎒 01背包问题 动态规划可视化</h2>
<div class="controls">
  <div style="margin-bottom:10px">
    <label>背包容量 C：</label><input id="cap" type="number" value="10" min="1" max="20">
  </div>
  <div id="items-input">
    <div style="font-size:0.85rem;color:#374151;margin-bottom:6px">物品（重量,价值）：</div>
    <div><input type="number" class="w" value="2" min="1"> , <input type="number" class="v" value="6" min="0"> &nbsp;
         <input type="number" class="w" value="3" min="1"> , <input type="number" class="v" value="10" min="0"> &nbsp;
         <input type="number" class="w" value="5" min="1"> , <input type="number" class="v" value="12" min="0"></div>
  </div>
  <button class="btn" onclick="runViz()">▶ 开始可视化</button>
</div>
<div id="output"></div>

<script>
async function runViz() {
  var C = parseInt(document.getElementById('cap').value);
  var ws = Array.from(document.querySelectorAll('.w')).map(x=>parseInt(x.value));
  var vs = Array.from(document.querySelectorAll('.v')).map(x=>parseInt(x.value));
  var n = ws.length;

  // 构建 dp 表
  var dp = Array.from({length:n+1},()=>new Array(C+1).fill(0));

  var out = document.getElementById('output');
  out.innerHTML = '';

  // 物品信息表
  var itbl = '<table class="items-table"><tr><th>物品</th><th>重量</th><th>价值</th></tr>';
  for(var i=0;i<n;i++) itbl += '<tr><td>'+(i+1)+'</td><td>'+ws[i]+'</td><td>'+vs[i]+'</td></tr>';
  itbl += '</table>';
  out.innerHTML = itbl;

  // DP 表格（逐步填充）
  var steps = [];
  for(var i=1;i<=n;i++){
    for(var j=0;j<=C;j++){
      dp[i][j] = dp[i-1][j];
      if(ws[i-1]<=j) dp[i][j] = Math.max(dp[i][j], dp[i-1][j-ws[i-1]]+vs[i-1]);
      steps.push({i:i,j:j,val:dp[i][j],updated:ws[i-1]<=j&&dp[i-1][j-ws[i-1]]+vs[i-1]>dp[i-1][j]});
    }
  }

  // 渲染初始空表
  function renderTable(highlight_i, highlight_j) {
    var h = '<table><tr><th>i \\ j</th>';
    for(var j=0;j<=C;j++) h+='<th>'+j+'</th>';
    h+='</tr>';
    for(var i=0;i<=n;i++){
      h+='<tr><td><b>'+(i===0?'0':'物品'+i)+'</b></td>';
      for(var j=0;j<=C;j++){
        var cls = (i===highlight_i&&j===highlight_j)?'updated':'';
        h+='<td class="'+cls+'">'+dp[i][j]+'</td>';
      }
      h+='</tr>';
    }
    h+='</table>';
    return h;
  }

  var tableDiv = document.createElement('div');
  out.appendChild(tableDiv);
  var stepDiv = document.createElement('div');
  stepDiv.className='step-info';
  out.appendChild(stepDiv);

  // 逐步动画
  for(var s=0;s<steps.length;s++){
    var st=steps[s];
    tableDiv.innerHTML = renderTable(st.i, st.j);
    stepDiv.textContent = '处理物品'+st.i+'（重量'+ws[st.i-1]+'，价值'+vs[st.i-1]+'），容量'+st.j+'：dp['+st.i+']['+st.j+'] = '+st.val;
    await new Promise(r=>setTimeout(r,80));
  }

  out.innerHTML += '<div class="result">✅ 最大价值：' + dp[n][C] + '</div>';
}
runViz();
</script>
</body></html>'''


def _viz_sorting(question_text: str) -> str:
    algo = '冒泡排序'
    if '快速' in question_text: algo = '快速排序'
    elif '归并' in question_text: algo = '归并排序'
    elif '插入' in question_text: algo = '插入排序'
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>排序可视化</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#f0f4f8;padding:20px;margin:0}
h2{color:#374151;font-size:1.1rem}
.bar-wrap{display:flex;align-items:flex-end;gap:4px;height:200px;margin:20px 0;background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08)}
.bar{background:linear-gradient(180deg,#667eea,#764ba2);border-radius:4px 4px 0 0;transition:height 0.2s,background 0.2s;display:flex;align-items:flex-start;justify-content:center;padding-top:4px;color:#fff;font-size:0.72rem;font-weight:700;min-width:32px}
.bar.comparing{background:linear-gradient(180deg,#f59e0b,#d97706)}
.bar.sorted{background:linear-gradient(180deg,#10b981,#059669)}
.btn{padding:8px 20px;border:none;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:0.85rem;font-weight:600;cursor:pointer;margin-right:8px}
.info{font-size:0.82rem;color:#6b7280;margin-top:10px}
input{border:1.5px solid #e5e7eb;border-radius:6px;padding:6px 10px;font-size:0.85rem;width:240px}
</style></head>
<body>
<h2>''' + algo + ''' 可视化</h2>
<div><input id="arr-input" value="64,34,25,12,22,11,90" placeholder="输入数组，逗号分隔">
<button class="btn" onclick="startSort()">▶ 开始排序</button></div>
<div class="bar-wrap" id="bars"></div>
<div class="info" id="info">点击开始排序</div>
<script>
var arr=[], sorting=false;
function renderBars(a, ci=-1, cj=-1, sorted_idx=new Set()){
  var mx=Math.max(...a);
  var html='';
  for(var i=0;i<a.length;i++){
    var cls='bar'+(ci===i||cj===i?' comparing':'')+(sorted_idx.has(i)?' sorted':'');
    var h=Math.round(a[i]/mx*160)+20;
    html+='<div class="'+cls+'" style="height:'+h+'px">'+a[i]+'</div>';
  }
  document.getElementById('bars').innerHTML=html;
}
async function startSort(){
  var raw=document.getElementById('arr-input').value;
  arr=raw.split(',').map(x=>parseInt(x.trim())).filter(x=>!isNaN(x));
  if(!arr.length)return;
  var a=[...arr], n=a.length, sorted=new Set();
  document.getElementById('info').textContent='排序中...';
  for(var i=0;i<n-1;i++){
    for(var j=0;j<n-i-1;j++){
      renderBars(a,j,j+1,sorted);
      document.getElementById('info').textContent='比较 a['+j+']='+a[j]+' 和 a['+(j+1)+']='+a[j+1];
      await new Promise(r=>setTimeout(r,300));
      if(a[j]>a[j+1]){var t=a[j];a[j]=a[j+1];a[j+1]=t;}
    }
    sorted.add(n-1-i);
  }
  sorted.add(0);
  renderBars(a,-1,-1,sorted);
  document.getElementById('info').textContent='✅ 排序完成：['+a.join(', ')+']';
}
renderBars([64,34,25,12,22,11,90]);
</script></body></html>'''


def _viz_fibonacci() -> str:
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>斐波那契可视化</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#f0f4f8;padding:20px;margin:0}
h2{color:#374151;font-size:1.1rem}
.fib-row{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
.fib-cell{background:#fff;border:2px solid #e5e7eb;border-radius:8px;padding:10px 14px;font-size:0.85rem;text-align:center;transition:all 0.3s;min-width:50px}
.fib-cell.active{background:#667eea;color:#fff;border-color:#667eea;transform:scale(1.1)}
.fib-cell.done{background:#d1fae5;border-color:#86efac;color:#166534}
.btn{padding:8px 20px;border:none;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:0.85rem;font-weight:600;cursor:pointer}
input{border:1.5px solid #e5e7eb;border-radius:6px;padding:6px 10px;font-size:0.85rem;width:80px}
.info{font-size:0.82rem;color:#6b7280;margin-top:8px}
</style></head>
<body>
<h2>🌀 斐波那契数列 动态规划可视化</h2>
<div><label>计算 F(n)，n=</label><input id="n-input" type="number" value="10" min="1" max="20">
<button class="btn" onclick="runFib()">▶ 开始</button></div>
<div class="fib-row" id="fib-row"></div>
<div class="info" id="info"></div>
<script>
async function runFib(){
  var n=parseInt(document.getElementById('n-input').value);
  var dp=new Array(n+1).fill(0);
  dp[0]=0; if(n>=1)dp[1]=1;
  var row=document.getElementById('fib-row');
  row.innerHTML='';
  var cells=[];
  for(var i=0;i<=n;i++){
    var d=document.createElement('div');
    d.className='fib-cell'; d.innerHTML='F('+i+')<br><b>?</b>';
    row.appendChild(d); cells.push(d);
  }
  cells[0].innerHTML='F(0)<br><b>0</b>'; cells[0].className='fib-cell done';
  if(n>=1){cells[1].innerHTML='F(1)<br><b>1</b>'; cells[1].className='fib-cell done';}
  await new Promise(r=>setTimeout(r,400));
  for(var i=2;i<=n;i++){
    cells[i].className='fib-cell active';
    document.getElementById('info').textContent='F('+i+') = F('+(i-1)+') + F('+(i-2)+') = '+dp[i-1]+' + '+dp[i-2];
    dp[i]=dp[i-1]+dp[i-2];
    await new Promise(r=>setTimeout(r,500));
    cells[i].innerHTML='F('+i+')<br><b>'+dp[i]+'</b>';
    cells[i].className='fib-cell done';
  }
  document.getElementById('info').textContent='✅ F('+n+') = '+dp[n];
}
runFib();
</script></body></html>'''


def _viz_binary_search() -> str:
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>二分查找可视化</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#f0f4f8;padding:20px;margin:0}
h2{color:#374151;font-size:1.1rem}
.arr-row{display:flex;gap:4px;margin:16px 0;flex-wrap:wrap}
.cell{background:#fff;border:2px solid #e5e7eb;border-radius:8px;padding:10px 14px;font-size:0.85rem;text-align:center;min-width:40px;transition:all 0.3s;position:relative}
.cell.range{border-color:#a78bfa;background:#f3e8ff}
.cell.mid{background:#667eea;color:#fff;border-color:#667eea;transform:scale(1.15)}
.cell.found{background:#10b981;color:#fff;border-color:#10b981}
.cell.eliminated{opacity:0.3}
.btn{padding:8px 20px;border:none;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:0.85rem;font-weight:600;cursor:pointer;margin-right:6px}
input{border:1.5px solid #e5e7eb;border-radius:6px;padding:6px 10px;font-size:0.85rem;width:200px;margin-right:6px}
.info{font-size:0.85rem;color:#374151;margin-top:10px;padding:10px;background:#fff;border-radius:8px}
</style></head>
<body>
<h2>🔍 二分查找可视化</h2>
<div>
  <input id="arr-in" value="1,3,5,7,9,11,13,15,17,19" placeholder="有序数组">
  <input id="target-in" type="number" value="13" style="width:80px" placeholder="目标值">
  <button class="btn" onclick="runBS()">▶ 开始查找</button>
</div>
<div class="arr-row" id="arr-row"></div>
<div class="info" id="info">点击开始查找</div>
<script>
async function runBS(){
  var arr=document.getElementById('arr-in').value.split(',').map(x=>parseInt(x.trim()));
  var target=parseInt(document.getElementById('target-in').value);
  var row=document.getElementById('arr-row');
  function render(lo,hi,mid,found){
    row.innerHTML='';
    for(var i=0;i<arr.length;i++){
      var cls='cell';
      if(found===i)cls+=' found';
      else if(i===mid)cls+=' mid';
      else if(i>=lo&&i<=hi)cls+=' range';
      else cls+=' eliminated';
      row.innerHTML+='<div class="'+cls+'"><div style="font-size:0.68rem;color:#9ca3af">'+i+'</div>'+arr[i]+'</div>';
    }
  }
  var lo=0,hi=arr.length-1;
  render(lo,hi,-1,-1);
  await new Promise(r=>setTimeout(r,400));
  while(lo<=hi){
    var mid=Math.floor((lo+hi)/2);
    render(lo,hi,mid,-1);
    document.getElementById('info').textContent='范围 ['+lo+','+hi+']，mid='+mid+'，arr[mid]='+arr[mid]+'，目标='+target;
    await new Promise(r=>setTimeout(r,700));
    if(arr[mid]===target){render(lo,hi,-1,mid);document.getElementById('info').textContent='✅ 找到目标 '+target+'，索引='+mid;return;}
    else if(arr[mid]<target){lo=mid+1;document.getElementById('info').textContent+='\n→ 目标更大，搜索右半部分';}
    else{hi=mid-1;document.getElementById('info').textContent+='\n→ 目标更小，搜索左半部分';}
    await new Promise(r=>setTimeout(r,500));
  }
  render(0,-1,-1,-1);
  document.getElementById('info').textContent='❌ 未找到目标值 '+target;
}
runBS();
</script></body></html>'''


def _viz_general(question_text: str, answer_text: str) -> str:
    """通用：把解题步骤格式化展示"""
    import html as html_module
    q_esc = html_module.escape(question_text[:200])
    # 把解答按行分割成步骤
    lines = [l.strip() for l in answer_text.split('\n') if l.strip()][:15]
    steps_html = ''
    for i, line in enumerate(lines):
        steps_html += f'<div class="step" id="s{i}" style="display:none"><span class="num">{i+1}</span>{html_module.escape(line)}</div>\n'

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>解题步骤可视化</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;background:#f0f4f8;padding:20px;margin:0}}
h2{{color:#374151;font-size:1rem;margin-bottom:12px}}
.question{{background:#eff6ff;border-left:4px solid #667eea;border-radius:4px;padding:12px;font-size:0.88rem;color:#1e40af;margin-bottom:16px}}
.step{{background:#fff;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.88rem;color:#374151;border:1.5px solid #e5e7eb;animation:fadeIn 0.4s ease}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:none}}}}
.num{{display:inline-block;background:#667eea;color:#fff;border-radius:6px;padding:1px 8px;font-size:0.75rem;font-weight:700;margin-right:8px}}
.btn{{padding:8px 20px;border:none;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:0.85rem;font-weight:600;cursor:pointer}}
</style></head>
<body>
<h2>📋 解题步骤展示</h2>
<div class="question">{q_esc}</div>
<div id="steps">{steps_html}</div>
<button class="btn" id="btn" onclick="nextStep()">▶ 下一步</button>
<script>
var cur=0, total={len(lines)};
function nextStep(){{
  if(cur<total){{document.getElementById('s'+cur).style.display='flex';cur++;}}
  if(cur>=total)document.getElementById('btn').textContent='✅ 完成';
}}
nextStep();
</script></body></html>'''

# def generate_visualization_html(
#     question_text: str,
#     answer_text: str
# ) -> str:
#     """
#     针对程序类题目，生成可视化 HTML 代码
#
#     Args:
#         question_text: 题目
#         answer_text: 解答
#
#     Returns:
#         可嵌入 iframe 的 HTML 字符串
#     """
#     _load_llm()
#
#     prompt = f"""请将以下程序题的解析转化为一个独立可运行的HTML页面，使用HTML+CSS+JavaScript实现可视化展示（如动态规划表格填充动画、流程图等）。只输出完整HTML代码，从<!DOCTYPE html>开始，不要任何说明。
#
#     题目：{question_text}
#
#     解答：{answer_text[:300]}"""
#
#     response = _llm(
#         prompt=prompt,
#         max_tokens=1500,
#         temperature=0.2,
#         stop=["</s>"],
#     )
#
#     raw = response["choices"][0]["text"].strip()
#
#     # _strip_thinking 现在返回 (text, thinking) 元组，只取第一个
#     if isinstance(raw, tuple):
#         raw = raw[0]
#
#     # 去掉思考过程
#     think_match = re.search(r'<think>.*?</think>', raw, re.DOTALL)
#     if think_match:
#         raw = raw[think_match.end():].strip()
#
#     # 跳过 JSON 前的自然语言（思考段落）
#     html_start = raw.find('<!DOCTYPE')
#     if html_start == -1:
#         html_start = raw.find('<html')
#     if html_start > 0:
#         raw = raw[html_start:]
#
#     # 提取 ```html 代码块
#     if '```html' in raw:
#         raw = raw.split('```html')[1].split('```')[0].strip()
#     elif '```' in raw:
#         raw = raw.split('```')[1].split('```')[0].strip()
#
#     # 验证是否是合法 HTML
#     if len(raw.strip()) < 50 or '<' not in raw:
#         return None
#
#     return raw
