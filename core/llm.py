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
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1,  # 如果有 GPU 则全量卸载，没有 GPU 则自动回退 CPU
        verbose=False,
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

#
# def answer_question(question_text: str, retrieved: list) -> dict:
#     _load_llm()
#
#     writing_keywords = ['写作', '作文', '写一篇', '写一段', '写出', '写信', '日记', '短文', '写文章', '议论文', '说明文', '记叙文']
#     is_writing = any(kw in question_text for kw in writing_keywords)
#     max_tokens = 2048 if is_writing else 800
#     writing_hint = "（写作题，请写完整文章，不少于600字）" if is_writing else ""
#
#     context = ""
#     if retrieved:
#         context = "\n参考题目：\n"
#         for i, r in enumerate(retrieved[:3]):
#             context += f"{i+1}. {r.get('ques_content', '')}\n"
#             if r.get('ques_answer'):
#                 context += f"   答案：{r['ques_answer']}\n"
#
#     prompt = f"""你是一位全科专业教师。请解答以下题目{writing_hint}，然后只输出JSON结果。
#
# 题目：
# {question_text}
# {context}
# 只输出如下JSON，不要输出其他任何内容：
# {{"answer": "解答内容", "subject": "学科", "ques_type": "题型", "ques_difficulty": "难度", "knowledges": ["知识点1", "知识点2"]}}"""
#
#     response = _llm(
#         prompt=prompt,
#         max_tokens=max_tokens,
#         temperature=0.1,
#         stop=["</s>", "```"],
#     )
#
#     raw = response["choices"][0]["text"].strip()
#
#     import json, re
#
#     # 分离思考过程和正文
#     # 模式1：<think>...</think>
#     thinking = ""
#     think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
#     if think_match:
#         thinking = think_match.group(1).strip()
#         raw = raw[think_match.end():].strip()
#
#     # 模式2：JSON 前面的自然语言思考过程（"好的，我需要..."）
#     json_start = raw.find('{')
#     if json_start > 30:
#         thinking = (thinking + "\n" + raw[:json_start]).strip()
#         raw = raw[json_start:]
#
#     try:
#         match = re.search(r'\{.*\}', raw, re.DOTALL)
#         if match:
#             data = json.loads(match.group())
#             answer = data.get("answer", "").strip()
#             if not answer:
#                 answer = raw
#             return {
#                 "llm_answer": answer,
#                 "thinking": thinking,          # ← 思考过程单独返回
#                 "subject": data.get("subject", ""),
#                 "ques_type": data.get("ques_type", ""),
#                 "ques_difficulty": data.get("ques_difficulty", "一般"),
#                 "knowledges": data.get("knowledges", []),
#             }
#     # except Exception as e:
#     #     print(f"[DEBUG] JSON解析失败: {e}\nraw={raw[:200]}")
#     except Exception as e:
#         print(f"[DEBUG] JSON解析失败: {e}")
#         print(f"[DEBUG] raw输出: {repr(raw[:400])}")
#
#     return {
#         "llm_answer": raw[:2000],
#         "thinking": thinking,
#         "subject": "",
#         "ques_type": "",
#         "ques_difficulty": "一般",
#         "knowledges": [],
#     }

def answer_question(question_text: str, retrieved: list) -> dict:
    _load_llm()

    writing_keywords = ['写作', '作文', '写一篇', '写一段', '写出', '写信', '日记', '短文', '写文章', '议论文', '说明文', '记叙文']
    is_writing = any(kw in question_text for kw in writing_keywords)
    max_tokens = 2048 if is_writing else 800

    context = ""
    if retrieved:
        context = "\n\n参考题目：\n"
        for i, r in enumerate(retrieved[:3]):
            context += f"{i+1}. {r.get('ques_content', '')}\n"
            if r.get('ques_answer'):
                context += f"   答案：{r['ques_answer']}\n"

    # 第一步：让模型自然作答
    prompt = f"""你是一位专业教师，请用中文解答以下题目。{'要求写完整文章，不少于600字。' if is_writing else ''}

题目：{question_text}
{context}
请给出答案和解析："""

    response = _llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        stop=["</s>"],
    )

    raw = response["choices"][0]["text"].strip()

    # 分离思考过程（<think> 标签 或 JSON前的自然语言）
    import re
    thinking = ""
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        raw = raw[think_match.end():].strip()

    # JSON前的自然语言思考（"好的，我需要..."之类）
    # 找到正文起始：跳过开头的思考段落
    lines = raw.split('\n')
    answer_lines = []
    skip_thinking = True
    for line in lines:
        # 思考段落特征：以"好的"、"首先"、"接下来"、"现在"、"我需要"开头
        thinking_starters = ['好的，', '首先，', '接下来，', '现在，', '我需要', '我来', '让我']
        if skip_thinking and any(line.startswith(s) for s in thinking_starters):
            thinking = (thinking + '\n' + line).strip()
            continue
        skip_thinking = False
        answer_lines.append(line)

    answer_text = '\n'.join(answer_lines).strip()
    if not answer_text:
        answer_text = raw  # 全部都是思考，直接用原文

    # 第二步：用第二次 LLM 调用提取结构化元数据（轻量调用）
    meta_prompt = f"""根据以下题目，只输出学科、题型、难度、知识点，格式如下，不要其他内容：
学科：数学
题型：填空题
难度：一般
知识点：知识点1,知识点2,知识点3

题目：{question_text[:200]}"""

    meta_resp = _llm(
        prompt=meta_prompt,
        max_tokens=80,
        temperature=0.1,
        stop=["</s>", "\n\n"],
    )
    meta_raw = meta_resp["choices"][0]["text"].strip()

    # 解析元数据
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


def generate_visualization_html(
    question_text: str,
    answer_text: str
) -> str:
    """
    针对程序类题目，生成可视化 HTML 代码

    Args:
        question_text: 题目
        answer_text: 解答

    Returns:
        可嵌入 iframe 的 HTML 字符串
    """
    _load_llm()

    prompt = f"""请将以下程序题的解析转化为一个独立可运行的 HTML 页面，
使用 HTML + CSS + JavaScript 实现可视化展示（如流程图、动画、交互等）。
只输出完整 HTML 代码，不要任何额外说明。

题目：{question_text}

解答：{answer_text}
"""

    messages = [
        {"role": "system", "content": "你是专业的前端工程师，用 HTML/CSS/JS 实现算法可视化。只输出代码。"},
        {"role": "user", "content": prompt}
    ]

    response = _llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )

    html_content = response["choices"][0]["message"]["content"].strip()
    html_content = _strip_thinking(html_content)

    # 提取 HTML 代码块
    if "```html" in html_content:
        html_content = html_content.split("```html")[1].split("```")[0].strip()
    elif "```" in html_content:
        html_content = html_content.split("```")[1].split("```")[0].strip()

    return html_content
