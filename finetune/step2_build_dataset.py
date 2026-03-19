"""
Step 2: 构造多任务 Alpaca 格式指令数据集
三种任务：
  Task A - 解题（主任务）：给题目→输出答案+解析
  Task B - 知识点提取：给题目→输出涉及的知识点
  Task C - 错题分析：给题目+错误答案→输出错误原因+正确解析
"""
import json
import random
import os

INPUT_PATH  = '/home/zsy/workspace/250606/code/work/260221_gd/finetune/cleaned_data.json'
OUTPUT_DIR  = '/home/zsy/workspace/250606/code/work/260221_gd/finetune'

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    cleaned = json.load(f)

random.seed(42)
random.shuffle(cleaned)

# ==================== 指令模板 ====================

TASK_A_INSTRUCTIONS = [
    "请解答以下{subject}题目，给出完整的解题过程和答案。",
    "作为一名{subject}教师，请分析并解答下面的题目。",
    "请认真阅读以下{subject}题目，给出答案并详细说明解题步骤。",
    "下面是一道{subject}{type}，请给出正确答案和详细解析。",
    "请解答这道{subject}题目，要求写出解题思路和最终答案。",
]

TASK_B_INSTRUCTIONS = [
    "请分析以下题目涉及的知识点，用逗号分隔列出。",
    "这道{subject}题目考查了哪些知识点？请逐一列出。",
    "请识别下面{subject}题目中涉及的核心知识点。",
    "分析这道题目，指出它覆盖的知识点有哪些。",
]

TASK_C_INSTRUCTIONS = [
    "学生在做这道{subject}题时选了{wrong_ans}，请分析错误原因并给出正确解析。",
    "某同学做以下{subject}题目时给出了错误答案{wrong_ans}，请指出错在哪里并讲解正确思路。",
    "针对以下{subject}题，学生错误地认为答案是{wrong_ans}，请帮助分析原因并给出正确解答。",
]

def get_wrong_answer(correct_answers, ques_type):
    """生成一个合理的错误答案"""
    options = ['A', 'B', 'C', 'D']
    correct_set = set(correct_answers)
    wrong_options = [o for o in options if o not in correct_set]
    if wrong_options:
        return random.choice(wrong_options)
    return '错误答案'

def format_answer(answers):
    if isinstance(answers, list):
        return '、'.join(str(a) for a in answers)
    return str(answers)

# ==================== 构造数据 ====================

all_samples = []
task_counts = {'A': 0, 'B': 0, 'C': 0}

for item in cleaned:
    subject  = item['subject']
    ques_type= item['ques_type']
    content  = item['content']
    answers  = item['answers']
    analyze  = item['analyze']
    kns      = item['knowledges']
    ans_str  = format_answer(answers)

    # Task A: 解题（每道题都做）
    instr_a = random.choice(TASK_A_INSTRUCTIONS).format(
        subject=subject, type=ques_type
    )
    output_a = f"答案：{ans_str}\n\n解析：{analyze}"
    all_samples.append({
        'instruction': instr_a,
        'input': content,
        'output': output_a,
        'task': 'A',
        'subject': subject,
    })
    task_counts['A'] += 1

    # Task B: 知识点提取（有知识点标签的题目）
    if kns and len(kns) >= 1:
        instr_b = random.choice(TASK_B_INSTRUCTIONS).format(subject=subject)
        output_b = '、'.join(kns)
        all_samples.append({
            'instruction': instr_b,
            'input': content,
            'output': output_b,
            'task': 'B',
            'subject': subject,
        })
        task_counts['B'] += 1

    # Task C: 错题分析（选择题/判断题，50%概率采样）
    if ques_type in ['选择题', '单选题', '多选题', '判断题', '填空题'] and random.random() < 0.5:
        wrong_ans = get_wrong_answer(answers, ques_type)
        instr_c = random.choice(TASK_C_INSTRUCTIONS).format(
            subject=subject, wrong_ans=wrong_ans
        )
        output_c = f"错误分析：学生选择了{wrong_ans}，这是不正确的。\n\n正确答案是{ans_str}。\n\n解析：{analyze}"
        all_samples.append({
            'instruction': instr_c,
            'input': content,
            'output': output_c,
            'task': 'C',
            'subject': subject,
        })
        task_counts['C'] += 1

print(f"总样本数: {len(all_samples)}")
print(f"Task A (解题): {task_counts['A']}")
print(f"Task B (知识点提取): {task_counts['B']}")
print(f"Task C (错题分析): {task_counts['C']}")

# ==================== 划分训练/验证/测试集 ====================

random.shuffle(all_samples)
total = len(all_samples)
train_end = int(total * 0.85)
val_end   = int(total * 0.92)

train_data = all_samples[:train_end]
val_data   = all_samples[train_end:val_end]
test_data  = all_samples[val_end:]

print(f"\n=== 数据集划分 ===")
print(f"训练集: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
print(f"验证集: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
print(f"测试集: {len(test_data)} ({len(test_data)/total*100:.1f}%)")

# ==================== 转为 Alpaca 格式保存 ====================

def to_alpaca(samples):
    return [{'instruction': s['instruction'], 'input': s['input'], 'output': s['output']} for s in samples]

for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    path = os.path.join(OUTPUT_DIR, f'{name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_alpaca(data), f, ensure_ascii=False, indent=2)
    print(f"已保存: {path}")

# 保存完整元数据（含 task 标签，用于评估分析）
meta_path = os.path.join(OUTPUT_DIR, 'test_with_meta.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
print(f"已保存测试集元数据: {meta_path}")
