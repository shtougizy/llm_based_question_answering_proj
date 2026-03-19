"""
Step 4: 微调前基线评估
在测试集上运行原始 Qwen3 模型，记录基线性能
评估指标：答案准确率、知识点召回率、解析质量（ROUGE-L）
"""
import json
import os
import re
import sys
sys.path.insert(0, '/home/zsy/workspace/250606/code/work/260221_gd')

TEST_META_PATH = '/home/zsy/workspace/250606/code/work/260221_gd/finetune/test_with_meta.json'
OUTPUT_DIR     = '/home/zsy/workspace/250606/code/work/260221_gd/finetune/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TEST_META_PATH, 'r') as f:
    test_data = json.load(f)

# 只取各任务各100条做快速评估
task_samples = {'A': [], 'B': [], 'C': []}
for item in test_data:
    t = item.get('task', 'A')
    if len(task_samples[t]) < 100:
        task_samples[t].append(item)

eval_samples = task_samples['A'][:50] + task_samples['B'][:30] + task_samples['C'][:20]
print(f"评估样本数: {len(eval_samples)}")

def rouge_l(pred, ref):
    """计算 ROUGE-L（最长公共子序列）"""
    if not pred or not ref:
        return 0.0
    p_tokens = list(pred)
    r_tokens = list(ref)
    m, n = len(p_tokens), len(r_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if p_tokens[i-1]==r_tokens[j-1] else max(dp[i-1][j],dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / m if m > 0 else 0
    recall    = lcs / n if n > 0 else 0
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)

def extract_answer(text):
    """从输出文本中提取答案"""
    m = re.search(r'答案[：:]\s*([A-D√×是否\w、，,]+)', text)
    if m: return m.group(1).strip()
    return text[:20].strip()

def run_baseline_eval():
    """调用现有 LLM 模块做基线评估"""
    from core.llm import _load_llm, _llm as llm_module
    import core.llm as llm_mod
    llm_mod._load_llm()
    
    results = []
    answer_correct = 0
    rouge_scores = []
    
    for i, sample in enumerate(eval_samples):
        instruction = sample['instruction']
        inp         = sample['input']
        expected    = sample['output']
        task        = sample.get('task', 'A')
        
        prompt = f"{instruction}\n\n{inp}"
        
        try:
            resp = llm_mod._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "/no_think 你是专业教师，简洁回答。"},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1,
            )
            pred = resp["choices"][0]["message"]["content"].strip()
            pred = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
        except Exception as e:
            pred = ""
            print(f"  样本{i}推理失败: {e}")
        
        # 计算指标
        rl = rouge_l(pred, expected)
        rouge_scores.append(rl)
        
        if task == 'A':
            expected_ans = extract_answer(expected)
            pred_ans     = extract_answer(pred)
            if expected_ans and pred_ans and expected_ans[0] == pred_ans[0]:
                answer_correct += 1
        
        results.append({
            'task': task,
            'instruction': instruction[:50],
            'input': inp[:100],
            'expected': expected[:200],
            'predicted': pred[:200],
            'rouge_l': round(rl, 4),
        })
        
        if (i+1) % 10 == 0:
            print(f"  进度: {i+1}/{len(eval_samples)}, 平均ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")
    
    task_a_total = sum(1 for s in eval_samples if s.get('task')=='A')
    metrics = {
        'model': 'Qwen3-1.7B-baseline',
        'total_samples': len(eval_samples),
        'avg_rouge_l': sum(rouge_scores)/len(rouge_scores) if rouge_scores else 0,
        'task_a_answer_accuracy': answer_correct/task_a_total if task_a_total > 0 else 0,
        'task_a_total': task_a_total,
    }
    
    print("\n=== 基线评估结果 ===")
    print(f"平均 ROUGE-L:    {metrics['avg_rouge_l']:.4f}")
    print(f"Task A 答案准确率: {metrics['task_a_answer_accuracy']:.4f} ({answer_correct}/{task_a_total})")
    
    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, 'baseline_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'samples': results[:20]}, f, ensure_ascii=False, indent=2)
    print(f"\n基线结果已保存: {save_path}")
    return metrics

if __name__ == '__main__':
    print("开始基线评估（使用现有 Qwen3 模型）...")
    metrics = run_baseline_eval()
