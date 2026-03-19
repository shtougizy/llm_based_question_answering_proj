"""
Step 5: 微调后评估 & 与基线对比
使用微调后的 HF 模型做推理，计算同样的指标，生成对比报告
"""
import json
import os
import re

FINETUNE_DIR  = '/home/zsy/workspace/250606/code/work/260221_gd/finetune'
OUTPUT_DIR    = os.path.join(FINETUNE_DIR, 'output')
BASELINE_PATH = os.path.join(OUTPUT_DIR, 'baseline_results.json')
TEST_META_PATH= os.path.join(FINETUNE_DIR, 'test_with_meta.json')
MERGED_MODEL  = os.path.join(OUTPUT_DIR, 'qwen3-1.7b-k12-merged')

def rouge_l(pred, ref):
    if not pred or not ref: return 0.0
    p_tokens, r_tokens = list(pred), list(ref)
    m, n = len(p_tokens), len(r_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if p_tokens[i-1]==r_tokens[j-1] else max(dp[i-1][j],dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs/m if m>0 else 0
    rec  = lcs/n if n>0 else 0
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)

def extract_answer(text):
    m = re.search(r'答案[：:]\s*([A-D√×是否\w、，,]+)', text)
    return m.group(1).strip() if m else text[:20].strip()

def run_finetuned_eval():
    print("加载微调后模型...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("模型加载完成")

    with open(TEST_META_PATH, 'r') as f:
        test_data = json.load(f)

    task_samples = {'A': [], 'B': [], 'C': []}
    for item in test_data:
        t = item.get('task', 'A')
        if len(task_samples[t]) < 100:
            task_samples[t].append(item)
    eval_samples = task_samples['A'][:50] + task_samples['B'][:30] + task_samples['C'][:20]

    results = []
    rouge_scores = []
    answer_correct = 0

    for i, sample in enumerate(eval_samples):
        instruction = sample['instruction']
        inp         = sample['input']
        expected    = sample['output']
        task        = sample.get('task', 'A')

        messages = [
            {"role": "system", "content": "你是专业教师，简洁回答。"},
            {"role": "user",   "content": f"{instruction}\n\n{inp}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=False)
        pred = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        rl = rouge_l(pred, expected)
        rouge_scores.append(rl)

        if task == 'A':
            if extract_answer(expected) and extract_answer(pred):
                if extract_answer(expected)[0] == extract_answer(pred)[0]:
                    answer_correct += 1

        results.append({'task': task, 'expected': expected[:200], 'predicted': pred[:200], 'rouge_l': round(rl,4)})

        if (i+1) % 10 == 0:
            print(f"  进度: {i+1}/{len(eval_samples)}, 平均ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")

    task_a_total = sum(1 for s in eval_samples if s.get('task')=='A')
    metrics = {
        'model': 'Qwen3-1.7B-k12-finetuned',
        'avg_rouge_l': sum(rouge_scores)/len(rouge_scores) if rouge_scores else 0,
        'task_a_answer_accuracy': answer_correct/task_a_total if task_a_total>0 else 0,
    }

    # 加载基线结果并对比
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)
    base_metrics = baseline['metrics']

    print("\n" + "="*50)
    print("         微调效果对比报告")
    print("="*50)
    print(f"{'指标':<25} {'微调前':>10} {'微调后':>10} {'提升':>10}")
    print("-"*55)

    rouge_diff = metrics['avg_rouge_l'] - base_metrics['avg_rouge_l']
    acc_diff   = metrics['task_a_answer_accuracy'] - base_metrics['task_a_answer_accuracy']

    print(f"{'平均 ROUGE-L':<25} {base_metrics['avg_rouge_l']:>10.4f} {metrics['avg_rouge_l']:>10.4f} {rouge_diff:>+10.4f}")
    print(f"{'解题答案准确率':<23} {base_metrics['task_a_answer_accuracy']:>10.4f} {metrics['task_a_answer_accuracy']:>10.4f} {acc_diff:>+10.4f}")
    print("="*55)

    report = {
        'baseline': base_metrics,
        'finetuned': metrics,
        'improvement': {'rouge_l': rouge_diff, 'answer_accuracy': acc_diff},
        'samples': results[:20],
    }
    report_path = os.path.join(OUTPUT_DIR, 'comparison_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n对比报告已保存: {report_path}")

if __name__ == '__main__':
    run_finetuned_eval()
