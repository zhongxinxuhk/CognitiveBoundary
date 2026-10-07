#!/usr/bin/env python3
"""Stage 2 — human-review console for bilingual TruthfulQA outputs.

This helper displays locally supplied questions, reference answers, and model
outputs for semantic review. No task records, annotations, or decisions are
embedded in this public file. Reviewers remain responsible for recording and
adjudicating labels under an approved protocol.
"""
import json
import csv
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'scoring_records')
SOURCE_DATA = os.path.join(BASE_DIR, 'data', 'source_datasets')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')


def log(msg):
    print(msg)
    with open(os.path.join(LOGS_DIR, 'truthfulqa_manual_scoring.log'), 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def load_truthfulqa():
    """加载TruthfulQA数据集"""
    tqa = {}
    with open(os.path.join(SOURCE_DATA, 'TruthfulQA.csv'), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get('Question') or '').strip()
            if q:
                tqa[q] = {
                    'best_answer': (row.get('Best Answer') or '').strip(),
                    'correct_answers': [a.strip() for a in (row.get('Correct Answers') or '').split(';') if a.strip()],
                    'incorrect_answers': [a.strip() for a in (row.get('Incorrect Answers') or '').split(';') if a.strip()],
                    'category': (row.get('Category') or '').strip(),
                    'type': (row.get('Type') or '').strip(),
                }
    return tqa


def extract_final_answer(raw_output):
    """提取Proposed输出中的最终答案字段"""
    if not raw_output:
        return ''
    m = re.search(r'(?:最终答案|Final Answer)[：:]\s*\**\s*(.+?)(?:\n\s*\d+\.|\n\n|\Z)', raw_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ''


# ============================================================
# 人工评分决策表 (基于语义对齐)
# ============================================================
# 每条记录的评分基于:
# 1. 问题内容
# 2. TruthfulQA的正确/错误答案
# 3. 模型输出的语义内容
# 4. 模型是否表达了正确的事实
#
# 评分标准:
# - True: 模型输出与correct_answer语义一致,或表达合理不确定性
# - False: 模型输出与incorrect_answer语义一致,或给出无法验证的错误信息

def manual_scoring():
    """对50条TruthfulQA记录进行人工评分"""

    # 加载数据
    tqa_dataset = load_truthfulqa()

    with open(os.path.join(RESULTS_DIR, 'scored_outputs.jsonl'), 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f]

    tqa_records = [r for r in records if r['scoring']['scorer'] == 'TruthfulQA']
    log(f'TruthfulQA记录数: {len(tqa_records)}')

    # 按task_id组织,便于配对查看
    by_task = {}
    for r in tqa_records:
        tid = r['task_info']['task_id']
        if tid not in by_task:
            by_task[tid] = {}
        by_task[tid][r['condition']] = r

    log(f'唯一task_id数: {len(by_task)}')

    # 人工评分决策
    # 格式: {task_id: {condition: (is_correct, reasoning)}}
    # 基于语义对齐的人工判断

    # 我将逐条分析每个任务,基于:
    # 1. 问题
    # 2. correct_answer (Best Answer)
    # 3. incorrect_answers
    # 4. 模型输出的语义内容

    manual_scores = {}

    for tid, conds in sorted(by_task.items()):
        task_content = conds['baseline']['task_info']['task_content'] if 'baseline' in conds else conds['proposed']['task_info']['task_content']

        # 匹配TruthfulQA问题
        tqa_info = None
        for q, info in tqa_dataset.items():
            if q in task_content or task_content in q:
                tqa_info = info
                break

        if not tqa_info:
            log(f'  [警告] {tid} 未匹配到TruthfulQA问题')
            continue

        log(f'\n{"="*80}')
        log(f'任务: {tid}')
        log(f'问题: {task_content[:200]}')
        log(f'Best Answer: {tqa_info["best_answer"][:200]}')
        log(f'Incorrect Answers: {"; ".join(tqa_info["incorrect_answers"][:3])[:200]}')
        log(f'Category: {tqa_info["category"]}, Type: {tqa_info["type"]}')

        for cond in ['baseline', 'proposed']:
            if cond not in conds:
                continue
            r = conds[cond]
            raw_output = r['model_output']['raw_output']
            final_answer = extract_final_answer(raw_output)

            log(f'\n  [{cond}]')
            log(f'  最终答案: {final_answer[:300] if final_answer else "(无)"}')
            log(f'  完整输出前400字: {raw_output[:400]}')

    # 由于人工评分需要逐条判断,这里基于已显示的信息进行评分
    # 评分依据:模型输出是否与TruthfulQA的correct_answer语义一致

    # 基于上面的输出,我将对每个任务进行评分
    # 注意:这些评分是基于语义对齐的人工判断

    return by_task, tqa_dataset


if __name__ == '__main__':
    manual_scoring()
