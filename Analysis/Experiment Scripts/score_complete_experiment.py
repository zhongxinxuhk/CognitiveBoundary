#!/usr/bin/env python3
"""
正式实验评分脚本：对 complete_20260726_195433 的 Baseline 和 Proposed 输出进行自动评分。

数据结构（嵌套式）：
  record['condition']           → 'baseline' | 'proposed'
  record['task_info']['task_id']
  record['task_info']['task_type']         → 'knowledge' | 'reasoning' | 'planning'
  record['task_info']['complexity_level']  → 'L1'–'L5'
  record['task_info']['ground_truth_type'] → 'single_answer' | 'multi_point' | 'rubric' | 'N/A'
  record['task_info']['ground_truth_source']
  record['model_output']['raw_output']
  record['performance']['latency_seconds']
"""
import json
import re
import os
import sys
import math
from datetime import datetime
from collections import Counter
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 答案提取 ──────────────────────────────────────────────

def extract_answer_single(raw_output: str, condition: str) -> str:
    """从模型输出中提取 single_answer 类型答案。"""
    if not raw_output:
        return ""

    # Proposed 条件：尝试从结构化格式提取"最终答案"
    if condition == 'proposed':
        # "最终答案：\n   **(B) ...**" 或 "最终答案：**B**"
        m = re.search(r'最终答案[：:]\s*\**\s*\(?([A-F])\)?', raw_output, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # "最终答案：**True**" / "最终答案：**False**"
        m = re.search(r'最终答案[：:]\s*\**\s*(True|False)', raw_output, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    # 通用：提取"最终答案"
    m = re.search(r'最终答案[：:]\s*(.+?)(?:\n|$)', raw_output, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        # 清理 markdown
        clean = re.sub(r'[*_`]', '', ans).strip()
        # 提取括号内字母 (B) → B
        pm = re.search(r'\(?([A-F])\)?', clean)
        if pm:
            return pm.group(1).upper()
        return clean

    # BBH True/False
    m = re.search(r'\b(True|False)\b', raw_output, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    # "答案是 **(D)**" 格式
    m = re.search(r'答案[是：:]\s*\**\s*\(?([A-F])\)?', raw_output, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 选择题 A-F (BBH 有到 F)
    m = re.search(r'\b\(?([A-F])\)?\b', raw_output)
    if m:
        return m.group(1).upper()

    # 返回第一行
    lines = raw_output.strip().split('\n')
    return lines[0].strip() if lines else raw_output.strip()


def extract_ground_truth_single(ground_truth_source: str) -> str:
    """从 ground_truth_source 提取 single_answer 的正确答案。"""
    if not ground_truth_source:
        return ""

    # MMLU: "MMLU answer key: D"
    m = re.search(r'answer key:\s*([A-D])', ground_truth_source, re.IGNORECASE)
    if m:
        return m.group(1)

    # BBH: "BBH target: (B)" — 括号内字母
    m = re.search(r'BBH target:\s*\(?([A-Z])\)?', ground_truth_source, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # BBH: "BBH target: False" / "BBH target: True"
    m = re.search(r'BBH target:\s*(True|False)', ground_truth_source, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    return ground_truth_source.strip()


def check_single_answer(extracted: str, ground_truth: str) -> bool:
    """比较 single_answer 答案是否正确。"""
    if not extracted or not ground_truth:
        return False

    ext = extracted.upper().strip()
    cor = ground_truth.upper().strip()

    if ext == cor:
        return True

    # 提取选项字母 (A-F, BBH 有到 F)
    ext_m = re.search(r'\(?([A-F])\)?', ext)
    if ext_m:
        return ext_m.group(1) == cor

    # True/False
    if cor in ('TRUE', 'FALSE'):
        ext_tf = re.search(r'\b(TRUE|FALSE)\b', ext)
        if ext_tf:
            return ext_tf.group(1) == cor

    return False


# ── 评分主逻辑 ────────────────────────────────────────────

def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """对单条记录评分，返回带评分字段的副本。"""
    ti = record.get('task_info', {})
    mo = record.get('model_output', {})
    condition = record.get('condition', '')
    gt_type = ti.get('ground_truth_type', '')
    gt_source = ti.get('ground_truth_source', '')
    raw_output = mo.get('raw_output', '')

    scored = json.loads(json.dumps(record))  # deep copy
    scored['scoring'] = {
        'scored_at': datetime.now().isoformat(),
        'ground_truth_type': gt_type,
    }

    if gt_type == 'single_answer':
        extracted = extract_answer_single(raw_output, condition)
        truth = extract_ground_truth_single(gt_source)
        is_correct = check_single_answer(extracted, truth)
        scored['scoring'].update({
            'extracted_answer': extracted,
            'ground_truth': truth,
            'is_correct': is_correct,
            'auto_scored': True,
            'needs_human_review': False,
        })
    elif gt_type == 'multi_point':
        # TruthfulQA 开放式，无法自动评分
        scored['scoring'].update({
            'extracted_answer': None,
            'ground_truth': gt_source[:200],
            'is_correct': None,
            'auto_scored': False,
            'needs_human_review': True,
            'notes': 'TruthfulQA 开放式回答，需要语义评分或人工标注',
        })
    elif gt_type == 'rubric':
        # Natural Plan，需要官方评测脚本
        scored['scoring'].update({
            'extracted_answer': None,
            'ground_truth': gt_source[:200],
            'is_correct': None,
            'auto_scored': False,
            'needs_human_review': True,
            'notes': 'Natural Plan 需要官方评测脚本或人工评分',
        })
    else:
        # N/A 或未知
        scored['scoring'].update({
            'extracted_answer': None,
            'ground_truth': None,
            'is_correct': None,
            'auto_scored': False,
            'needs_human_review': True,
            'notes': f'未知或不适用的 ground_truth_type: {gt_type}',
        })

    return scored


# ── 配对分析 ──────────────────────────────────────────────

def paired_analysis(scored_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 Baseline vs Proposed 进行配对分析。"""
    # 按 task_id 分组
    groups = {}
    for r in scored_records:
        tid = r['task_info']['task_id']
        if tid not in groups:
            groups[tid] = {}
        groups[tid][r['condition']] = r

    # 仅保留有 ground_truth 且可自动评分的配对
    paired = []
    for tid, conds in groups.items():
        if 'baseline' in conds and 'proposed' in conds:
            b = conds['baseline']['scoring']
            p = conds['proposed']['scoring']
            if b.get('auto_scored') and p.get('auto_scored'):
                paired.append({
                    'task_id': tid,
                    'task_type': conds['baseline']['task_info']['task_type'],
                    'complexity': conds['baseline']['task_info']['complexity_level'],
                    'baseline_correct': b['is_correct'],
                    'proposed_correct': p['is_correct'],
                })

    n = len(paired)
    if n == 0:
        return {'n_paired': 0, 'note': '无可自动配对评分的任务'}

    both_correct = sum(1 for t in paired if t['baseline_correct'] and t['proposed_correct'])
    both_wrong = sum(1 for t in paired if not t['baseline_correct'] and not t['proposed_correct'])
    b_only = sum(1 for t in paired if t['baseline_correct'] and not t['proposed_correct'])
    p_only = sum(1 for t in paired if not t['baseline_correct'] and t['proposed_correct'])

    b_correct = both_correct + b_only
    p_correct = both_correct + p_only
    b_acc = b_correct / n
    p_acc = p_correct / n

    result = {
        'n_paired': n,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'baseline_only': b_only,
        'proposed_only': p_only,
        'baseline_accuracy': round(b_acc, 4),
        'proposed_accuracy': round(p_acc, 4),
        'accuracy_difference': round(p_acc - b_acc, 4),
    }

    # McNemar 检验
    if b_only + p_only > 0:
        mcnemar_stat = (abs(b_only - p_only) - 1) ** 2 / (b_only + p_only)  # 校正
        mcnemar_stat_uncorrected = (b_only - p_only) ** 2 / (b_only + p_only)
        try:
            from scipy.stats import chi2
            p_value = float(1 - chi2.cdf(mcnemar_stat, df=1))
            p_value_uncorrected = float(1 - chi2.cdf(mcnemar_stat_uncorrected, df=1))
        except ImportError:
            p_value = None
            p_value_uncorrected = None

        # Cohen's h
        h = 2 * math.asin(math.sqrt(p_acc)) - 2 * math.asin(math.sqrt(b_acc))

        # Wilson 95% CI
        def wilson_ci(k, n_total, z=1.96):
            if n_total == 0:
                return (0, 0)
            p_hat = k / n_total
            denom = 1 + z ** 2 / n_total
            center = (p_hat + z ** 2 / (2 * n_total)) / denom
            margin = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n_total)) / n_total) / denom
            return (round(center - margin, 4), round(center + margin, 4))

        result.update({
            'mcnemar_stat_corrected': round(mcnemar_stat, 4),
            'mcnemar_stat_uncorrected': round(mcnemar_stat_uncorrected, 4),
            'p_value_corrected': round(p_value, 6) if p_value else None,
            'p_value_uncorrected': round(p_value_uncorrected, 6) if p_value_uncorrected else None,
            'cohens_h': round(h, 4),
            'baseline_wilson_95ci': wilson_ci(b_correct, n),
            'proposed_wilson_95ci': wilson_ci(p_correct, n),
        })

    # 按任务类型分组
    by_type = {}
    for tt in ['knowledge', 'reasoning', 'planning']:
        subset = [t for t in paired if t['task_type'] == tt]
        if subset:
            b_c = sum(1 for t in subset if t['baseline_correct'])
            p_c = sum(1 for t in subset if t['proposed_correct'])
            by_type[tt] = {
                'n': len(subset),
                'baseline_correct': b_c,
                'proposed_correct': p_c,
                'baseline_accuracy': round(b_c / len(subset), 4),
                'proposed_accuracy': round(p_c / len(subset), 4),
            }
    result['by_task_type'] = by_type

    # 按复杂度分组
    by_cx = {}
    for cx in ['L1', 'L2', 'L3', 'L4', 'L5']:
        subset = [t for t in paired if t['complexity'] == cx]
        if subset:
            b_c = sum(1 for t in subset if t['baseline_correct'])
            p_c = sum(1 for t in subset if t['proposed_correct'])
            by_cx[cx] = {
                'n': len(subset),
                'baseline_correct': b_c,
                'proposed_correct': p_c,
                'baseline_accuracy': round(b_c / len(subset), 4),
                'proposed_accuracy': round(p_c / len(subset), 4),
            }
    result['by_complexity'] = by_cx

    return result


# ── 主函数 ────────────────────────────────────────────────

def main():
    input_file = os.path.join(BASE_DIR, 'outputs', 'raw_outputs_complete_20260726_195433.jsonl')
    output_file = os.path.join(BASE_DIR, 'outputs', 'scored_outputs_complete_20260726_195433.jsonl')
    analysis_file = os.path.join(BASE_DIR, 'outputs', 'paired_analysis_complete_20260726_195433.json')
    summary_file = os.path.join(BASE_DIR, 'outputs', 'scoring_summary_complete_20260726_195433.json')

    print("=" * 80)
    print("正式实验评分: complete_20260726_195433")
    print("=" * 80)

    # 加载记录
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"加载 {len(records)} 条记录")

    # 评分
    scored_records = []
    for r in records:
        scored_records.append(score_record(r))

    # 保存评分文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in scored_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"评分文件已保存: {output_file}")

    # 统计
    auto_scored = sum(1 for r in scored_records if r['scoring'].get('auto_scored'))
    needs_review = sum(1 for r in scored_records if r['scoring'].get('needs_human_review'))
    print(f"\n自动评分: {auto_scored} 条")
    print(f"需要人工评审: {needs_review} 条")

    # 按条件统计
    for cond in ['baseline', 'proposed']:
        subset = [r for r in scored_records if r['condition'] == cond and r['scoring'].get('auto_scored')]
        correct = sum(1 for r in subset if r['scoring']['is_correct'])
        acc = correct / len(subset) if subset else 0
        print(f"  {cond}: {acc:.4f} ({correct}/{len(subset)})")

    # 配对分析
    print("\n" + "=" * 80)
    print("配对分析")
    print("=" * 80)
    analysis = paired_analysis(scored_records)

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"配对分析已保存: {analysis_file}")

    if analysis.get('n_paired', 0) > 0:
        print(f"\n配对任务数: {analysis['n_paired']}")
        print(f"  Baseline 正确率: {analysis['baseline_accuracy']:.4f}")
        print(f"  Proposed 正确率: {analysis['proposed_accuracy']:.4f}")
        print(f"  差异: {analysis['accuracy_difference']:+.4f}")
        if 'p_value_corrected' in analysis and analysis['p_value_corrected'] is not None:
            print(f"  McNemar p (校正): {analysis['p_value_corrected']}")
        if 'cohens_h' in analysis:
            print(f"  Cohen's h: {analysis['cohens_h']}")

    # 汇总文件
    summary = {
        'run_id': 'complete_20260726_195433',
        'scored_at': datetime.now().isoformat(),
        'total_records': len(scored_records),
        'auto_scored': auto_scored,
        'needs_human_review': needs_review,
        'ground_truth_type_distribution': dict(Counter(r['task_info']['ground_truth_type'] for r in scored_records)),
        'actual_complexity_distribution': dict(Counter(r['task_info']['complexity_level'] for r in scored_records)),
        'actual_task_type_distribution': dict(Counter(r['task_info']['task_type'] for r in scored_records)),
        'paired_analysis': analysis,
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总文件已保存: {summary_file}")

    # 打印实际复杂度分布（用于核对表9）
    print("\n" + "=" * 80)
    print("实际复杂度分布（用于核对表9）")
    print("=" * 80)
    for tt in ['knowledge', 'reasoning', 'planning']:
        subset = [r for r in scored_records if r['task_info']['task_type'] == tt]
        cx = Counter(r['task_info']['complexity_level'] for r in subset)
        dist = '，'.join(f'{lv}: {cx.get(lv, 0)}' for lv in ['L1', 'L2', 'L3', 'L4', 'L5'])
        print(f"  {tt} (n={len(subset)}): {dist}，合计 {sum(cx.values())}")

    total_cx = Counter(r['task_info']['complexity_level'] for r in scored_records)
    dist = '，'.join(f'{lv}: {total_cx.get(lv, 0)}' for lv in ['L1', 'L2', 'L3', 'L4', 'L5'])
    print(f"  合计 (n={len(scored_records)}): {dist}，合计 {sum(total_cx.values())}")


if __name__ == '__main__':
    main()
