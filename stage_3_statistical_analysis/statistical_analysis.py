#!/usr/bin/env python3
"""Stage 3 — paired statistical-analysis pipeline.

The script contains analysis functions and output serialization only. It does
not include research records, computed results, manuscript text, or figures.
Inputs must be supplied locally under ``data/scoring_records``.
"""
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')


def log(msg):
    print(msg)
    with open(os.path.join(LOGS_DIR, 'statistical_analysis.log'), 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ============================================================
# 统计工具函数
# ============================================================

def wilson_ci(k, n, z=1.96):
    """Wilson 95% 置信区间 (含中心点调整项)"""
    if n == 0:
        return (0.0, 0.0, 0.0)  # (lower, center, upper)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - spread), center, min(1, center + spread))


def cohens_h(p1, p2):
    """Cohen's h 效应量"""
    if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
        return 0.0
    # 边界处理
    p1 = max(1e-10, min(1 - 1e-10, p1))
    p2 = max(1e-10, min(1 - 1e-10, p2))
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def mcnemar_test(b, c, use_continuity_correction=True):
    """
    McNemar 检验
    b: 仅baseline正确数
    c: 仅proposed正确数
    use_continuity_correction: 是否使用连续性校正
    """
    if b + c == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'method': 'McNemar (no discordant pairs)'}
    if use_continuity_correction:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        method = 'McNemar with continuity correction'
    else:
        stat = (b - c) ** 2 / (b + c)
        method = 'McNemar without continuity correction'
    # 卡方分布 p-value
    # 使用 scipy 若可用, 否则用近似
    try:
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(stat, df=1)
    except ImportError:
        # 近似公式: P(X > x) for chi2 with df=1
        # 使用正态近似
        z = math.sqrt(stat)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    # 同时计算精确检验 (二项分布)
    try:
        from scipy.stats import binom
        n = b + c
        k = min(b, c)
        # 双侧 p-value
        exact_p = 2 * sum(binom.pmf(i, n, 0.5) for i in range(k + 1))
        exact_p = min(1.0, exact_p)
    except ImportError:
        exact_p = None

    return {
        'statistic': stat,
        'p_value': p_value,
        'method': method,
        'b_only': b,
        'c_only': c,
        'n_discordant': b + c,
        'exact_p_value': exact_p,
    }


def bootstrap_ci_diff(paired_data, n_bootstrap=10000, seed=20260727, confidence=0.95):
    """Bootstrap 95% CI for paired difference (proposed_acc - baseline_acc)"""
    random.seed(seed)
    if not paired_data:
        return {'ci_low': 0.0, 'ci_high': 0.0, 'point_estimate': 0.0, 'n_bootstrap': n_bootstrap}

    n = len(paired_data)
    diffs = []
    for _ in range(n_bootstrap):
        sample = random.choices(paired_data, k=n)
        b_acc = sum(1 for t in sample if t['baseline_correct']) / n
        p_acc = sum(1 for t in sample if t['proposed_correct']) / n
        diffs.append(p_acc - b_acc)
    diffs.sort()
    alpha = (1 - confidence) / 2
    ci_low = diffs[int(alpha * n_bootstrap)]
    ci_high = diffs[int((1 - alpha) * n_bootstrap)]

    # 原始样本差异
    orig_b_acc = sum(1 for t in paired_data if t['baseline_correct']) / n
    orig_p_acc = sum(1 for t in paired_data if t['proposed_correct']) / n

    return {
        'ci_low': ci_low,
        'ci_high': ci_high,
        'point_estimate': orig_p_acc - orig_b_acc,
        'n_bootstrap': n_bootstrap,
        'seed': seed,
        'confidence_level': confidence,
    }


# ============================================================
# 过程字段分析
# ============================================================

def analyze_process_fields(records):
    """分析模型输出中的过程字段 (推理过程、置信度、不确定性等)"""

    def has_field(text, field_type):
        """检查输出是否包含特定字段"""
        if not text:
            return False
        if field_type == 'reasoning':
            # 推理过程
            return bool(
                '推理过程' in text or '推理步骤' in text or '逐步推理' in text or
                'Reasoning' in text or 'Step 1' in text or '步骤1' in text or
                '分析' in text
            )
        elif field_type == 'confidence':
            # 置信度
            return bool(
                '置信度' in text or 'Confidence' in text or '确定程度' in text or
                '高/中/低' in text
            )
        elif field_type == 'uncertainty':
            # 不确定性
            return bool(
                '不确定性' in text or 'Uncertainty' in text or '风险' in text or
                '假设' in text or '需要' in text or '建议' in text
            )
        elif field_type == 'final_answer':
            # 最终答案
            return bool(
                '最终答案' in text or 'Final Answer' in text or '答案' in text or
                'SOLUTION' in text or 'Answer' in text
            )
        return False

    results = {}
    for condition in ['baseline', 'proposed']:
        cond_records = [r for r in records if r['condition'] == condition and r['model_output']['success']]
        field_stats = {}
        for field in ['reasoning', 'confidence', 'uncertainty', 'final_answer']:
            count = sum(1 for r in cond_records if has_field(r['model_output']['raw_output'], field))
            field_stats[field] = {
                'count': count,
                'total': len(cond_records),
                'rate': count / len(cond_records) if cond_records else 0.0,
            }
        results[condition] = field_stats
    return results


def analyze_latency(records):
    """延迟分析"""
    results = {}
    for condition in ['baseline', 'proposed']:
        latencies = [r['performance']['latency_seconds'] for r in records
                     if r['condition'] == condition and r['model_output']['success']]
        if latencies:
            lat_sorted = sorted(latencies)
            n = len(lat_sorted)
            results[condition] = {
                'count': n,
                'mean': sum(latencies) / n,
                'median': lat_sorted[n // 2],
                'min': min(latencies),
                'max': max(latencies),
                'p25': lat_sorted[n // 4],
                'p75': lat_sorted[3 * n // 4],
                'p95': lat_sorted[int(0.95 * n)] if n > 1 else lat_sorted[0],
            }
    return results


# ============================================================
# 主分析流程
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log('=' * 80)
    log('CognitiveBoundary 正式实验统计分析')
    log('=' * 80)

    # 加载评分结果
    scored_file = os.path.join(BASE_DIR, 'data', 'scoring_records', 'scored_outputs.jsonl')
    with open(scored_file, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f]
    log(f'\n加载评分结果: {len(records)} 条记录')

    # 实验基本信息
    experiment_info = {
        'experiment_date': '2026-07-26',
        'analysis_date': datetime.now().isoformat(),
        'model_id': 'claude-opus-4-6',
        'model_provider': 'anthropic',
        'api_endpoint': 'https://lunagate.xyz',
        'temperature': 0.0,
        'max_output_tokens': 4096,
        'total_records': len(records),
        'conditions': dict(Counter(r['condition'] for r in records)),
        'task_types': dict(Counter(r['task_info']['task_type'] for r in records)),
        'complexity_levels': dict(Counter(r['task_info']['complexity_level'] for r in records)),
        'api_success_rate': sum(1 for r in records if r['model_output']['success']) / len(records),
        'total_paired_tasks': len(set(r['task_info']['task_id'] for r in records)),
    }
    log(f'\n[实验信息]')
    for k, v in experiment_info.items():
        log(f'  {k}: {v}')

    # ============================================================
    # 1. 总体正确率分析
    # ============================================================
    log('\n' + '=' * 60)
    log('[1] 总体正确率分析')
    log('=' * 60)

    overall_stats = {}
    for condition in ['baseline', 'proposed']:
        cond_records = [r for r in records if r['condition'] == condition and r['scoring']['is_correct'] is not None]
        correct = sum(1 for r in cond_records if r['scoring']['is_correct'])
        total = len(cond_records)
        ci_low, ci_center, ci_high = wilson_ci(correct, total)
        overall_stats[condition] = {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0.0,
            'wilson_ci_low': ci_low,
            'wilson_ci_center': ci_center,
            'wilson_ci_high': ci_high,
        }
        log(f'  {condition}: {correct}/{total} = {correct/total:.4f} (Wilson 95% CI: [{ci_low:.4f}, {ci_high:.4f}])')

    # ============================================================
    # 2. 配对分析 (核心统计)
    # ============================================================
    log('\n' + '=' * 60)
    log('[2] 配对分析')
    log('=' * 60)

    # 按task_id组织记录
    task_groups = defaultdict(dict)
    for r in records:
        if r['scoring']['is_correct'] is None:
            continue
        task_id = r['task_info']['task_id']
        task_groups[task_id][r['condition']] = r

    # 找出完整配对 (同时有 baseline 和 proposed)
    paired = []
    for task_id, conds in task_groups.items():
        if 'baseline' in conds and 'proposed' in conds:
            paired.append({
                'task_id': task_id,
                'task_type': conds['baseline']['task_info']['task_type'],
                'complexity_level': conds['baseline']['task_info']['complexity_level'],
                'domain': conds['baseline']['task_info']['domain'],
                'scorer': conds['baseline']['scoring']['scorer'],
                'baseline_correct': conds['baseline']['scoring']['is_correct'],
                'proposed_correct': conds['proposed']['scoring']['is_correct'],
                'baseline_latency': conds['baseline']['performance']['latency_seconds'],
                'proposed_latency': conds['proposed']['performance']['latency_seconds'],
            })

    n_paired = len(paired)
    both_correct = sum(1 for t in paired if t['baseline_correct'] and t['proposed_correct'])
    both_wrong = sum(1 for t in paired if not t['baseline_correct'] and not t['proposed_correct'])
    baseline_only = sum(1 for t in paired if t['baseline_correct'] and not t['proposed_correct'])
    proposed_only = sum(1 for t in paired if not t['baseline_correct'] and t['proposed_correct'])

    log(f'\n  配对任务数: {n_paired}')
    log(f'  两者都对: {both_correct}')
    log(f'  两者都错: {both_wrong}')
    log(f'  仅Baseline对: {baseline_only}')
    log(f'  仅Proposed对: {proposed_only}')

    # 计算正确率
    p_baseline = (both_correct + baseline_only) / n_paired if n_paired > 0 else 0
    p_proposed = (both_correct + proposed_only) / n_paired if n_paired > 0 else 0
    diff = p_proposed - p_baseline

    log(f'\n  Baseline 正确率: {p_baseline:.4f} ({both_correct + baseline_only}/{n_paired})')
    log(f'  Proposed 正确率: {p_proposed:.4f} ({both_correct + proposed_only}/{n_paired})')
    log(f'  差异: {diff:+.4f}')

    # Wilson CI for each
    ci_b_low, ci_b_center, ci_b_high = wilson_ci(both_correct + baseline_only, n_paired)
    ci_p_low, ci_p_center, ci_p_high = wilson_ci(both_correct + proposed_only, n_paired)
    log(f'  Baseline Wilson CI: [{ci_b_low:.4f}, {ci_b_high:.4f}]')
    log(f'  Proposed Wilson CI: [{ci_p_low:.4f}, {ci_p_high:.4f}]')

    # McNemar 检验 (含连续性校正和精确检验)
    mcnemar_cc = mcnemar_test(baseline_only, proposed_only, use_continuity_correction=True)
    mcnemar_no_cc = mcnemar_test(baseline_only, proposed_only, use_continuity_correction=False)

    log(f'\n  McNemar 检验 (含连续性校正):')
    log(f'    统计量: {mcnemar_cc["statistic"]:.4f}')
    log(f'    p-value: {mcnemar_cc["p_value"]:.6f}')
    if mcnemar_cc.get('exact_p_value') is not None:
        log(f'    精确 p-value (二项分布): {mcnemar_cc["exact_p_value"]:.6f}')

    log(f'\n  McNemar 检验 (无连续性校正):')
    log(f'    统计量: {mcnemar_no_cc["statistic"]:.4f}')
    log(f'    p-value: {mcnemar_no_cc["p_value"]:.6f}')

    # Cohen's h 效应量
    h = cohens_h(p_baseline, p_proposed)
    # 效应量解释
    h_abs = abs(h)
    if h_abs < 0.2:
        h_interpretation = '微小效应 (negligible)'
    elif h_abs < 0.5:
        h_interpretation = '小效应 (small)'
    elif h_abs < 0.8:
        h_interpretation = '中效应 (medium)'
    else:
        h_interpretation = '大效应 (large)'

    log(f'\n  Cohen\'s h: {h:+.4f} ({h_interpretation})')

    # Bootstrap 95% CI
    bootstrap = bootstrap_ci_diff(paired)
    log(f'\n  Bootstrap 95% CI for difference: [{bootstrap["ci_low"]:+.4f}, {bootstrap["ci_high"]:+.4f}]')
    log(f'  点估计: {bootstrap["point_estimate"]:+.4f}')

    # ============================================================
    # 3. 按任务类型分层分析
    # ============================================================
    log('\n' + '=' * 60)
    log('[3] 按任务类型分层分析')
    log('=' * 60)

    by_task_type = {}
    for tt in ['knowledge', 'reasoning', 'planning']:
        sub = [t for t in paired if t['task_type'] == tt]
        if not sub:
            continue
        n = len(sub)
        b_correct = sum(1 for t in sub if t['baseline_correct'])
        p_correct = sum(1 for t in sub if t['proposed_correct'])
        b_only = sum(1 for t in sub if t['baseline_correct'] and not t['proposed_correct'])
        p_only = sum(1 for t in sub if not t['baseline_correct'] and t['proposed_correct'])
        b_acc = b_correct / n
        p_acc = p_correct / n
        h_tt = cohens_h(b_acc, p_acc)
        ci_b = wilson_ci(b_correct, n)
        ci_p = wilson_ci(p_correct, n)
        mcnemar_tt = mcnemar_test(b_only, p_only)

        by_task_type[tt] = {
            'n': n,
            'baseline_correct': b_correct,
            'proposed_correct': p_correct,
            'baseline_accuracy': b_acc,
            'proposed_accuracy': p_acc,
            'diff': p_acc - b_acc,
            'baseline_only': b_only,
            'proposed_only': p_only,
            'both_correct': sum(1 for t in sub if t['baseline_correct'] and t['proposed_correct']),
            'both_wrong': sum(1 for t in sub if not t['baseline_correct'] and not t['proposed_correct']),
            'cohens_h': h_tt,
            'baseline_wilson_ci': [ci_b[0], ci_b[2]],
            'proposed_wilson_ci': [ci_p[0], ci_p[2]],
            'mcnemar': mcnemar_tt,
        }
        log(f'\n  {tt}: n={n}, Baseline={b_acc:.4f}, Proposed={p_acc:.4f}, diff={p_acc-b_acc:+.4f}, h={h_tt:+.4f}')
        log(f'    Wilson CI: B=[{ci_b[0]:.4f}, {ci_b[2]:.4f}], P=[{ci_p[0]:.4f}, {ci_p[2]:.4f}]')
        log(f'    McNemar p={mcnemar_tt["p_value"]:.4f}')

    # ============================================================
    # 4. 按复杂度等级分层分析
    # ============================================================
    log('\n' + '=' * 60)
    log('[4] 按复杂度等级分层分析')
    log('=' * 60)

    by_complexity = {}
    for lvl in ['L1', 'L2', 'L3', 'L4', 'L5']:
        sub = [t for t in paired if t['complexity_level'] == lvl]
        if not sub:
            continue
        n = len(sub)
        b_correct = sum(1 for t in sub if t['baseline_correct'])
        p_correct = sum(1 for t in sub if t['proposed_correct'])
        b_only = sum(1 for t in sub if t['baseline_correct'] and not t['proposed_correct'])
        p_only = sum(1 for t in sub if not t['baseline_correct'] and t['proposed_correct'])
        b_acc = b_correct / n
        p_acc = p_correct / n
        h_lvl = cohens_h(b_acc, p_acc)
        ci_b = wilson_ci(b_correct, n)
        ci_p = wilson_ci(p_correct, n)
        mcnemar_lvl = mcnemar_test(b_only, p_only)

        by_complexity[lvl] = {
            'n': n,
            'baseline_correct': b_correct,
            'proposed_correct': p_correct,
            'baseline_accuracy': b_acc,
            'proposed_accuracy': p_acc,
            'diff': p_acc - b_acc,
            'baseline_only': b_only,
            'proposed_only': p_only,
            'cohens_h': h_lvl,
            'baseline_wilson_ci': [ci_b[0], ci_b[2]],
            'proposed_wilson_ci': [ci_p[0], ci_p[2]],
            'mcnemar': mcnemar_lvl,
        }
        log(f'  {lvl}: n={n}, Baseline={b_acc:.4f}, Proposed={p_acc:.4f}, diff={p_acc-b_acc:+.4f}, h={h_lvl:+.4f}, McNemar p={mcnemar_lvl["p_value"]:.4f}')

    # ============================================================
    # 5. 按评分器分层分析 (MMLU/TruthfulQA/BBH/...)
    # ============================================================
    log('\n' + '=' * 60)
    log('[5] 按评分器(数据集)分层分析')
    log('=' * 60)

    by_scorer = {}
    for scorer_name in sorted(set(t['scorer'] for t in paired)):
        sub = [t for t in paired if t['scorer'] == scorer_name]
        if not sub:
            continue
        n = len(sub)
        b_correct = sum(1 for t in sub if t['baseline_correct'])
        p_correct = sum(1 for t in sub if t['proposed_correct'])
        b_only = sum(1 for t in sub if t['baseline_correct'] and not t['proposed_correct'])
        p_only = sum(1 for t in sub if not t['baseline_correct'] and t['proposed_correct'])
        b_acc = b_correct / n
        p_acc = p_correct / n
        h_sc = cohens_h(b_acc, p_acc)
        ci_b = wilson_ci(b_correct, n)
        ci_p = wilson_ci(p_correct, n)
        mcnemar_sc = mcnemar_test(b_only, p_only)

        by_scorer[scorer_name] = {
            'n': n,
            'baseline_correct': b_correct,
            'proposed_correct': p_correct,
            'baseline_accuracy': b_acc,
            'proposed_accuracy': p_acc,
            'diff': p_acc - b_acc,
            'baseline_only': b_only,
            'proposed_only': p_only,
            'cohens_h': h_sc,
            'baseline_wilson_ci': [ci_b[0], ci_b[2]],
            'proposed_wilson_ci': [ci_p[0], ci_p[2]],
            'mcnemar': mcnemar_sc,
        }
        log(f'  {scorer_name}: n={n}, B={b_acc:.4f}, P={p_acc:.4f}, diff={p_acc-b_acc:+.4f}, h={h_sc:+.4f}, p={mcnemar_sc["p_value"]:.4f}')

    # ============================================================
    # 6. 过程字段分析
    # ============================================================
    log('\n' + '=' * 60)
    log('[6] 过程字段可观测性分析')
    log('=' * 60)

    process_fields = analyze_process_fields(records)
    for cond, fields in process_fields.items():
        log(f'\n  {cond}:')
        for fname, stats in fields.items():
            log(f'    {fname}: {stats["count"]}/{stats["total"]} = {stats["rate"]:.4f}')

    # ============================================================
    # 7. 延迟分析
    # ============================================================
    log('\n' + '=' * 60)
    log('[7] 延迟分析')
    log('=' * 60)

    latency_stats = analyze_latency(records)
    for cond, stats in latency_stats.items():
        log(f'\n  {cond}:')
        for k, v in stats.items():
            log(f'    {k}: {v:.4f}' if isinstance(v, float) else f'    {k}: {v}')

    # ============================================================
    # 8. 错误率随复杂度变化趋势 (用于检验 H1/H2)
    # ============================================================
    log('\n' + '=' * 60)
    log('[8] 错误率随复杂度变化趋势')
    log('=' * 60)

    complexity_trend = {}
    for lvl in ['L1', 'L2', 'L3', 'L4', 'L5']:
        sub_b = [t for t in paired if t['complexity_level'] == lvl]
        if not sub_b:
            continue
        n = len(sub_b)
        b_err = 1 - sum(1 for t in sub_b if t['baseline_correct']) / n
        p_err = 1 - sum(1 for t in sub_b if t['proposed_correct']) / n
        complexity_trend[lvl] = {
            'n': n,
            'baseline_error_rate': b_err,
            'proposed_error_rate': p_err,
            'baseline_accuracy': 1 - b_err,
            'proposed_accuracy': 1 - p_err,
        }
        log(f'  {lvl}: n={n}, B_err={b_err:.4f}, P_err={p_err:.4f}')

    # 检查是否存在非线性增长 (简单趋势分析)
    if len(complexity_trend) >= 3:
        levels = sorted(complexity_trend.keys())
        b_errors = [complexity_trend[l]['baseline_error_rate'] for l in levels]
        p_errors = [complexity_trend[l]['proposed_error_rate'] for l in levels]

        # Spearman 相关 (使用 scipy)
        try:
            from scipy.stats import spearmanr, kendalltau
            rho_b, p_b = spearmanr(range(len(levels)), b_errors)
            rho_p, p_p = spearmanr(range(len(levels)), p_errors)
            tau_b, pt_b = kendalltau(range(len(levels)), b_errors)
            tau_p, pt_p = kendalltau(range(len(levels)), p_errors)
            complexity_trend['spearman'] = {
                'baseline': {'rho': rho_b, 'p_value': p_b},
                'proposed': {'rho': rho_p, 'p_value': p_p},
            }
            complexity_trend['kendall'] = {
                'baseline': {'tau': tau_b, 'p_value': pt_b},
                'proposed': {'tau': tau_p, 'p_value': pt_p},
            }
            log(f'\n  Spearman 相关 (复杂度 vs 错误率):')
            log(f'    Baseline: rho={rho_b:.4f}, p={p_b:.4f}')
            log(f'    Proposed: rho={rho_p:.4f}, p={p_p:.4f}')
            log(f'  Kendall Tau:')
            log(f'    Baseline: tau={tau_b:.4f}, p={pt_b:.4f}')
            log(f'    Proposed: tau={tau_p:.4f}, p={pt_p:.4f}')
        except ImportError:
            log('  [警告] scipy 不可用, 跳过 Spearman/Kendall 相关分析')

    # ============================================================
    # 9. 假设验证总结
    # ============================================================
    log('\n' + '=' * 60)
    log('[9] 研究假设验证总结')
    log('=' * 60)

    # H1: 结构化 Prompt 提高过程字段可观测性
    h1_evidence = process_fields
    h1_supported = (
        process_fields['proposed']['reasoning']['rate'] > process_fields['baseline']['reasoning']['rate'] and
        process_fields['proposed']['confidence']['rate'] > process_fields['baseline']['confidence']['rate']
    )
    log(f'\n  H1 (结构化Prompt提高过程字段可观测性): {"支持" if h1_supported else "不支持"}')
    log(f'    Baseline reasoning rate: {process_fields["baseline"]["reasoning"]["rate"]:.4f}')
    log(f'    Proposed reasoning rate: {process_fields["proposed"]["reasoning"]["rate"]:.4f}')

    # H2: Proposed 提高正确率
    h2_p_value = mcnemar_cc['p_value']
    h2_effect_size = abs(h)
    h2_supported = (diff > 0 and h2_p_value < 0.05 and h2_effect_size >= 0.2)
    h2_label = '支持' if h2_supported else ('部分支持' if diff > 0 else '不支持')
    log(f'\n  H2 (Proposed提高正确率): {h2_label}')
    log(f'    diff={diff:+.4f}, McNemar p={h2_p_value:.4f}, Cohen\'s h={h:+.4f}')

    # H3: 错误率随复杂度上升
    if 'spearman' in complexity_trend:
        h3_supported_b = complexity_trend['spearman']['baseline']['rho'] > 0
        h3_supported_p = complexity_trend['spearman']['proposed']['rho'] > 0
        h3_label = '支持' if (h3_supported_b and h3_supported_p) else '部分支持' if (h3_supported_b or h3_supported_p) else '不支持'
        log(f'\n  H3 (错误率随复杂度上升): {h3_label}')
        log(f'    Baseline: rho={complexity_trend["spearman"]["baseline"]["rho"]:+.4f}, p={complexity_trend["spearman"]["baseline"]["p_value"]:.4f}')
        log(f'    Proposed: rho={complexity_trend["spearman"]["proposed"]["rho"]:+.4f}, p={complexity_trend["spearman"]["proposed"]["p_value"]:.4f}')

    # ============================================================
    # 保存完整统计结果
    # ============================================================
    full_results = {
        'experiment_info': experiment_info,
        'overall_stats': overall_stats,
        'paired_analysis': {
            'n_paired': n_paired,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'baseline_only': baseline_only,
            'proposed_only': proposed_only,
            'baseline_accuracy': p_baseline,
            'proposed_accuracy': p_proposed,
            'accuracy_diff': diff,
            'baseline_wilson_ci': [ci_b_low, ci_b_high],
            'proposed_wilson_ci': [ci_p_low, ci_p_high],
            'mcnemar_with_cc': mcnemar_cc,
            'mcnemar_without_cc': mcnemar_no_cc,
            'cohens_h': h,
            'cohens_h_interpretation': h_interpretation,
            'bootstrap_ci_95': bootstrap,
        },
        'by_task_type': by_task_type,
        'by_complexity': by_complexity,
        'by_scorer': by_scorer,
        'process_fields': process_fields,
        'latency_stats': latency_stats,
        'complexity_trend': complexity_trend,
        'hypothesis_verification': {
            'H1_process_field_observability': {
                'supported': h1_supported,
                'evidence': h1_evidence,
            },
            'H2_accuracy_improvement': {
                'label': h2_label,
                'supported': h2_supported,
                'diff': diff,
                'p_value': h2_p_value,
                'effect_size_h': h,
            },
            'H3_complexity_error_rate': {
                'label': h3_label if 'spearman' in complexity_trend else '无法验证',
                'evidence': complexity_trend.get('spearman', {}),
            } if 'spearman' in complexity_trend else {'label': '无法验证 (scipy 不可用)'},
        },
    }

    output_file = os.path.join(BASE_DIR, 'data', 'analysis_records', 'statistical_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2, default=str)
    log(f'\n[完成] 完整统计结果已保存: {output_file}')

    # 保存配对分析(单独文件)
    paired_file = os.path.join(BASE_DIR, 'data', 'experiment_records', 'paired_analysis.json')
    with open(paired_file, 'w', encoding='utf-8') as f:
        json.dump(full_results['paired_analysis'], f, ensure_ascii=False, indent=2, default=str)
    log(f'  配对分析: {paired_file}')

    # 保存配对样本(用于 Excel)
    paired_samples_file = os.path.join(BASE_DIR, 'data', 'experiment_records', 'paired_samples.json')
    with open(paired_samples_file, 'w', encoding='utf-8') as f:
        json.dump(paired, f, ensure_ascii=False, indent=2, default=str)
    log(f'  配对样本: {paired_samples_file}')


if __name__ == '__main__':
    main()
