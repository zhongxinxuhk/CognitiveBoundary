#!/usr/bin/env python3
"""
对比实验数据分析
baseline（无干预）vs proposed（认知边界检测方案）
汇总 → 清洗 → 计算 → 对比
"""
import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. 数据加载与清洗
# ============================================================

def load_outputs(file_path: str) -> pd.DataFrame:
    """加载JSONL输出文件"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    # 移除完全失败的记录（raw_output为空）
    df = df[df['raw_output'].notna() & (df['raw_output'] != '')].copy()
    
    # 标准化成功状态
    df['success'] = df['success'].fillna(False)
    
    return df

# ============================================================
# 2. 答案解析
# ============================================================

def extract_answer_choice(text: str) -> Optional[str]:
    """提取选择题答案 (A/B/C/D)"""
    if not text:
        return None
    # 查找 "答案是X" 或 "Answer: X" 等模式
    patterns = [
        r'(?:答案|answer)[是为：:\s]*([A-D])',
        r'(?:最终答案|final answer)[是为：:\s]*([A-D])',
        r'^([A-D])[\.\s）\)]',
        r'\b([A-D])\b'
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None

def extract_answer_content(text: str) -> str:
    """提取答案内容"""
    if not text:
        return ""
    # 尝试提取"最终答案"后的内容
    patterns = [
        r'(?:最终答案|final answer)[：:\s]*(.*?)(?:\n|$)',
        r'(?:答案|answer)[：:\s]*(.*?)(?:\n|$)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # 返回前200字符
    return text[:200].strip()

def extract_confidence(text: str) -> Optional[str]:
    """提取置信度表达"""
    if not text:
        return None
    patterns = [
        r'置信度[（(]高/中/低[）)][：:]\s*(高|中|低)',
        r'置信度[：:]\s*(高|中|低)',
        r'(high|medium|low)\s*confidence',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = m.group(1).lower()
            if val in ['高', 'high']:
                return 'high'
            elif val in ['中', 'medium']:
                return 'medium'
            elif val in ['低', 'low']:
                return 'low'
    return None

def extract_refusal(text: str) -> bool:
    """检测是否拒答"""
    if not text:
        return False
    refusal_keywords = [
        "无法确定", "无法判断", "需要进一步验证", "不确定",
        "不知道", "无法回答", "无法完成", "无法规划",
        "cannot determine", "uncertain", "I don't know",
        "无法确认", "没有足够信息", "信息不足"
    ]
    for kw in refusal_keywords:
        if kw.lower() in text.lower():
            return True
    return False

# ============================================================
# 3. 评分
# ============================================================

def score_task(row: pd.Series) -> Dict[str, Any]:
    """对单条记录评分"""
    result = {
        'correctness': None,
        'has_reasoning': False,
        'has_confidence': False,
        'confidence_level': None,
        'has_refusal': False,
        'is_correct': None
    }
    
    raw = row.get('raw_output', '')
    if not raw:
        return result
    
    # 提取答案
    gt_type = row.get('ground_truth_type', '')
    gt_source = row.get('ground_truth_source', '')
    
    if gt_type == 'single_answer' and gt_source:
        # 提取参考答案
        m = re.search(r'answer key:\s*([A-D])', gt_source, re.IGNORECASE)
        if m:
            expected = m.group(1).upper()
            predicted = extract_answer_choice(raw)
            if predicted:
                result['correctness'] = 1.0 if predicted == expected else 0.0
                result['is_correct'] = predicted == expected
    
    # 检测推理过程
    result['has_reasoning'] = bool(re.search(r'(?:推理|reasoning|步骤|step)', raw, re.IGNORECASE))
    
    # 提取置信度
    conf = extract_confidence(raw)
    result['has_confidence'] = conf is not None
    result['confidence_level'] = conf
    
    # 检测拒答
    result['has_refusal'] = extract_refusal(raw)
    
    return result

# ============================================================
# 4. 指标计算
# ============================================================

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """计算实验指标"""
    metrics = {}
    
    # 基础指标
    metrics['total_tasks'] = len(df)
    metrics['success_rate'] = df['success'].mean()
    
    # 正确率（仅对有ground_truth的任务）
    scored = df[df['is_correct'].notna()]
    if len(scored) > 0:
        metrics['accuracy'] = scored['is_correct'].mean()
    else:
        metrics['accuracy'] = None
    
    # 按复杂度统计
    complexity_stats = {}
    for level in sorted(df['complexity_level'].unique()):
        sub = df[df['complexity_level'] == level]
        scored_sub = sub[sub['is_correct'].notna()]
        complexity_stats[level] = {
            'count': len(sub),
            'accuracy': scored_sub['is_correct'].mean() if len(scored_sub) > 0 else None,
            'success_rate': sub['success'].mean()
        }
    metrics['by_complexity'] = complexity_stats
    
    # 按任务类型统计
    type_stats = {}
    for ttype in df['task_type'].unique():
        sub = df[df['task_type'] == ttype]
        scored_sub = sub[sub['is_correct'].notna()]
        type_stats[ttype] = {
            'count': len(sub),
            'accuracy': scored_sub['is_correct'].mean() if len(scored_sub) > 0 else None,
        }
    metrics['by_task_type'] = type_stats
    
    # 元认知指标（仅proposed条件）
    if 'has_confidence' in df.columns:
        metrics['confidence_rate'] = df['has_confidence'].mean()
        metrics['refusal_rate'] = df['has_refusal'].mean()
        metrics['reasoning_rate'] = df['has_reasoning'].mean()
        
        # 置信度分布
        conf_dist = df['confidence_level'].value_counts(normalize=True).to_dict()
        metrics['confidence_distribution'] = conf_dist
        
        # 高置信错误率（关键指标）
        high_conf = df[df['confidence_level'] == 'high']
        if len(high_conf) > 0:
            scored_high = high_conf[high_conf['is_correct'].notna()]
            if len(scored_high) > 0:
                metrics['high_confidence_error_rate'] = 1 - scored_high['is_correct'].mean()
        
        # 合理拒答率
        refused = df[df['has_refusal'] == True]
        if len(refused) > 0:
            unanswerable = df[df['answerable_label'] == 'unanswerable']
            if len(unanswerable) > 0:
                refused_unanswerable = refused[refused['answerable_label'] == 'unanswerable']
                metrics['correct_refusal_rate'] = len(refused_unanswerable) / len(unanswerable) if len(unanswerable) > 0 else 0
    
    # 延迟统计
    if 'latency' in df.columns:
        latencies = df['latency'].dropna()
        if len(latencies) > 0:
            metrics['latency_mean'] = latencies.mean()
            metrics['latency_median'] = latencies.median()
            metrics['latency_std'] = latencies.std()
    
    return metrics

# ============================================================
# 5. 对比分析
# ============================================================

def compare_conditions(baseline_metrics: Dict, proposed_metrics: Dict) -> Dict[str, Any]:
    """对比两个条件的指标"""
    comparison = {}
    
    # 正确率对比
    if baseline_metrics.get('accuracy') is not None and proposed_metrics.get('accuracy') is not None:
        comparison['accuracy_baseline'] = baseline_metrics['accuracy']
        comparison['accuracy_proposed'] = proposed_metrics['accuracy']
        comparison['accuracy_diff'] = proposed_metrics['accuracy'] - baseline_metrics['accuracy']
    
    # 按复杂度对比
    complexity_comparison = {}
    for level in set(list(baseline_metrics.get('by_complexity', {}).keys()) + 
                     list(proposed_metrics.get('by_complexity', {}).keys())):
        b = baseline_metrics.get('by_complexity', {}).get(level, {})
        p = proposed_metrics.get('by_complexity', {}).get(level, {})
        complexity_comparison[level] = {
            'accuracy_baseline': b.get('accuracy'),
            'accuracy_proposed': p.get('accuracy'),
            'error_rate_baseline': 1 - b['accuracy'] if b.get('accuracy') else None,
            'error_rate_proposed': 1 - p['accuracy'] if p.get('accuracy') else None,
        }
    comparison['by_complexity'] = complexity_comparison
    
    # 元认知指标（仅proposed有）
    comparison['proposed_only'] = {
        'confidence_rate': proposed_metrics.get('confidence_rate'),
        'refusal_rate': proposed_metrics.get('refusal_rate'),
        'reasoning_rate': proposed_metrics.get('reasoning_rate'),
        'high_confidence_error_rate': proposed_metrics.get('high_confidence_error_rate'),
        'confidence_distribution': proposed_metrics.get('confidence_distribution'),
    }
    
    # 延迟对比
    comparison['latency_baseline'] = baseline_metrics.get('latency_mean')
    comparison['latency_proposed'] = proposed_metrics.get('latency_mean')
    
    return comparison

# ============================================================
# 6. 生成报告
# ============================================================

def generate_report(baseline_metrics: Dict, proposed_metrics: Dict, 
                   comparison: Dict, output_dir: str, run_id: str) -> str:
    """生成对比报告"""
    report_path = os.path.join(output_dir, f"comparison_report_{run_id}.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 认知边界实验对比报告\n\n")
        f.write(f"**运行ID**: {run_id}  \n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        
        f.write("## 1. 实验概述\n\n")
        f.write(f"| 指标 | Baseline | Proposed |\n")
        f.write(f"|------|----------|----------|\n")
        f.write(f"| 总任务数 | {baseline_metrics.get('total_tasks', 'N/A')} | {proposed_metrics.get('total_tasks', 'N/A')} |\n")
        f.write(f"| 成功率 | {baseline_metrics.get('success_rate', 0):.2%} | {proposed_metrics.get('success_rate', 0):.2%} |\n")
        
        b_acc = baseline_metrics.get('accuracy')
        p_acc = proposed_metrics.get('accuracy')
        f.write(f"| 正确率 | {f'{b_acc:.2%}' if b_acc is not None else 'N/A'} | {f'{p_acc:.2%}' if p_acc is not None else 'N/A'} |\n")
        
        f.write(f"| 平均延迟 | {f\"{baseline_metrics.get('latency_mean', 0):.1f}s\"} | {f\"{proposed_metrics.get('latency_mean', 0):.1f}s\"} |\n\n")
        
        # 按复杂度对比
        f.write("## 2. 按复杂度等级对比\n\n")
        f.write("| 复杂度 | Baseline错误率 | Proposed错误率 | 差异 |\n")
        f.write("|--------|---------------|---------------|------|\n")
        for level in sorted(comparison.get('by_complexity', {}).keys()):
            bc = comparison['by_complexity'][level]
            b_err = bc.get('error_rate_baseline')
            p_err = bc.get('error_rate_proposed')
            diff = (p_err - b_err) if (b_err is not None and p_err is not None) else None
            f.write(f"| {level} | {f'{b_err:.2%}' if b_err is not None else 'N/A'} | {f'{p_err:.2%}' if p_err is not None else 'N/A'} | {f'{diff:+.2%}' if diff is not None else 'N/A'} |\n")
        
        # 元认知指标
        f.write("\n## 3. 元认知指标（仅Proposed条件）\n\n")
        po = comparison.get('proposed_only', {})
        f.write(f"| 指标 | 值 |\n")
        f.write(f"|------|----|\n")
        f.write(f"| 置信度表达率 | {po.get('confidence_rate', 0):.2%} |\n")
        f.write(f"| 拒答率 | {po.get('refusal_rate', 0):.2%} |\n")
        f.write(f"| 推理过程率 | {po.get('reasoning_rate', 0):.2%} |\n")
        f.write(f"| 高置信错误率 | {po.get('high_confidence_error_rate', 0):.2%} |\n\n")
        
        conf_dist = po.get('confidence_distribution', {})
        if conf_dist:
            f.write("### 置信度分布\n\n")
            f.write("| 级别 | 比例 |\n")
            f.write("|------|------|\n")
            for level, ratio in conf_dist.items():
                f.write(f"| {level} | {ratio:.2%} |\n")
        
        f.write("\n## 4. 结论\n\n")
        if b_acc is not None and p_acc is not None:
            if p_acc > b_acc:
                f.write("✅ **Proposed方案正确率高于Baseline**，认知边界检测方案有效。\n")
            elif p_acc < b_acc:
                f.write("⚠️ **Proposed方案正确率低于Baseline**，需要进一步优化。\n")
            else:
                f.write("➡️ 两者正确率相同，需要从元认知指标角度分析。\n")
        
        f.write("\n---\n")
        f.write("*本报告由 CognitiveBoundary 自动生成*\n")
    
    return report_path

# ============================================================
# 主流程
# ============================================================

def main():
    if len(sys.argv) < 2:
        # 查找最新的run_id
        output_dir = os.path.join(BASE_DIR, "outputs")
        files = [f for f in os.listdir(output_dir) if f.startswith("raw_outputs_") and f.endswith(".jsonl")]
        if not files:
            print("❌ 没有找到实验输出文件")
            print("请先运行: python3 scripts/run_experiment.py")
            return
        
        # 按文件名中的run_id分组
        run_ids = set()
        for f in files:
            # raw_outputs_condition_run_id.jsonl
            parts = f.replace("raw_outputs_", "").replace(".jsonl", "").rsplit("_", 1)
            if len(parts) >= 2:
                run_ids.add(parts[-1])
        
        if not run_ids:
            print("❌ 无法解析run_id")
            return
        
        print("可用的实验运行:")
        for i, rid in enumerate(sorted(run_ids, reverse=True), 1):
            matching = [f for f in files if rid in f]
            print(f"  {i}. {rid} ({', '.join(matching)})")
        
        choice = input("\n选择 (输入数字): ").strip()
        try:
            run_id = sorted(run_ids, reverse=True)[int(choice) - 1]
        except (ValueError, IndexError):
            print("❌ 无效选择")
            return
    else:
        run_id = sys.argv[1]
    
    output_dir = os.path.join(BASE_DIR, "outputs")
    
    # 查找对应的文件
    baseline_file = os.path.join(output_dir, f"raw_outputs_baseline_{run_id}.jsonl")
    proposed_file = os.path.join(output_dir, f"raw_outputs_proposed_{run_id}.jsonl")
    
    has_baseline = os.path.exists(baseline_file)
    has_proposed = os.path.exists(proposed_file)
    
    if not has_baseline and not has_proposed:
        print(f"❌ 找不到 run_id={run_id} 的输出文件")
        return
    
    print(f"\n分析 run_id: {run_id}")
    print(f"  Baseline: {'✅' if has_baseline else '❌'}")
    print(f"  Proposed: {'✅' if has_proposed else '❌'}")
    
    # 处理每个条件
    all_metrics = {}
    all_dfs = {}
    
    for cond, filepath in [("baseline", baseline_file), ("proposed", proposed_file)]:
        if not os.path.exists(filepath):
            continue
        
        print(f"\n处理 {cond}...")
        df = load_outputs(filepath)
        df = clean_data(df)
        
        # 解析和评分
        scores = df.apply(score_task, axis=1, result_type='expand')
        df = pd.concat([df, scores], axis=1)
        
        all_dfs[cond] = df
        all_metrics[cond] = compute_metrics(df)
        
        # 保存评分后的数据
        scored_path = os.path.join(BASE_DIR, "annotations", f"scored_{cond}_{run_id}.csv")
        os.makedirs(os.path.dirname(scored_path), exist_ok=True)
        df.to_csv(scored_path, index=False, encoding='utf-8-sig')
        print(f"  保存: {scored_path}")
    
    # 对比分析
    if has_baseline and has_proposed:
        print("\n生成对比报告...")
        comparison = compare_conditions(all_metrics['baseline'], all_metrics['proposed'])
        report_path = generate_report(all_metrics['baseline'], all_metrics['proposed'], 
                                     comparison, output_dir, run_id)
        print(f"报告: {report_path}")
        
        # 保存对比数据为JSON
        comparison_json = os.path.join(output_dir, f"comparison_{run_id}.json")
        with open(comparison_json, 'w', encoding='utf-8') as f:
            json.dump({
                "run_id": run_id,
                "baseline": all_metrics['baseline'],
                "proposed": all_metrics['proposed'],
                "comparison": comparison
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"数据: {comparison_json}")
    else:
        # 单条件报告
        cond = "baseline" if has_baseline else "proposed"
        print(f"\n{cond} 指标:")
        for k, v in all_metrics[cond].items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
    
    print("\n✅ 分析完成")

if __name__ == "__main__":
    main()
