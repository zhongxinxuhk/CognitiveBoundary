#!/usr/bin/env python3
"""
评分脚本，对解析后的输出进行评分
"""
import json
import os
import re
import sys
from typing import Dict, Any, List
import pandas as pd

def extract_answer_from_ground_truth(ground_truth_source: str) -> str:
    """从ground_truth_source中提取答案"""
    # 格式: "MMLU answer key: D"
    if "answer key:" in ground_truth_source.lower():
        match = re.search(r"answer key:\s*([A-Z])", ground_truth_source, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 其他格式
    return ground_truth_source

def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    if not answer:
        return ""
    
    # 移除空白字符
    answer = answer.strip()
    
    # 如果是单个字母，转换为大写
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    
    # 如果是选项格式，提取字母
    match = re.match(r"^([A-Z])[\.\):]", answer)
    if match:
        return match.group(1).upper()
    
    return answer

def score_single_answer(record: Dict[str, Any]) -> Dict[str, Any]:
    """评分single_answer类型"""
    final_answer = record.get('final_answer', '')
    ground_truth = record.get('ground_truth_source', '')
    
    if not final_answer or not ground_truth:
        record['correctness'] = 0.0
        record['score_notes'] = "缺少答案或参考答案"
        return record
    
    # 提取参考答案
    expected_answer = extract_answer_from_ground_truth(ground_truth)
    
    # 标准化答案
    normalized_final = normalize_answer(final_answer)
    normalized_expected = normalize_answer(expected_answer)
    
    # 比较答案
    if normalized_final == normalized_expected:
        record['correctness'] = 1.0
        record['score_notes'] = "答案正确"
    else:
        record['correctness'] = 0.0
        record['score_notes'] = f"答案错误: 期望 {normalized_expected}, 得到 {normalized_final}"
    
    return record

def score_rubric(record: Dict[str, Any]) -> Dict[str, Any]:
    """评分rubric类型（占位，需要人工评分）"""
    record['correctness'] = None
    record['score_notes'] = "需要人工评分"
    record['needs_human_review'] = True
    return record

def score_executable_check(record: Dict[str, Any]) -> Dict[str, Any]:
    """评分executable_check类型（占位，需要可执行检查）"""
    record['correctness'] = None
    record['score_notes'] = "需要可执行检查"
    record['needs_human_review'] = True
    return record

def score_expert_review(record: Dict[str, Any]) -> Dict[str, Any]:
    """评分expert_review类型（占位，需要专家评审）"""
    record['correctness'] = None
    record['score_notes'] = "需要专家评审"
    record['needs_human_review'] = True
    return record

def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """对单条记录进行评分"""
    ground_truth_type = record.get('ground_truth_type', '')
    
    # 初始化评分字段
    record['correctness'] = None
    record['task_success'] = None
    record['score_notes'] = ""
    record['needs_human_review'] = False
    
    # 根据ground_truth_type选择评分方法
    if ground_truth_type == 'single_answer':
        record = score_single_answer(record)
    elif ground_truth_type == 'rubric':
        record = score_rubric(record)
    elif ground_truth_type == 'executable_check':
        record = score_executable_check(record)
    elif ground_truth_type == 'expert_review':
        record = score_expert_review(record)
    else:
        record['score_notes'] = f"未知的ground_truth_type: {ground_truth_type}"
        record['needs_human_review'] = True
    
    # 设置task_success（对于规划和Agent任务）
    if record['task_type'] in ['planning', 'agent'] and record['correctness'] is not None:
        record['task_success'] = record['correctness']
    
    return record

def score_outputs_file(input_path: str, output_path: str):
    """对整个输出文件进行评分"""
    records = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                scored_record = score_record(record)
                records.append(scored_record)
    
    # 保存评分后的记录
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"评分完成，共 {len(records)} 条记录")
    print(f"保存到: {output_path}")
    
    # 统计评分结果
    auto_scored = sum(1 for r in records if r.get('correctness') is not None)
    needs_review = sum(1 for r in records if r.get('needs_human_review', False))
    
    print(f"自动评分: {auto_scored} 条")
    print(f"需要人工评审: {needs_review} 条")
    
    # 按任务类型统计正确率
    task_type_stats = {}
    for record in records:
        task_type = record.get('task_type')
        if task_type not in task_type_stats:
            task_type_stats[task_type] = {'correct': 0, 'total': 0}
        
        if record.get('correctness') is not None:
            task_type_stats[task_type]['total'] += 1
            if record.get('correctness') == 1.0:
                task_type_stats[task_type]['correct'] += 1
    
    print("\n按任务类型正确率:")
    for task_type, stats in task_type_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"  {task_type}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    return records

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python score_outputs.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)
    
    score_outputs_file(input_path, output_path)