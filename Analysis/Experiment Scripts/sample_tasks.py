#!/usr/bin/env python3
"""
从pilot_tasks_manifest.csv中按配置抽样
"""
import csv
import json
import os
import random
from collections import defaultdict

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_manifest(manifest_path):
    """加载manifest文件，按task_type和complexity_level分组"""
    tasks = defaultdict(lambda: defaultdict(list))
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_type = row['task_type']
            complexity_level = row['complexity_level']
            tasks[task_type][complexity_level].append(row)
    
    return tasks

def sample_tasks(tasks, sampling_config, random_seed):
    """根据配置抽样"""
    random.seed(random_seed)
    sampled_tasks = []
    
    for task_type, config in sampling_config.items():
        if task_type not in tasks:
            print(f"警告: 任务类型 {task_type} 在manifest中不存在")
            continue
        
        print(f"抽样 {task_type}:")
        for level, count in config['per_complexity'].items():
            if count == 0:
                continue
            
            available = tasks[task_type].get(level, [])
            if len(available) < count:
                print(f"  {level}: 需要 {count}, 可用 {len(available)}，使用所有可用")
                sampled = available
            else:
                sampled = random.sample(available, count)
                print(f"  {level}: 抽样 {count}/{len(available)}")
            
            sampled_tasks.extend(sampled)
    
    return sampled_tasks

def save_sampled_tasks(sampled_tasks, output_path):
    """保存抽样结果"""
    if not sampled_tasks:
        print("错误: 没有抽样到任何任务")
        return
    
    fieldnames = sampled_tasks[0].keys()
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_tasks)
    
    print(f"\n抽样完成，共 {len(sampled_tasks)} 条任务")
    print(f"保存到: {output_path}")

def print_summary(sampled_tasks):
    """打印抽样摘要"""
    task_counts = defaultdict(lambda: defaultdict(int))
    for task in sampled_tasks:
        task_counts[task['task_type']][task['complexity_level']] += 1
    
    print("\n抽样摘要:")
    for task_type, complexity_counts in sorted(task_counts.items()):
        print(f"  {task_type}: {sum(complexity_counts.values())}")
        for level, count in sorted(complexity_counts.items()):
            print(f"    {level}: {count}")

if __name__ == "__main__":
    base_dir = "/Users/xuzhongxin/.trae/work/6a43ca227e513dd811e706c6/CognitiveBoundary"
    manifest_path = "/Users/xuzhongxin/Desktop/人工智能论文/pilot_data/manifests/pilot_tasks_manifest.csv"
    config_path = os.path.join(base_dir, "configs/experiment_config.json")
    output_path = os.path.join(base_dir, "benchmark/pilot_tasks_sampled.csv")
    
    # 加载配置
    config = load_config(config_path)
    random_seed = config['random_seed']
    
    # 加载manifest
    tasks = load_manifest(manifest_path)
    
    # 抽样
    sampled_tasks = sample_tasks(tasks, config['task_sampling'], random_seed)
    
    # 保存结果
    save_sampled_tasks(sampled_tasks, output_path)
    
    # 打印摘要
    print_summary(sampled_tasks)