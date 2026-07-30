#!/usr/bin/env python3
"""
完整持久化实验脚本 v3
- 每条实验数据完整保存（输入、输出、所有元数据）
- 增量保存：每完成一条立即写入文件
- 断点续传：自动跳过已完成的任务
- 数据可视化：JSON格式，结构清晰
- 实时进度监控
"""
import csv
import json
import os
import platform
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Set
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_clients import get_client

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{run_id}.log")
    logger = logging.getLogger(f'CognitiveBoundary_v3_{run_id}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    tasks = []
    with open(tasks_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    return tasks


def load_prompt_template(template_path: str) -> Dict[str, Any]:
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_baseline_prompt(task_content: str) -> tuple:
    system_prompt = "你是一个专业的问答助手。请根据提供的问题给出准确的答案。"
    user_prompt = f"问题：{task_content}\n\n请给出你的答案。"
    return system_prompt, user_prompt


def format_proposed_prompt(task_content: str, task_type: str, template: Dict[str, Any]) -> tuple:
    system_prompt = template['system_prompt']
    user_prompt = template['user_prompt_template'].format(task_content=task_content)
    return system_prompt, user_prompt


def create_complete_record(
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    condition: str,
    response: Dict[str, Any],
    run_id: str,
    system_prompt: str,
    user_prompt: str,
    attempt_count: int = 1
) -> Dict[str, Any]:
    """
    创建完整的实验记录，包含所有输入、输出和元数据
    """
    timestamp = datetime.now().isoformat()
    record_id = f"{task['task_id']}_{model_config['model_id']}_{condition}_{run_id}"

    return {
        # ============ 唯一标识 ============
        "record_id": record_id,
        "run_id": run_id,

        # ============ 实验条件 ============
        "condition": condition,  # baseline 或 proposed
        "timestamp": timestamp,

        # ============ 任务信息（输入）============
        "task_info": {
            "task_id": task['task_id'],
            "task_type": task['task_type'],
            "complexity_level": task['complexity_level'],
            "domain": task['domain'],
            "task_content": task['task_content'],
            "ground_truth_type": task.get('ground_truth_type', ''),
            "ground_truth_source": task.get('ground_truth_source', '')
        },

        # ============ 模型配置 ============
        "model_config": {
            "model_id": model_config['model_id'],
            "model_name": model_config['model_name'],
            "model_version": model_config.get('model_version', ''),
            "provider": model_config['provider'],
            "temperature": model_config.get('temperature', 0.0),
            "max_output_tokens": model_config.get('max_output_tokens', 4096)
        },

        # ============ Prompt（完整输入）============
        "prompts": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        },

        # ============ 模型输出 ============
        "model_output": {
            "raw_output": response.get('raw_output'),
            "success": response.get('success', False),
            "error": response.get('error'),
            "finish_reason": response.get('finish_reason'),
            "model_version_returned": response.get('model_version')
        },

        # ============ 性能指标 ============
        "performance": {
            "latency_seconds": response.get('latency'),
            "attempt_count": attempt_count
        },

        # ============ API信息 ============
        "api_info": {
            "api_base": response.get('api_base'),
            "usage": response.get('usage', {})
        }
    }


def load_completed_records(output_file: str) -> Set[str]:
    """仅将成功且具有非空正文的记录视为已完成。"""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        output = record.get('model_output', {})
                        if output.get('success') and str(output.get('raw_output') or '').strip():
                            completed.add(record['record_id'])
                    except Exception:
                        pass
    return completed


def save_record_incremental(output_file: str, record: Dict[str, Any]) -> None:
    """按record_id追加或原子替换记录，并强制刷新到磁盘。"""
    existing_records = []
    record_replaced = False
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                existing = json.loads(line)
                if existing.get('record_id') == record['record_id']:
                    existing_records.append(record)
                    record_replaced = True
                else:
                    existing_records.append(existing)

    if not record_replaced:
        existing_records.append(record)

    temp_file = f"{output_file}.tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for existing in existing_records:
            f.write(json.dumps(existing, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, output_file)


def summarize_output_file(output_file: str) -> Dict[str, Any]:
    """从完整JSONL结果文件计算全量摘要，避免断点续跑只统计本轮新增。"""
    summary = {
        "total_jsonl_records": 0,
        "valid_json_records": 0,
        "success_count": 0,
        "fail_count": 0,
        "empty_success_count": 0,
        "conditions": {},
        "task_types": {},
        "paired_tasks": 0,
        "incomplete_pairs": 0,
        "duplicate_record_ids": 0,
    }
    seen_ids = set()
    task_conditions = {}
    if not os.path.exists(output_file):
        return summary

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            summary["total_jsonl_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            summary["valid_json_records"] += 1
            record_id = record.get('record_id')
            if record_id in seen_ids:
                summary["duplicate_record_ids"] += 1
            seen_ids.add(record_id)

            condition = record.get('condition', '')
            task_info = record.get('task_info', {})
            task_id = task_info.get('task_id', record.get('task_id', ''))
            task_type = task_info.get('task_type', record.get('task_type', ''))
            output = record.get('model_output', {})
            success = bool(output.get('success'))
            raw_output = str(output.get('raw_output') or '').strip()

            summary["conditions"][condition] = summary["conditions"].get(condition, 0) + 1
            summary["task_types"][task_type] = summary["task_types"].get(task_type, 0) + 1
            if success:
                summary["success_count"] += 1
                if not raw_output:
                    summary["empty_success_count"] += 1
            else:
                summary["fail_count"] += 1

            task_conditions.setdefault(task_id, set()).add(condition)

    complete_pairs = sum(1 for conds in task_conditions.values() if {'baseline', 'proposed'}.issubset(conds))
    summary["paired_tasks"] = complete_pairs
    summary["incomplete_pairs"] = len(task_conditions) - complete_pairs
    return summary


def collect_environment_metadata() -> Dict[str, Any]:
    """记录可复现实验所需的运行环境元数据。"""
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
    }


def test_model_with_timeout(client, model_id: str, timeout_seconds: int = 60) -> bool:
    """带超时的模型可用性测试"""
    import signal

    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("测试超时")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        test_response = client.generate(
            "你是一个测试助手。",
            "请回复'测试成功'四个字。"
        )
        signal.alarm(0)
        return test_response.get('success', False)
    except TimeoutError:
        return False
    except Exception:
        return False
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def run_experiment(resume_run_id: str = None):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, 'configs', '.env'))

    # 断点续传：使用指定的run_id或创建新的
    if resume_run_id:
        run_id = resume_run_id
        logger = setup_logging(os.path.join(BASE_DIR, 'logs'), run_id)
        logger.info("=" * 80)
        logger.info(f"断点续传实验 v3：{run_id}")
        logger.info("=" * 80)
    else:
        # 使用新的run_id，重新开始实验
        run_id = f"complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = setup_logging(os.path.join(BASE_DIR, 'logs'), run_id)
        logger.info("=" * 80)
        logger.info("完整持久化实验 v3：每条数据完整保存、支持断点续传")
        logger.info(f"运行ID: {run_id}")
        logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info("完整持久化实验 v3：每条数据完整保存、支持断点续传")
    logger.info(f"运行ID: {run_id}")
    logger.info("=" * 80)

    # 加载所有任务
    tasks_path = os.path.join(BASE_DIR, 'benchmark', 'pilot_tasks_sampled.csv')
    all_tasks = load_tasks(tasks_path)
    logger.info(f"总任务数: {len(all_tasks)}")

    # 使用全部150个任务
    target_tasks = 300
    if len(all_tasks) >= target_tasks:
        import random
        random.seed(20260726)
        tasks = random.sample(all_tasks, target_tasks)
    else:
        tasks = all_tasks
        logger.warning(f"任务数不足{target_tasks}，使用全部{len(tasks)}个任务")

    logger.info(f"选择 {len(tasks)} 个任务")

    # 加载Prompt模板
    templates = {}
    for task_type in ['knowledge', 'reasoning', 'planning']:
        template_path = os.path.join(BASE_DIR, 'prompts', f'{task_type}_template.json')
        templates[task_type] = load_prompt_template(template_path)
        logger.info(f"已加载模板: {task_type}")

    # API凭据从configs/.env或运行环境读取；禁止在脚本中硬编码密钥。
    if not os.getenv('LUNAGATE_API_KEY') and os.getenv('ANTHROPIC_API_KEY'):
        os.environ['LUNAGATE_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    if not os.getenv('LUNAGATE_API_BASE'):
        os.environ['LUNAGATE_API_BASE'] = 'https://lunagate.xyz'

    # 配置模型
    model_configs = [
        {
            "model_id": "claude-opus-4-6",
            "model_name": "claude-opus-4-6",
            "model_version": "claude-opus-4-6",
            "provider": "anthropic",
            "inference_mode": "api",
            "temperature": 0.0,
            "max_output_tokens": 4096,
            "api_key_env": "LUNAGATE_API_KEY",
            "api_base_env": "LUNAGATE_API_BASE",
            "enabled": True,
            "max_retries": 8,
            "retry_delay": 3
        }
    ]

    # 初始化客户端
    clients = {}
    for model_config in model_configs:
        try:
            client = get_client(model_config)
            clients[model_config['model_id']] = client
            logger.info(f"✅ 初始化模型: {model_config['model_id']}")
        except Exception as e:
            logger.error(f"❌ 初始化失败 {model_config['model_id']}: {e}")

    if not clients:
        logger.error("没有可用的模型客户端，实验终止")
        return

    # 测试模型可用性
    logger.info("=" * 80)
    logger.info("测试模型可用性（60秒超时）...")
    logger.info("=" * 80)

    available_clients = {}
    for model_id, client in clients.items():
        logger.info(f"正在测试模型 {model_id}...")
        if test_model_with_timeout(client, model_id, timeout_seconds=60):
            logger.info(f"✅ 模型 {model_id} 可用")
            available_clients[model_id] = client
        else:
            logger.error(f"❌ 模型 {model_id} 不可用或超时")

    if not available_clients:
        logger.error("没有可用的模型，实验终止")
        return

    clients = available_clients
    logger.info(f"可用模型数量: {len(clients)}/{len(model_configs)}")

    # 准备输出目录
    output_dir = os.path.join(BASE_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"raw_outputs_{run_id}.jsonl")

    # 加载已完成的记录（断点续传）
    completed_records = load_completed_records(output_file)
    logger.info(f"已完成记录数: {len(completed_records)}")

    # 保存元数据
    metadata = {
        "experiment_name": "Complete_Persistence_Experiment_v3",
        "run_id": run_id,
        "design": "within-subjects",
        "description": "完整持久化实验：每条数据完整保存（输入、输出、所有元数据），支持断点续传",
        "num_tasks": len(tasks),
        "conditions": ["baseline", "proposed"],
        "created_at": datetime.now().isoformat(),
        "version": "v3",
        "features": [
            "incremental_save",
            "checkpoint_resume",
            "complete_data_persistence",
            "json_visualization"
        ],
        "model_configs": model_configs,
        "environment": collect_environment_metadata(),
        "random_seed": 20260726,
        "task_source": tasks_path
    }

    metadata_file = os.path.join(output_dir, f"metadata_{run_id}.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"实验元数据已保存: {metadata_file}")

    # 计算总运行次数
    total_runs = len(tasks) * len(clients) * 2
    logger.info(f"总运行次数: {total_runs} ({len(tasks)} 任务 × {len(clients)} 模型 × 2 条件)")

    # 运行实验
    success_count = 0
    fail_count = 0
    skipped_count = 0

    with tqdm(total=total_runs, desc="Running experiment", initial=len(completed_records)) as pbar:
        for task_idx, task in enumerate(tasks):
            task_type = task['task_type']
            task_content = task['task_content']

            for model_id, client in clients.items():
                model_config = next(m for m in model_configs if m['model_id'] == model_id)

                # ============ Baseline 条件 ============
                record_id_baseline = f"{task['task_id']}_{model_config['model_id']}_baseline_{run_id}"
                if record_id_baseline in completed_records:
                    skipped_count += 1
                else:
                    try:
                        system_prompt, user_prompt = format_baseline_prompt(task_content)
                        response = client.generate(system_prompt, user_prompt)

                        record = create_complete_record(
                            task=task,
                            model_config=model_config,
                            condition='baseline',
                            response=response,
                            run_id=run_id,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt
                        )

                        # 增量保存
                        save_record_incremental(output_file, record)
                        completed_records.add(record_id_baseline)

                        if response.get('success'):
                            success_count += 1
                        else:
                            fail_count += 1

                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Baseline运行失败: {task['task_id']} - {e}")
                        fail_count += 1

                pbar.update(1)

                # ============ Proposed 条件 ============
                record_id_proposed = f"{task['task_id']}_{model_config['model_id']}_proposed_{run_id}"
                if record_id_proposed in completed_records:
                    skipped_count += 1
                else:
                    try:
                        template = templates.get(task_type, templates['knowledge'])
                        system_prompt, user_prompt = format_proposed_prompt(task_content, task_type, template)
                        response = client.generate(system_prompt, user_prompt)

                        record = create_complete_record(
                            task=task,
                            model_config=model_config,
                            condition='proposed',
                            response=response,
                            run_id=run_id,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt
                        )

                        # 增量保存
                        save_record_incremental(output_file, record)
                        completed_records.add(record_id_proposed)

                        if response.get('success'):
                            success_count += 1
                        else:
                            fail_count += 1

                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Proposed运行失败: {task['task_id']} - {e}")
                        fail_count += 1

                pbar.update(1)

            # 每10个任务打印一次进度摘要
            if (task_idx + 1) % 10 == 0:
                logger.info(
                    f"进度摘要: 任务 {task_idx + 1}/{len(tasks)} | "
                    f"成功: {success_count} | 失败: {fail_count} | 跳过: {skipped_count} | "
                    f"总记录: {len(completed_records)}/{total_runs}"
                )

    # 实验完成摘要
    logger.info("=" * 80)
    logger.info("实验完成")
    logger.info(f"总记录数: {len(completed_records)}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {fail_count}")
    logger.info(f"跳过（断点续传）: {skipped_count}")
    logger.info(f"结果文件: {output_file}")
    logger.info("=" * 80)

    # 生成实验摘要JSON：从完整结果文件重新统计，而不是只统计本轮续跑新增。
    file_summary = summarize_output_file(output_file)
    summary = {
        "run_id": run_id,
        "completed_at": datetime.now().isoformat(),
        "total_records": file_summary["valid_json_records"],
        "success_count": file_summary["success_count"],
        "fail_count": file_summary["fail_count"],
        "skipped_count_this_invocation": skipped_count,
        "new_success_count_this_invocation": success_count,
        "new_fail_count_this_invocation": fail_count,
        "output_file": output_file,
        "file_summary": file_summary
    }
    summary_file = os.path.join(output_dir, f"summary_{run_id}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_file


if __name__ == "__main__":
    # 支持命令行参数指定run_id进行断点续传
    resume_id = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = run_experiment(resume_run_id=resume_id)
    print(f"\n实验完成，结果保存在: {output_file}")
