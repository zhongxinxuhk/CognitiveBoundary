#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 — pre-publication four-arm experiment runner.

Public-release scope
--------------------
This file contains execution logic only. It does not include the manuscript,
research data, model outputs, statistical results, credentials, or private
service endpoints.

Safety properties
-----------------
* Planning is the default; network execution requires the explicit
  ``--execute`` flag.
* The API credential is read only from ``P2_API_KEY``.
* A connectivity preflight runs before any model request.
* Outputs include model, seed, temperature, prompt hash, and timestamps.
* Existing ``(arm, seed, task_id)`` records are skipped on resume.

Example
-------
    python exp_four_arm_runner.py --task-file /path/to/scored_outputs.jsonl
    P2_API_KEY=... python exp_four_arm_runner.py --execute \
        --task-file /path/to/scored_outputs.jsonl \
        --model YOUR_MODEL --arms a0,ac,ai,ap --seeds 0,1,2
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ------------------------------------------------------------------ 提示词模板
# 说明：a0 / ap 两臂的模板**逐字取自原实验记录**（results/scored_outputs.jsonl 的
# prompts 字段），不得改动，否则复现性失效。

SYS_BASE = {
    "knowledge": "你是一个专业的问答助手。请根据提供的问题给出准确的答案。",
    "reasoning": "你是一个专业的问答助手。请根据提供的问题给出准确的答案。",
    "planning": "你是一个专业的问答助手。请根据提供的问题给出准确的答案。",
}
SYS_PROPOSED = {
    "knowledge": "你是一个专业的知识问答助手。请根据提供的问题给出准确、可靠的答案。"
                 "如果问题超出你的知识范围，或者证据不足，请明确表示无法确定或需要进一步验证。"
                 "请展示你的推理过程，并表达你的置信度。",
    "reasoning": "你是一个专业的推理助手。请仔细分析问题，逐步推理，确保逻辑严密。"
                 "如果遇到不确定或矛盾之处，请明确指出。请展示完整的推理过程，并评估每一步的可靠性。",
    "planning": "你是一个专业的规划助手。请根据任务目标和约束条件，制定详细、可执行的计划。"
                "请考虑所有约束条件，评估计划的可行性，并识别潜在风险。如果任务无法完成，请说明原因。",
}
# 【新增】A2 对照臂系统提示：要求逐步推理，但**不涉及**置信度/不确定性自报
SYS_COT = {
    "knowledge": "你是一个专业的问答助手。请先逐步推理，再给出准确答案。",
    "reasoning": "你是一个专业的推理助手。请逐步推理，确保逻辑严密，最后给出结论。",
    "planning": "你是一个专业的规划助手。请先分析目标与约束，再给出可执行的计划。",
}

TAIL_A0 = {
    "knowledge": "请只在最终答案中给出 A/B/C/D，并说明必要依据。\n\n请给出你的答案。",
    "reasoning": "请给出你的答案。",
    "planning": "请给出你的答案。",
}
TAIL_AP = {
    "knowledge": "请只在最终答案中给出 A/B/C/D，并说明必要依据。\n\n请按照以下格式回答：\n"
                 "1. 最终答案：\n2. 推理过程：\n3. 置信度（高/中/低）：\n"
                 "4. 不确定性说明（如果需要）：",
    "reasoning": "请按照以下格式回答：\n1. 最终答案：\n2. 逐步推理过程：\n"
                 "   - 步骤1：\n   - 步骤2：\n   - ...\n3. 关键假设：\n"
                 "4. 不确定性或潜在错误：\n5. 置信度（高/中/低）：",
    "planning": "请按照以下格式回答：\n1. 最终答案：\n2. 目标分析：\n3. 约束条件：\n"
                "4. 计划步骤：\n5. 潜在风险：\n6. 置信度（高/中/低）：",
}
# 【新增】A2 对照臂尾句：长度匹配约束（目标 ≥ 400 字符，对齐 AP 臂中位数 475）
TAIL_AC = {
    "knowledge": "请先逐步推理，再只在最终答案中给出 A/B/C/D，并说明必要依据。"
                 "推理过程请充分展开（不少于 400 字符），但不要给出置信度或不确定性评级。",
    "reasoning": "请逐步展开推理过程（不少于 400 字符）后给出最终答案。"
                 "不要给出置信度或不确定性评级。",
    "planning": "请先分析目标与约束，再给出详细可执行的计划步骤（整体不少于 400 字符）。"
                "不要给出置信度或不确定性评级。",
}

ARM_DEFS = {
    "a0": {"sys": SYS_BASE, "tail": TAIL_A0, "logprobs": False,
           "desc": "Baseline 复现臂"},
    "ac": {"sys": SYS_COT, "tail": TAIL_AC, "logprobs": False,
           "desc": "长度匹配通用 CoT 对照臂（新增，检验 A2）"},
    "ai": {"sys": SYS_BASE, "tail": TAIL_A0, "logprobs": True,
           "desc": "内部信号臂（新增，采集 logprobs，检验 A1）"},
    "ap": {"sys": SYS_PROPOSED, "tail": TAIL_AP, "logprobs": False,
           "desc": "Proposed 复现臂"},
}

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"


# ------------------------------------------------------------------ 任务清单

def load_tasks(src, limit=None):
    """Load unique tasks from a user-supplied scored-output JSONL file."""
    tasks, seen = [], set()
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ti = r.get("task_info") or {}
            tid = ti.get("task_id")
            if not tid or tid in seen:
                continue
            seen.add(tid)
            up = (r.get("prompts") or {}).get("user_prompt", "")
            # 去掉原臂的尾部指令，保留**题干本体及其原有结尾换行**
            # （不可 rstrip 后再自行拼接，否则与原始记录不再逐字一致 —— 2026-09-20 踩到）
            stem = up
            for tail in list(TAIL_AP.values()) + list(TAIL_A0.values()):
                if tail in up:
                    stem = up.split(tail)[0]
                    break
            tasks.append({
                "task_id": tid,
                "task_type": ti.get("task_type"),
                "level": ti.get("complexity_level"),
                "stem": stem,
            })
    tasks.sort(key=lambda x: x["task_id"])
    return tasks[:limit] if limit else tasks


def build_prompt(arm, task):
    """拼装提示。stem 已含原始结尾换行，此处**直接拼接**，不额外插入分隔符。"""
    d = ARM_DEFS[arm]
    tt = task["task_type"]
    return {
        "system_prompt": d["sys"][tt],
        "user_prompt": f"{task['stem']}{d['tail'][tt]}",
    }


def prompt_sha(pr):
    h = hashlib.sha256()
    h.update((pr["system_prompt"] + "\x00" + pr["user_prompt"]).encode("utf-8"))
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ 预检

def endpoint_preflight(endpoint, timeout=6):
    """TCP + HTTPS 可达性预检。返回 (ok, detail)。"""
    p = urlparse(endpoint)
    host, port = p.hostname, (p.port or 443)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        return False, f"TCP 连接失败 {host}:{port} — {e}"
    try:
        import urllib.request
        req = urllib.request.Request(endpoint.rstrip("/") + "/models",
                                     headers={"Authorization":
                                              "Bearer " + os.environ.get("P2_API_KEY", "")})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except Exception as e:                                    # noqa: BLE001
        return False, f"HTTP 探测失败 — {type(e).__name__}: {e}"


# ------------------------------------------------------------------ 执行

def run_one(endpoint, model, prompt, temperature, seed, want_logprobs):
    import urllib.request
    body = {
        "model": model,
        "messages": [{"role": "system", "content": prompt["system_prompt"]},
                     {"role": "user", "content": prompt["user_prompt"]}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": 2048,
    }
    if want_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["P2_API_KEY"]},
        method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task-file",
        default=os.path.join(ROOT, "data", "scoring_records", "scored_outputs.jsonl"),
        help="User-supplied scored-output JSONL. Data are not included in this repository.",
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "four_arm_run"))
    ap.add_argument("--plan-only", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--endpoint", default=os.environ.get("P2_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--arms", default="a0,ac,ai,ap")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for a in arms:
        if a not in ARM_DEFS:
            raise SystemExit(f"未知臂：{a}（可选 {list(ARM_DEFS)}）")

    if not os.path.isfile(args.task_file):
        raise SystemExit(
            "Task file not found. Supply an external JSONL file with --task-file; "
            "research data are intentionally excluded from this release."
        )
    tasks = load_tasks(args.task_file, args.limit)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    # ---- 生成提示模板清单与运行计划（零网络请求）----
    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint, "model": args.model,
        "arms": {}, "n_tasks": len(tasks), "seeds": seeds,
        "temperature": args.temperature,
        "total_requests": len(tasks) * len(arms) * len(seeds),
        "note": "本文件仅为计划；执行结果须由 real API 返回写入，禁止人工填写。",
    }
    for a in arms:
        d = ARM_DEFS[a]
        samples = {}
        for tt in ("knowledge", "reasoning", "planning"):
            t = next((x for x in tasks if x["task_type"] == tt), None)
            if t:
                pr = build_prompt(a, t)
                samples[tt] = {"example_task_id": t["task_id"],
                               "system_prompt": pr["system_prompt"],
                               "user_prompt": pr["user_prompt"],
                               "prompt_sha16": prompt_sha(pr)}
        plan["arms"][a] = {"desc": d["desc"], "request_logprobs": d["logprobs"],
                           "n_prompts_sha": len({prompt_sha(build_prompt(a, t)) for t in tasks}),
                           "samples": samples}

    # 长度匹配前置检验的设计阈值（供执行后判定，非结果）
    plan["length_matching_criterion"] = {
        "rule": "The AC median output length should fall within the AP IQR, "
                "and the two-sided KS test should have P > 0.05.",
        "reference": "Compute the reference distribution from the supplied run; "
                     "no unpublished result is embedded in this release.",
    }
    with open(os.path.join(outdir, "run_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "prompt_manifest.jsonl"), "w", encoding="utf-8") as f:
        for a in arms:
            for t in tasks:
                pr = build_prompt(a, t)
                f.write(json.dumps({"arm": a, "task_id": t["task_id"],
                                    "task_type": t["task_type"], "level": t["level"],
                                    "prompt_sha16": prompt_sha(pr), **pr},
                                   ensure_ascii=False) + "\n")

    print(f"[plan] 任务 {len(tasks)} × 臂 {len(arms)} × 种子 {len(seeds)} "
          f"= {plan['total_requests']} 次请求")
    for a in arms:
        print(f"       {a:3s} logprobs={ARM_DEFS[a]['logprobs']!s:5s} {ARM_DEFS[a]['desc']}")
    print(f"[plan] 写出 {outdir}/run_plan.json, prompt_manifest.jsonl")

    # ---- 预检 ----
    if args.preflight or args.execute:
        ok, detail = endpoint_preflight(args.endpoint)
        key = bool(os.environ.get("P2_API_KEY"))
        print(f"[preflight] 端点 {args.endpoint} → {'可达' if ok else '不可达'}（{detail}）")
        print(f"[preflight] P2_API_KEY {'已设置' if key else '未设置'}")
        if not ok or not key:
            print("[ABORT] 前置条件未满足。**不执行任何请求，也不生成任何结果文件。**")
            print("        这符合'禁止造假'原则：缺 API 可达性时，P2 实验只能停留在设计阶段。")
            return 2

    if not args.execute:
        print("[plan-only] 未指定 --execute，结束。")
        return 0

    # ---- 真实执行（断点续跑）----
    done = set()
    for a in arms:
        fp = os.path.join(outdir, f"raw_{a}_{args.model}.jsonl")
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done.add((r["arm"], r["seed"], r["task_id"]))
                    except Exception:                              # noqa: BLE001
                        pass
    print(f"[resume] 已完成 {len(done)} 条，跳过。")

    for a in arms:
        fp = os.path.join(outdir, f"raw_{a}_{args.model}.jsonl")
        with open(fp, "a", encoding="utf-8") as fh:
            for seed in seeds:
                for t in tasks:
                    if (a, seed, t["task_id"]) in done:
                        continue
                    pr = build_prompt(a, t)
                    try:
                        payload, dt = run_one(args.endpoint, args.model, pr,
                                              args.temperature, seed,
                                              ARM_DEFS[a]["logprobs"])
                        choice = (payload.get("choices") or [{}])[0]
                        rec = {
                            "arm": a, "seed": seed, "task_id": t["task_id"],
                            "task_type": t["task_type"], "level": t["level"],
                            "model": payload.get("model") or args.model,
                            "temperature": args.temperature,
                            "prompt_sha16": prompt_sha(pr),
                            "system_prompt": pr["system_prompt"],
                            "user_prompt": pr["user_prompt"],
                            "raw_output": (choice.get("message") or {}).get("content"),
                            "logprobs": choice.get("logprobs"),
                            "finish_reason": choice.get("finish_reason"),
                            "usage": payload.get("usage"),
                            "latency_seconds": dt,
                            "requested_at": datetime.now(timezone.utc).isoformat(),
                        }
                        # 判分不在本脚本内：须由 score_experiment.py 以冻结规则执行
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                    except Exception as e:                          # noqa: BLE001
                        fh.write(json.dumps({
                            "arm": a, "seed": seed, "task_id": t["task_id"],
                            "error": f"{type(e).__name__}: {e}",
                            "requested_at": datetime.now(timezone.utc).isoformat(),
                        }, ensure_ascii=False) + "\n")
                        fh.flush()
    print(f"[done] 结果写入 {outdir}/raw_*.jsonl；判分请运行 score_experiment.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
