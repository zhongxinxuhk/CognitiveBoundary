#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 — confidence-label parsing and rule-sensitivity analysis.

Public-release scope
--------------------
This is a pre-publication reference implementation for evaluating the
sensitivity of explicit confidence extraction to alternative parsing rules.
It includes no research records, computed results, or manuscript prose.

用途
----
本脚本用于检验显式置信度抽取是否对文本解析口径敏感。

设计原则
--------
1. **规则先行（rule-first）**：三条解析规则在**数据采集完成后、分析开始前**
   即固定于本文件，运行时不提供「调参以取得期望结果」的入口。
2. **纯标准库依赖**：AUROC / Wilson 区间 / Bootstrap 全部自行实现，
   不依赖 sklearn，保证任何环境可复算。
3. **敏感性全量报告**：三条规则一律报告，**不做择优**。若某条规则能得出
   与另两条相反的结论，该事实必须保留在输出中。

输入数据契约
------------
JSON Lines，每行一条记录，须含以下字段：

    {
      "condition": "proposed" | "baseline",
      "task_info": {"task_id": str, "task_type": str},
      "model_output": {"raw_output": str},
      "scoring": {"has_ground_truth": bool, "is_correct": bool}
    }

配对口径：`task_id` 在两条件中均存在，且两条件 `has_ground_truth` 均为真。

三条规则
--------
规则 P（窄窗口同行匹配）
    正则：``置信度[^：:\\n]{0,8}[：:]\\s*\\**\\s*(高|中|低)``
    要求槽位标签与取值在**同一行**内以冒号相连，且标签后至冒号间隔 ≤ 8 字符。
    该口径**不能**覆盖模型以隐式换行或 Markdown 列表形式给出的等级。

规则 C（结构锚定口径）
    先定位槽位标签 ``置信度（高/中/低）``，再在其后 40 字符窗口内锚定取值，
    容忍换行、空白与 Markdown 星号过渡，但**仍要求槽位标签存在**。

规则 R（模板相对口径）
    在规则 C 基础上，进一步回收**无槽位标签**的列表式等级标注（形如 ``- **高**``）。
    该规则仅作为预注册候选口径之一报告。

预期口径
--------
三条规则的可解析数构成一个单调序列，全部由真实数据决定，脚本不得硬编码：

    规则 P  ⊂  规则 C  ⊂  规则 R

用法
----
    python parse_confidence_rules.py --input scored_outputs.jsonl --outdir out/

输出
----
    confidence_rule_sensitivity.json   机器可读的完整结果
    confidence_rule_sensitivity.md     人可读的敏感性表
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

# --------------------------------------------------------------------- 常量
# Fixed analysis constants for reproducible Bootstrap output.
SEED = 20260920
N_BOOT = 20000
CI_ALPHA = 0.05

# 三级离散标签 → 有序数值（用于 AUROC 的秩统计）
SCORE = {"低": 0.0, "中": 0.5, "高": 1.0}

LEVELS = ("高", "中", "低")

# ------------------------------------------------------------------ 解析规则
# 规则 P：窄窗口同行匹配，保留以供敏感性对照
LAB_P = re.compile(r"置信度[^：:\n]{0,8}[：:]\s*\**\s*(高|中|低)")

# 规则 C：槽位标签锚定，允许换行 / 空白 / Markdown 星号过渡
LAB_C = re.compile(
    r"置信度\s*[（(]?\s*高\s*[/／]\s*中\s*[/／]\s*低\s*[）)]?\s*[:：]?\s*"
)
WINDOW_C = 40

# 规则 R：在规则 C 失败时，回收行首项目符号式的等级标注
INLINE_R = re.compile(
    r"(?:^|[\s\-•*>*])(?:置信度\s*[:：]?\s*)?\**\s*(高|中|低)\s*\**"
)
WINDOW_R = 120


def conf_rule_p(text):
    """规则 P：窄窗口同行匹配。"""
    m = LAB_P.search(text or "")
    return m.group(1) if m else None


def conf_rule_c(text):
    """规则 C：槽位标签锚定 + 结构容忍。"""
    text = text or ""
    m = LAB_C.search(text)
    if not m:
        return None
    tail = text[m.end(): m.end() + WINDOW_C]
    mm = re.match(r"[\s*]*\n?[\s*]*(高|中|低)", tail)
    return mm.group(1) if mm else None


def conf_rule_r(text):
    """规则 R：规则 C 失败时回收列表式等级标注。"""
    v = conf_rule_c(text)
    if v:
        return v
    text = text or ""
    m = LAB_C.search(text)
    seg = text[m.end(): m.end() + WINDOW_R] if m else text
    hits = INLINE_R.findall(seg)
    return hits[0] if hits else None


RULES = [
    ("P", "窄窗口同行匹配", conf_rule_p),
    ("C", "结构锚定口径：槽位标签 + 换行容忍", conf_rule_c),
    ("R", "模板相对口径：回收无标签的列表式等级", conf_rule_r),
]

# ------------------------------------------------------------------ 统计工具


def wilson_ci(k, n, z=1.959963985):
    """二项比例的 Wilson 区间（比 Wald 区间在极端比例下更稳健）。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - s), min(1.0, c + s))


def auroc(labels, positive):
    """
    精确 AUROC（Mann–Whitney U 形式，含并列值 0.5 计分）。

    labels : list[float]  排序依据（此处为置信度等级分值）
    positive : list[bool] 是否为正类（此处为「答案**正确**」）

    口径说明（关键，易错）：AUROC 取正类为「正确」，
    即 AUROC = P(置信度_正确 > 置信度_错误)。
    该值 > 0.5 表示置信度具有判别力；< 0.5 表示方向反转。
    若误将正类取为「错误」，会得到 1 − AUROC 的镜像值，因此这里显式固定
    正类定义以避免方向性错误。

    返回 (auroc, n_pos, n_neg)；任一类为空时返回 None。
    """
    pos = [l for l, p in zip(labels, positive) if p]
    neg = [l for l, p in zip(labels, positive) if not p]
    if not pos or not neg:
        return None, len(pos), len(neg)
    # 秩和法：处理并列时等价于逐对比较并计 0.5
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    ranks = [0.0] * len(labels)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and labels[order[j + 1]] == labels[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(r for r, p in zip(ranks, positive) if p)
    n_pos, n_neg = len(pos), len(neg)
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg), n_pos, n_neg


def bootstrap_auroc_ci(labels, positive, n_boot=N_BOOT, seed=SEED, alpha=CI_ALPHA):
    """AUROC 的百分位 Bootstrap 区间。"""
    if len(labels) < 3:
        return (None, None)
    rng = random.Random(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a, _, _ = auroc([labels[i] for i in idx], [positive[i] for i in idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return (None, None)
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (lo, hi)


# ------------------------------------------------------------------ 数据加载


def load_pairs(path):
    """加载并按配对口径筛选，返回 [(task_id, task_type, label_text, is_correct)]。"""
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    def ok(r):
        return bool((r.get("scoring") or {}).get("has_ground_truth"))

    base = {r["task_info"]["task_id"] for r in recs
            if r.get("condition") == "baseline" and ok(r)}
    pairs = []
    for r in recs:
        if r.get("condition") != "proposed" or not ok(r):
            continue
        tid = r["task_info"]["task_id"]
        if tid not in base:
            continue
        pairs.append((
            tid,
            r["task_info"].get("task_type"),
            (r.get("model_output") or {}).get("raw_output") or "",
            bool((r.get("scoring") or {}).get("is_correct")),
        ))
    pairs.sort(key=lambda x: x[0])
    return pairs


# ------------------------------------------------------------------ 主流程


def analyse(pairs):
    results = {}
    for key, desc, fn in RULES:
        parsed, unparsed = [], []
        for tid, ttype, text, correct in pairs:
            lab = fn(text)
            (parsed if lab else unparsed).append((tid, ttype, lab, correct))

        dist = Counter(lab for _, _, lab, _ in parsed)
        n_parsed = len(parsed)
        n_error = sum(1 for _, _, _, c in parsed if not c)

        # 高置信错误率（条件概率 P(错误 | 高置信)）
        hi = [(tid, t) for tid, t, lab, c in parsed if lab == "高"]
        hi_err = sum(1 for tid, t, lab, c in parsed if lab == "高" and not c)
        n_hi = len(hi)
        hi_err_rate = (hi_err / n_hi) if n_hi else None
        hi_ci = wilson_ci(hi_err, n_hi) if n_hi else (None, None)
        # 覆盖率：自报高置信条目占全部配对样本的比例
        coverage = n_hi / len(pairs) if pairs else None

        labels = [SCORE[lab] for _, _, lab, _ in parsed if lab in SCORE]
        # 正类 = 答案正确（见 auroc() 的口径说明）
        positive = [c for _, _, lab, c in parsed if lab in SCORE]
        a, n_pos, n_neg = auroc(labels, positive)
        lo, hi_a = bootstrap_auroc_ci(labels, positive) if a is not None else (None, None)

        results[key] = {
            "rule": key,
            "description": desc,
            "n_universe": len(pairs),
            "n_parsed": n_parsed,
            "n_unparsed": len(unparsed),
            "distribution": {lv: dist.get(lv, 0) for lv in LEVELS},
            "n_correct": n_parsed - n_error,
            "n_error": n_error,
            "self_reported_high": {
                "n": n_hi,
                "coverage": coverage,
                "n_error": hi_err,
                "error_rate_given_high": hi_err_rate,
                "error_rate_ci95_wilson": list(hi_ci) if n_hi else None,
            },
            "auroc": {
                "value": a,
                "ci95": [lo, hi_a] if a is not None else None,
                "n": len(labels),
                "n_positive_correct": n_pos,
                "n_negative_error": n_neg,
                "orientation": "positive=正确；AUROC=P(置信度_正确 > 置信度_错误)",
            },
            "unparsed_task_ids": [tid for tid, _, _, _ in unparsed],
        }
    return results


def write_outputs(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    jp = os.path.join(outdir, "confidence_rule_sensitivity.json")
    mp = os.path.join(outdir, "confidence_rule_sensitivity.md")
    payload = {
        "generated_by": "CognitiveBoundary/scripts/parse_confidence_rules.py",
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "rules": results,
    }
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# 置信度解析规则敏感性（补充材料 S-Conf-Parsing）",
        "",
        "三条解析规则在数据采集完成后即固定，未依结果调整。全部结果一律报告，不做择优。",
        "",
        "| 规则 | 说明 | 可解析 / 配对 | 高 | 中 | 低 | 自报高置信错误率 | AUROC [95% CI] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, _, _ in RULES:
        r = results[key]
        d = r["distribution"]
        srh = r["self_reported_high"]
        au = r["auroc"]
        er = srh["error_rate_given_high"]
        er_s = "—" if er is None else f"{er * 100:.2f}%"
        if au["value"] is None:
            au_s = "—"
        elif au["ci95"] and au["ci95"][0] is not None:
            au_s = f"{au['value']:.3f} [{au['ci95'][0]:.3f}, {au['ci95'][1]:.3f}]"
        else:
            au_s = f"{au['value']:.3f}"
        lines.append(
            f"| {key} | {r['description']} | {r['n_parsed']} / {r['n_universe']} "
            f"| {d['高']} | {d['中']} | {d['低']} | {er_s} | {au_s} |"
        )

    lines += [
        "",
        "## 不变量与差异（逐规则如实报告）",
        "",
    ]
    for key, _, _ in RULES:
        r = results[key]
        au = r["auroc"]
        ci = au["ci95"]
        if ci and ci[0] is not None:
            covers = ci[0] <= 0.5 <= ci[1]
            ci_s = (f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]，"
                    f"{'覆盖' if covers else '**不覆盖**'}随机水平 0.5")
        else:
            ci_s = "CI 不可计算"
        lines.append(
            f"- **规则 {key}**：可解析 {r['n_parsed']}/{r['n_universe']}；"
            f"「低」置信自报 {r['distribution']['低']} 条；"
            f"自报高置信错误率 "
            + ("—" if r['self_reported_high']['error_rate_given_high'] is None
               else f"{r['self_reported_high']['error_rate_given_high'] * 100:.2f}%")
            + f"；AUROC = {au['value']:.3f}，{ci_s}。"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Treat all rule-specific estimates as a sensitivity analysis.",
        "- Do not select a parsing rule after inspecting which result is most favorable.",
        "- Scientific claims, if any, require a separately approved analysis protocol and",
        "  are intentionally not generated by this public script.",
        "",
    ]
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return jp, mp


def main():
    ap = argparse.ArgumentParser(description="置信度解析规则敏感性分析")
    ap.add_argument("--input", required=True, help="scored_outputs.jsonl 路径")
    ap.add_argument("--outdir", required=True, help="输出目录")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ABORT] 输入不存在：{args.input}", file=sys.stderr)
        return 2

    pairs = load_pairs(args.input)
    if not pairs:
        print("[ABORT] 配对样本为空，检查输入与配对口径。", file=sys.stderr)
        return 2

    results = analyse(pairs)
    jp, mp = write_outputs(results, args.outdir)

    print(f"配对样本 n = {len(pairs)}")
    print()
    for key, _, _ in RULES:
        r = results[key]
        d = r["distribution"]
        au = r["auroc"]
        print(f"规则 {key}：可解析 {r['n_parsed']}/{r['n_universe']}  "
              f"高={d['高']} 中={d['中']} 低={d['低']}  "
              f"AUROC={au['value']:.3f} (n={au['n']})")
    print()
    print(f"写出 {jp}")
    print(f"写出 {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
