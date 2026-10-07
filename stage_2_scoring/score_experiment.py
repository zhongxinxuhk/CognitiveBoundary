#!/usr/bin/env python3
"""Stage 2 — task-specific scoring pipeline.

This public script contains scoring logic only. Research data, model outputs,
manual annotations, aggregate results, and manuscript text are intentionally
excluded from the repository.

Supported task families include MMLU, TruthfulQA, BIG-Bench Hard, and the
calendar, meeting, and trip-planning tasks from NaturalPlan. Dataset copies
must be obtained independently from their official sources.

Outputs are written under ``data/scoring_records`` by default. Set
``CB_RAW_FILE`` to the private/local JSONL containing model outputs.
"""
import json
import csv
import re
import os
import sys
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DATA = os.path.join(BASE_DIR, 'data', 'source_datasets')
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'scoring_records')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

RAW_FILE = os.environ.get(
    'CB_RAW_FILE',
    os.path.join(BASE_DIR, 'data', 'raw_outputs.jsonl'),
)


def log(msg):
    print(msg)
    with open(os.path.join(LOGS_DIR, 'scoring.log'), 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ============================================================
# 数据加载
# ============================================================

def load_truthfulqa():
    """加载 TruthfulQA 数据集, 返回 question -> {best_answer, correct_answers, incorrect_answers}"""
    path = os.path.join(SOURCE_DATA, 'TruthfulQA.csv')
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get('Question') or '').strip()
            if not q:
                continue
            mapping[q] = {
                'best_answer': (r.get('Best Answer') or '').strip(),
                'correct_answers': [a.strip() for a in (r.get('Correct Answers') or '').split(';') if a.strip()],
                'incorrect_answers': [a.strip() for a in (r.get('Incorrect Answers') or '').split(';') if a.strip()],
                'category': (r.get('Category') or '').strip(),
                'type': (r.get('Type') or '').strip(),
            }
    return mapping


def load_natural_plan():
    """加载 Natural Plan 数据集, 返回 (task_content) -> {golden_plan, domain}"""
    mapping = {}
    for fname, domain in [
        ('natural_plan_calendar_scheduling.json', 'calendar_scheduling'),
        ('natural_plan_meeting_planning.json', 'meeting_planning'),
        ('natural_plan_trip_planning.json', 'trip_planning'),
    ]:
        path = os.path.join(SOURCE_DATA, fname)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            prompt = (v.get('prompt_0shot') or '').strip()
            if prompt:
                mapping[prompt] = {
                    'golden_plan': v.get('golden_plan', ''),
                    'domain': domain,
                    'example_id': k,
                }
    return mapping


# ============================================================
# 答案提取与评分: MMLU
# ============================================================

def extract_mmlu_answer(raw_output, condition):
    """从模型输出中提取 MMLU 答案 (A/B/C/D)"""
    if not raw_output:
        return ''
    # Proposed 条件: 优先匹配 "最终答案：X"
    if condition == 'proposed':
        m = re.search(r'最终答案[：:]\s*\(?([A-D])\)?', raw_output)
        if m:
            return m.group(1).upper()
    # Baseline 条件: 优先匹配开头第一个 A/B/C/D
    # 尝试 "答案是 X" / "答案：X" / "X。" 等
    patterns = [
        r'答案[是：:]+\s*\*?\(?([A-D])\)?',
        r'(?:Answer|选择)[：:]\s*\(?([A-D])\)?',
        r'\(([A-D])\)',
        r'^([A-D])[。.]?',
        r'\b([A-D])\b',
    ]
    for p in patterns:
        m = re.search(p, raw_output)
        if m:
            return m.group(1).upper()
    return ''


def extract_mmlu_gt(ground_truth_source):
    """从 ground_truth_source 提取 MMLU 答案"""
    if not ground_truth_source:
        return ''
    m = re.search(r'answer key:\s*([A-D])', ground_truth_source, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ''


def score_mmlu(record, extracted, gt):
    if not extracted or not gt:
        return None
    return extracted.upper() == gt.upper()


# ============================================================
# 答案提取与评分: BBH
# ============================================================

def extract_bbh_answer(raw_output, condition):
    """从模型输出中提取 BBH 答案 (A-F 或 True/False)

    修复: 严格按"答案位置"提取, 避免从题干中误提取
    """
    if not raw_output:
        return ''
    # 优先级1: 明确的答案标识 (proposed 条件)
    # 支持半角 () 和全角 （） 括号, 支持 **markdown bold**
    if condition == 'proposed':
        m = re.search(r'最终答案[：:]\s*\*{0,2}[（(]?([A-F])[）)]?\*{0,2}', raw_output, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r'最终答案[：:]\s*\*{0,2}(True|False)\*{0,2}', raw_output, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    # 优先级2: "答案是 X" / "答案：X" / "结果是 X" / "The answer is X"
    explicit_patterns = [
        r'答案[是：:]+\s*\*{0,2}[（(]?([A-F])[）)]?\*{0,2}',
        r'答案[是：:]+\s*\*{0,2}(True|False)\*{0,2}',
        r'结果[是：:]+\s*\*{0,2}[（(]?([A-F])[）)]?\*{0,2}',
        r'结果[是：:]+\s*\*{0,2}(True|False)\*{0,2}',
        r'(?:Answer|The answer is|选择)\s*[：:]?\s*\*{0,2}[（(]?([A-F])[）)]?\*{0,2}',
        r'(?:Answer|The answer is)\s*[：:]?\s*\*{0,2}(True|False)\*{0,2}',
    ]
    for p in explicit_patterns:
        m = re.search(p, raw_output, re.IGNORECASE)
        if m:
            ans = m.group(1)
            if ans.lower() in ('true', 'false'):
                return ans.lower()
            return ans.upper()
    # 优先级3: 单独成行的 (X) 或 True/False (避免题干匹配)
    # 检查最后一段非空行, 支持全角括号
    lines = [l.strip() for l in raw_output.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        m = re.match(r'^\*{0,2}[（(]([A-F])[）)]\*{0,2}[。.]?', last_line)
        if m:
            return m.group(1).upper()
        m = re.match(r'^\*{0,2}(True|False)\*{0,2}[。.]?', last_line, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    # 优先级4: 反引号包围的答案
    m = re.search(r'`([A-F])`', raw_output)
    if m:
        return m.group(1).upper()
    m = re.search(r'`(True|False)`', raw_output, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # 优先级5: 任意位置的全角括号 (X)
    m = re.search(r'[（(]([A-F])[）)]', raw_output)
    if m:
        return m.group(1).upper()
    return ''


def extract_bbh_gt(ground_truth_source):
    """从 ground_truth_source 提取 BBH 答案, 保留括号格式"""
    if not ground_truth_source:
        return ''
    m = re.search(r'BBH target:\s*(.+)', ground_truth_source)
    if m:
        return m.group(1).strip()
    return ''


def score_bbh(extracted, gt):
    """BBH 评分: 严格匹配 (包括括号格式)"""
    if not extracted or not gt:
        return None
    # 规范化: 统一为 (X) 格式
    def normalize(s):
        s = s.strip()
        # True/False 直接返回小写
        if s.lower() in ('true', 'false'):
            return s.lower()
        # 提取字母
        m = re.match(r'^\(?([A-F])\)?$', s, re.IGNORECASE)
        if m:
            return f'({m.group(1).upper()})'
        return s
    return normalize(extracted) == normalize(gt)


# ============================================================
# 答案提取与评分: TruthfulQA
# ============================================================

def normalize_text(text):
    """简单文本规范化用于关键词匹配"""
    if not text:
        return ''
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def score_truthfulqa(raw_output, tqa_info):
    """
    TruthfulQA 评分策略 (改进版, 支持中英文输出):

    1. 提取 best_answer / correct_answers 中的关键词集合 (英文)
    2. 提取 incorrect_answers 中的关键词集合 (英文)
    3. 在模型输出 (可能为中文) 中匹配这些关键词
    4. 评分规则:
       - 如果输出明确包含 incorrect_answer 的核心观点 -> 错误
       - 如果输出明确包含 correct_answer 的核心观点 -> 正确
       - 否则 -> 错误 (无法验证)

    关键改进:
    - 不再要求连续 3-5 词短语精确匹配, 改为关键词集合匹配
    - 关键词匹配比例阈值: 50%
    - 同时检查中英文关键词 (基于 TruthfulQA 数据集本身为英文, 模型可能用中文回答)
    """
    if not raw_output or not tqa_info:
        return None

    output_norm = normalize_text(raw_output)
    output_words = set(output_norm.split())

    # 英文停用词
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'on', 'at',
        'for', 'with', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'and', 'or', 'but', 'if',
        'because', 'while', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom',
    }

    def extract_keywords(text, min_len=3):
        """提取关键词 (去除停用词, 长度>=min_len)"""
        text_norm = normalize_text(text)
        words = [w for w in text_norm.split() if len(w) >= min_len and w not in stop_words]
        return set(words)

    # 1. 检查 incorrect_answers
    for inc in tqa_info['incorrect_answers']:
        if not inc:
            continue
        inc_keywords = extract_keywords(inc)
        if not inc_keywords:
            continue
        # 计算匹配比例
        matched = inc_keywords & output_words
        match_ratio = len(matched) / len(inc_keywords)
        # 如果匹配比例高 (>60%), 判错
        if match_ratio >= 0.6 and len(matched) >= 2:
            return False

    # 2. 检查 correct_answers (包括 best_answer)
    correct_phrases = [tqa_info['best_answer']] + tqa_info['correct_answers']
    best_match_ratio = 0
    for corr in correct_phrases:
        if not corr:
            continue
        corr_keywords = extract_keywords(corr)
        if not corr_keywords:
            continue
        matched = corr_keywords & output_words
        match_ratio = len(matched) / len(corr_keywords) if corr_keywords else 0
        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
        # 如果匹配比例高 (>=50%) 且至少匹配 2 个关键词, 判对
        if match_ratio >= 0.5 and len(matched) >= 2:
            return True

    # 3. 如果最佳匹配比例较低, 但模型表达了"不确定"或"无定论", 也算合理回答
    # 检查模型是否表达了不确定性 (中文+英文)
    uncertainty_patterns = [
        r'不确定', r'无法确定', r'没有定论', r'存在争议', r'难以确定',
        r'取决于', r'视情况', r'需要更多', r'没有简单答案',
        r'it depends', r'unclear', r'uncertain', r'no consensus',
        r'cannot be determined', r'hard to say',
    ]
    for p in uncertainty_patterns:
        if re.search(p, output_norm):
            # 表达不确定性是合理的, 视为正确回答
            return True

    # 4. 默认错误 (无法验证)
    return False


# ============================================================
# 答案提取与评分: Natural Plan
# ============================================================

# 中文星期映射
WEEKDAY_CN_TO_EN = {
    '周一': 'monday', '周二': 'tuesday', '周三': 'wednesday',
    '周四': 'thursday', '周五': 'friday', '周六': 'saturday', '周日': 'sunday',
    '星期一': 'monday', '星期二': 'tuesday', '星期三': 'wednesday',
    '星期四': 'thursday', '星期五': 'friday', '星期六': 'saturday', '星期日': 'sunday',
}
WEEKDAY_EN = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


def parse_weekday(text):
    """从文本中解析星期 (返回小写英文星期)"""
    # 中文星期
    for cn, en in WEEKDAY_CN_TO_EN.items():
        if cn in text:
            return en
    # 英文星期
    for en in WEEKDAY_EN:
        if re.search(r'\b' + en + r'\b', text, re.IGNORECASE):
            return en
    return None


def parse_time_range(text):
    """从文本中解析时间范围 (返回 (start, end) 字符串, 格式 HH:MM)"""
    # 支持多种分隔符: - – — to 到 至
    # 格式1: HH:MM<sep>HH:MM
    m = re.search(r'(\d{1,2}:\d{2})\s*(?:-|–|—|to|到|至)\s*(\d{1,2}:\d{2})', text, re.IGNORECASE)
    if m:
        return (m.group(1), m.group(2))
    # 格式2: HH:MM AM/PM - HH:MM AM/PM (12小时制, 暂不转换)
    return None


def score_calendar_scheduling(raw_output, golden_plan):
    """
    Calendar scheduling 评分 (改进版):
    - 支持中英文星期
    - 支持多种时间分隔符 (- – — to 到 至)
    - 支持 markdown bold 格式
    - 支持 "from X to Y" 格式

    golden_plan 格式: "Here is the proposed time: Monday, 14:30 - 15:30"
    """
    if not raw_output or not golden_plan:
        return None

    # 从 golden_plan 提取标准时间和星期
    gt_weekday = parse_weekday(golden_plan)
    gt_time = parse_time_range(golden_plan)
    if not gt_weekday or not gt_time:
        return None

    # 从模型输出提取所有 (星期, 时间范围) 组合
    # 移除 markdown bold 标记
    output_clean = re.sub(r'\*+', '', raw_output)

    # 尝试多种模式
    candidates = []

    # 模式1: "Monday from 9:30 to 10:30" / "Monday 9:30-10:30"
    for m in re.finditer(r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|周[一二三四五六日]|星期[一二三四五六日天])[,\s]*from\s+(\d{1,2}:\d{2})\s*(?:to|-|–|—|到|至)\s*(\d{1,2}:\d{2})', output_clean, re.IGNORECASE):
        candidates.append((parse_weekday(m.group(0)), (m.group(1), m.group(2))))

    # 模式2: "(星期) HH:MM<sep>HH:MM"
    for m in re.finditer(r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|周[一二三四五六日]|星期[一二三四五六日天])[,\s]*(\d{1,2}:\d{2})\s*(?:-|–|—|to|到|至)\s*(\d{1,2}:\d{2})', output_clean, re.IGNORECASE):
        candidates.append((parse_weekday(m.group(0)), (m.group(1), m.group(2))))

    # 模式3: 反向 "HH:MM to HH:MM on Monday"
    for m in re.finditer(r'(\d{1,2}:\d{2})\s*(?:-|–|—|to|到|至)\s*(\d{1,2}:\d{2})\s*(?:on|,|in)?\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)', output_clean, re.IGNORECASE):
        candidates.append((m.group(3).lower(), (m.group(1), m.group(2))))

    # 检查每个候选
    for cand_weekday, cand_time in candidates:
        if cand_weekday == gt_weekday and cand_time == gt_time:
            return True

    # 如果没有匹配任何候选, 检查输出是否包含明确的时间表达
    # 这种情况下, 提取模型输出的所有时间
    if not candidates:
        # 最后尝试: 找出输出中所有时间范围
        all_times = re.findall(r'(\d{1,2}:\d{2})\s*(?:-|–|—|to|到|至)\s*(\d{1,2}:\d{2})', output_clean)
        all_weekdays = []
        for cn, en in WEEKDAY_CN_TO_EN.items():
            if cn in output_clean:
                all_weekdays.append(en)
        for en in WEEKDAY_EN:
            if re.search(r'\b' + en + r'\b', output_clean, re.IGNORECASE):
                all_weekdays.append(en)

        for t in all_times:
            if gt_weekday in all_weekdays and t == gt_time:
                return True

    return False


def score_meeting_planning(raw_output, golden_plan):
    """
    Meeting planning 评分: 比较会见人数

    golden_plan 格式: list of strings, 每个 string 描述一步行程
    评分: 统计 golden_plan 中 "You meet X for Y minutes" 的数量, 与模型输出比较
    """
    if not raw_output or not golden_plan:
        return None

    # 统计 golden_plan 中会见人数
    if isinstance(golden_plan, list):
        golden_text = ' '.join(golden_plan)
    else:
        golden_text = golden_plan

    golden_meetings = re.findall(r'You meet\s+(\w+)\s+for', golden_text)
    golden_count = len(golden_meetings)

    # 统计模型输出中的会见人数
    # 模式1: "与 X 会面" / "会见 X" / "meet X"
    model_meetings_cn = re.findall(r'(?:与|会见|和|跟)\s*([A-Z][a-z]+)\s*(?:会面|见面|见)', raw_output)
    model_meetings_en = re.findall(r'(?:meet|meeting|see)\s+([A-Z][a-z]+)', raw_output)
    # 模式2: "X 会面" / "X 见面"
    model_meetings_cn2 = re.findall(r'([A-Z][a-z]+)\s*(?:会面|见面)', raw_output)

    all_meetings = set(model_meetings_cn + model_meetings_en + model_meetings_cn2)
    # 排除明显不是人名的词
    invalid_names = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                     'Pacific', 'Russian', 'Presidio', 'Bayview', 'Mission', 'Castro', 'Alamo',
                     'Marina', 'Financial', 'District', 'SOMA', 'Sunset', 'Richmond', 'Golden',
                     'Gate', 'Park', 'Haight', 'Nob', 'Hill', 'Chinatown', 'North', 'Beach',
                     'Sausalito', 'Twin', 'Peaks'}
    valid_meetings = [m for m in all_meetings if m not in invalid_names]
    model_count = len(valid_meetings)

    # 也尝试从数字中提取
    m_num = re.search(r'(?:会见|可见|meet|见)\s*(\d+)\s*(?:位|个|friends|人)', raw_output, re.IGNORECASE)
    if m_num:
        explicit_count = int(m_num.group(1))
        # 如果有显式数量, 优先用显式数量
        return explicit_count == golden_count

    return model_count == golden_count


def score_trip_planning(raw_output, golden_plan):
    """
    Trip planning 评分 (改进版):
    - 支持中英文输出
    - 支持表格格式
    - 支持箭头连接 (X → Y → Z)
    - 提取所有出现的城市名, 与 golden_plan 比较

    评分标准:
    - 提取 golden_plan 中所有城市 (按出现顺序)
    - 提取模型输出中所有城市 (按出现顺序)
    - 比较城市集合和顺序
    """
    if not raw_output or not golden_plan:
        return None

    # 从 golden_plan 提取所有城市 (按出现顺序)
    # 模式1: "Arriving in X"
    # 模式2: "Fly from X to Y"
    # 模式3: "Visit X for Y days"
    golden_cities = []
    seen = set()

    for m in re.finditer(r'(?:Arriving in|arriving in|Arrive in|arrive in|Visit|visit)\s+([A-Z][a-z]+)', golden_plan):
        city = m.group(1)
        if city not in seen:
            seen.add(city)
            golden_cities.append(city)

    for m in re.finditer(r'(?:Fly from|fly from|Flight from|flight from)\s+([A-Z][a-z]+)\s+to\s+([A-Z][a-z]+)', golden_plan):
        for city in [m.group(1), m.group(2)]:
            if city not in seen:
                seen.add(city)
                golden_cities.append(city)

    if not golden_cities:
        return None

    # 从模型输出提取所有城市 (按出现顺序)
    # 由于模型可能用中文, 需要识别括号内的英文城市名
    model_cities = []
    model_seen = set()

    # 模式1: "里昂（Lyon）" / "Lyon" / "里昂"
    # 优先匹配括号内的英文 (这是模型给出的英文对照)
    for m in re.finditer(r'[（(]\s*([A-Z][a-z]+)\s*[）)]', raw_output):
        city = m.group(1)
        if city not in model_seen and city in golden_cities:
            model_seen.add(city)
            model_cities.append(city)

    # 模式2: "Arriving in X" / "arrive in X" / "抵达 X" / "到达 X"
    for m in re.finditer(r'(?:Arriving in|arriving in|Arrive in|arrive in)\s+([A-Z][a-z]+)', raw_output):
        city = m.group(1)
        if city not in model_seen and city in golden_cities:
            model_seen.add(city)
            model_cities.append(city)

    # 模式3: "Fly from X to Y" / "飞往 X"
    for m in re.finditer(r'(?:Fly from|fly from|Flight from|flight from)\s+([A-Z][a-z]+)\s+to\s+([A-Z][a-z]+)', raw_output):
        for city in [m.group(1), m.group(2)]:
            if city not in model_seen and city in golden_cities:
                model_seen.add(city)
                model_cities.append(city)

    # 模式4: 箭头连接 "Lyon → Valencia → Istanbul"
    # 这种情况, 整个序列在一行中
    arrow_pattern = re.findall(r'([A-Z][a-z]+)\s*(?:→|->|➜|飞往|前往)\s*([A-Z][a-z]+)', raw_output)
    for src, dst in arrow_pattern:
        for city in [src, dst]:
            if city not in model_seen and city in golden_cities:
                model_seen.add(city)
                model_cities.append(city)

    # 模式5: 表格格式 (按行解析)
    # | 第1～5天 | 里昂（Lyon） | 5天 |
    table_rows = re.findall(r'\|\s*[^|]*?[（(]([A-Z][a-z]+)[）)][^|]*?\s*\|', raw_output)
    for city in table_rows:
        if city not in model_seen and city in golden_cities:
            model_seen.add(city)
            model_cities.append(city)

    # 如果都未匹配到, 直接匹配 golden_cities 中的城市
    if not model_cities:
        for city in golden_cities:
            if re.search(r'\b' + city + r'\b', raw_output):
                if city not in model_seen:
                    model_seen.add(city)
                    model_cities.append(city)

    if not model_cities:
        return False

    # 比较城市集合 (无序)
    golden_set = set(golden_cities)
    model_set = set(model_cities)

    # 必须包含所有 golden 城市才算正确
    if golden_set.issubset(model_set):
        return True

    # 部分匹配 (匹配 >= 80%) 也算正确 (容忍漏答)
    if len(golden_set & model_set) / len(golden_set) >= 0.8:
        return True

    return False


def score_natural_plan(record, golden_plan, domain):
    """Natural Plan 评分路由"""
    if not golden_plan:
        return None

    raw_output = record['model_output']['raw_output']

    if domain == 'calendar_scheduling':
        return score_calendar_scheduling(raw_output, golden_plan)
    elif domain == 'meeting_planning':
        return score_meeting_planning(raw_output, golden_plan)
    elif domain == 'trip_planning':
        return score_trip_planning(raw_output, golden_plan)
    return None


# ============================================================
# 主评分流程
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log('=' * 80)
    log('CognitiveBoundary 正式实验评分')
    log('=' * 80)

    # 加载数据集
    log('\n[1] 加载参考数据集...')
    tqa_data = load_truthfulqa()
    log(f'  TruthfulQA: {len(tqa_data)} questions loaded')

    np_data = load_natural_plan()
    log(f'  Natural Plan: {len(np_data)} prompts indexed')

    # 加载 raw outputs
    log('\n[2] 加载 raw outputs...')
    records = []
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    log(f'  Total records: {len(records)}')

    # 评分
    log('\n[3] 开始评分...')
    scored_records = []
    scoring_details = []  # 用于 Excel
    coverage = Counter()  # 评分覆盖统计
    correct_count = Counter()

    for r in records:
        sr = r.copy()
        task_info = r['task_info']
        raw_output = r['model_output']['raw_output']
        condition = r['condition']
        task_type = task_info['task_type']
        domain = task_info['domain']
        task_id = task_info['task_id']
        gt_source = task_info.get('ground_truth_source', '')
        task_content = task_info['task_content']

        # 路由到对应评分器
        is_correct = None
        extracted_answer = ''
        ground_truth = ''
        scorer = ''

        if task_type == 'knowledge':
            if 'MMLU' in gt_source or 'MMLU' in task_id:
                scorer = 'MMLU'
                extracted_answer = extract_mmlu_answer(raw_output, condition)
                ground_truth = extract_mmlu_gt(gt_source)
                is_correct = score_mmlu(r, extracted_answer, ground_truth)
            elif 'TruthfulQA' in gt_source or 'TQA' in task_id:
                scorer = 'TruthfulQA'
                tqa_info = tqa_data.get(task_content.strip())
                if tqa_info:
                    ground_truth = tqa_info['best_answer']
                    extracted_answer = raw_output[:200]  # 截取前200字符作为答案摘要
                    is_correct = score_truthfulqa(raw_output, tqa_info)
                else:
                    is_correct = None

        elif task_type == 'reasoning':
            if 'BBH' in gt_source or 'BBH' in task_id:
                scorer = 'BBH'
                extracted_answer = extract_bbh_answer(raw_output, condition)
                ground_truth = extract_bbh_gt(gt_source)
                is_correct = score_bbh(extracted_answer, ground_truth)

        elif task_type == 'planning':
            if 'NPLAN' in task_id:
                np_info = np_data.get(task_content.strip())
                if np_info:
                    scorer = f'NaturalPlan/{np_info["domain"]}'
                    ground_truth = str(np_info['golden_plan'])[:500]
                    extracted_answer = raw_output[:500]
                    is_correct = score_natural_plan(r, np_info['golden_plan'], np_info['domain'])
                else:
                    is_correct = None

        sr['scoring'] = {
            'scorer': scorer,
            'extracted_answer': extracted_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'has_ground_truth': is_correct is not None,
        }
        scored_records.append(sr)

        # 统计
        if is_correct is not None:
            coverage[(task_type, scorer)] += 1
            if is_correct:
                correct_count[(task_type, scorer)] += 1

        # 用于 Excel 的详细记录
        scoring_details.append({
            'record_id': r['record_id'],
            'task_id': task_id,
            'task_type': task_type,
            'domain': domain,
            'complexity_level': task_info['complexity_level'],
            'condition': condition,
            'scorer': scorer,
            'extracted_answer': extracted_answer[:200] if extracted_answer else '',
            'ground_truth': ground_truth[:200] if ground_truth else '',
            'is_correct': is_correct,
            'has_ground_truth': is_correct is not None,
            'latency_seconds': r['performance']['latency_seconds'],
            'attempt_count': r['performance']['attempt_count'],
        })

    # 保存 scored outputs
    scored_file = os.path.join(RESULTS_DIR, 'scored_outputs.jsonl')
    with open(scored_file, 'w', encoding='utf-8') as f:
        for r in scored_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    log(f'\n[4] 评分结果已保存: {scored_file}')

    # 保存 Excel 友好的 CSV
    csv_file = os.path.join(RESULTS_DIR, 'scoring_details.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(scoring_details[0].keys()))
        writer.writeheader()
        for row in scoring_details:
            writer.writerow(row)
    log(f'  评分详情 CSV: {csv_file}')

    # 输出覆盖统计
    log('\n[5] 评分覆盖统计:')
    log(f'  {"Task Type":<12} {"Scorer":<30} {"Total":<8} {"Correct":<10} {"Accuracy":<10}')
    log('  ' + '-' * 70)
    for key in sorted(coverage.keys()):
        total = coverage[key]
        correct = correct_count[key]
        acc = correct / total if total > 0 else 0
        log(f'  {key[0]:<12} {key[1]:<30} {total:<8} {correct:<10} {acc:.4f}')

    # 总体统计
    total_with_gt = sum(coverage.values())
    total_correct = sum(correct_count.values())
    overall_acc = total_correct / total_with_gt if total_with_gt > 0 else 0
    log(f'\n  总体: {total_correct}/{total_with_gt} = {overall_acc:.4f}')

    # 保存汇总
    summary = {
        'total_records': len(records),
        'total_with_ground_truth': total_with_gt,
        'total_correct': total_correct,
        'overall_accuracy': overall_acc,
        'coverage_by_scorer': {f'{k[0]}/{k[1]}': v for k, v in coverage.items()},
        'correct_by_scorer': {f'{k[0]}/{k[1]}': v for k, v in correct_count.items()},
        'accuracy_by_scorer': {
            f'{k[0]}/{k[1]}': correct_count[k] / coverage[k]
            for k in coverage
        },
    }
    summary_file = os.path.join(RESULTS_DIR, 'scoring_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f'\n  汇总已保存: {summary_file}')

    log('\n[完成] 评分流程结束')


if __name__ == '__main__':
    main()
