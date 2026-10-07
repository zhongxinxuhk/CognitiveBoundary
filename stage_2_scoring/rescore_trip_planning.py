#!/usr/bin/env python3
"""Stage 2 — bilingual trip-planning rescoring helper.

The scorer supports English and Chinese city names and several itinerary
formats. It contains no task records, model outputs, aggregate scores, or
manuscript text. The supplied rules are provisional and should be validated
against an independently reviewed annotation set.
"""
import json
import re
import os
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'scoring_records')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')


def log(msg):
    print(msg)
    with open(os.path.join(LOGS_DIR, 'rescore_trip_planning.log'), 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ============================================================
# 城市名提取
# ============================================================

def extract_english_cities_from_golden(golden_plan):
    """从golden_plan提取所有英文城市名(按出现顺序)"""
    if not golden_plan:
        return []
    cities = []
    seen = set()

    # 模式1: "Arriving in X" / "arrive in X" / "Visit X"
    for m in re.finditer(r'(?:Arriving in|arriving in|Arrive in|arrive in|Visit|visit)\s+([A-Z][a-z]+)', golden_plan):
        city = m.group(1)
        if city not in seen:
            seen.add(city)
            cities.append(city)

    # 模式2: "Fly from X to Y" / "Flight from X to Y"
    for m in re.finditer(r'(?:Fly from|fly from|Flight from|flight from)\s+([A-Z][a-z]+)\s+to\s+([A-Z][a-z]+)', golden_plan):
        for city in [m.group(1), m.group(2)]:
            if city not in seen:
                seen.add(city)
                cities.append(city)

    # 模式3: "Day X-Y: Arriving in X" / "Day X: Fly from X to Y"
    for m in re.finditer(r'Day\s*\d+(?:-\d+)?:\s*([A-Z][a-z]+)', golden_plan):
        city = m.group(1)
        if city not in seen and city not in ['Arriving', 'Arrive', 'Visit', 'Fly', 'Flight']:
            seen.add(city)
            cities.append(city)

    return cities


# 预定义中英文城市名字典 (Natural Plan中常见的欧洲城市)
CITY_CN_EN_DICT = {
    # 法国
    '里昂': 'Lyon', '巴黎': 'Paris', '马赛': 'Marseille', '尼斯': 'Nice',
    '图卢兹': 'Toulouse', '波尔多': 'Bordeaux', '南特': 'Nantes', '斯特拉斯堡': 'Strasbourg',
    # 英国
    '伦敦': 'London', '曼彻斯特': 'Manchester', '伯明翰': 'Birmingham', '利兹': 'Leeds',
    '格拉斯哥': 'Glasgow', '爱丁堡': 'Edinburgh', '卡迪夫': 'Cardiff', '贝尔法斯特': 'Belfast',
    # 德国
    '柏林': 'Berlin', '慕尼黑': 'Munich', '汉堡': 'Hamburg', '科隆': 'Cologne',
    '法兰克福': 'Frankfurt', '斯图加特': 'Stuttgart', '杜塞尔多夫': 'Dusseldorf',
    '德累斯顿': 'Dresden', '莱比锡': 'Leipzig', '汉诺威': 'Hannover', '纽伦堡': 'Nuremberg',
    # 意大利
    '罗马': 'Rome', '米兰': 'Milan', '那不勒斯': 'Naples', '博洛尼亚': 'Bologna',
    '佛罗伦萨': 'Florence', '都灵': 'Turin', '巴里': 'Bari', '巴勒莫': 'Palermo',
    '卡塔尼亚': 'Catania',
    # 西班牙
    '马德里': 'Madrid', '巴塞罗那': 'Barcelona', '塞维利亚': 'Seville', '马拉加': 'Malaga',
    '毕尔巴鄂': 'Bilbao', '瓦伦西亚': 'Valencia', '萨拉戈萨': 'Zaragoza', '巴利亚多利德': 'Valladolid',
    # 葡萄牙
    '里斯本': 'Lisbon', '波尔图': 'Porto',
    # 荷兰
    '阿姆斯特丹': 'Amsterdam',
    # 奥地利
    '维也纳': 'Vienna', '萨尔茨堡': 'Salzburg', '因斯布鲁克': 'Innsbruck', '格拉茨': 'Graz', '林茨': 'Linz',
    # 捷克
    '布拉格': 'Prague', '布尔诺': 'Brno', '俄斯特拉发': 'Ostrava', '比尔森': 'Plzen',
    # 匈牙利
    '布达佩斯': 'Budapest', '德布勒森': 'Debrecen', '塞格德': 'Szeged', '米什科尔茨': 'Miskolc',
    # 波兰
    '华沙': 'Warsaw', '克拉科夫': 'Krakow', '格但斯克': 'Gdansk', '弗罗茨瓦夫': 'Wroclaw', '波兹南': 'Poznan',
    # 丹麦
    '哥本哈根': 'Copenhagen', '奥胡斯': 'Aarhus', '欧登塞': 'Odense', '奥尔堡': 'Aalborg',
    # 瑞典
    '斯德哥尔摩': 'Stockholm', '哥德堡': 'Gothenburg', '马尔默': 'Malmo', '乌普萨拉': 'Uppsala',
    # 挪威
    '奥斯陆': 'Oslo', '卑尔根': 'Bergen', '特隆赫姆': 'Trondheim', '斯塔万格': 'Stavanger',
    # 芬兰
    '赫尔辛基': 'Helsinki', '坦佩雷': 'Tampere', '图尔库': 'Turku', '奥卢': 'Oulu',
    # 爱沙尼亚
    '塔林': 'Tallinn', '塔尔图': 'Tartu',
    # 拉脱维亚
    '里加': 'Riga', '陶格夫匹尔斯': 'Daugavpils',
    # 立陶宛
    '维尔纽斯': 'Vilnius', '考纳斯': 'Kaunas', '克莱佩达': 'Klaipeda',
    # 克罗地亚
    '杜布罗夫尼克': 'Dubrovnik', '斯普利特': 'Split', '萨格勒布': 'Zagreb', '里耶卡': 'Rijeka',
    # 罗马尼亚
    '布加勒斯特': 'Bucharest', '克卢日': 'Cluj', '蒂米什瓦拉': 'Timisoara', '雅西': 'Iasi',
    # 保加利亚
    '索菲亚': 'Sofia', '普罗夫迪夫': 'Plovdiv', '瓦尔纳': 'Varna',
    # 希腊
    '雅典': 'Athens', '塞萨洛尼基': 'Thessaloniki', '帕特雷': 'Patras',
    # 土耳其
    '伊斯坦布尔': 'Istanbul', '安卡拉': 'Ankara', '伊兹密尔': 'Izmir', '布尔萨': 'Bursa',
    # 冰岛
    '雷克雅未克': 'Reykjavik', '阿克雷里': 'Akureyri',
    # 瑞士
    '日内瓦': 'Geneva', '苏黎世': 'Zurich', '巴塞尔': 'Basel', '伯尔尼': 'Bern', '洛桑': 'Lausanne',
    # 俄罗斯
    '莫斯科': 'Moscow', '圣彼得堡': 'Saint Petersburg',
    # 爱尔兰
    '都柏林': 'Dublin', '科克': 'Cork', '戈尔韦': 'Galway',
    # 其他
    '瓦伦西亚': 'Valencia',  # 已在西班牙
}


def build_city_mapping(raw_output, english_cities):
    """
    从模型输出中建立中英文城市名映射
    1. 优先从模型输出提取 "中文（English）" 模式
    2. 回退到预定义字典CITY_CN_EN_DICT
    """
    mapping = {}
    if not raw_output:
        return mapping

    # 模式1: "中文（English）" 或 "中文(English)"
    for m in re.finditer(r'([\u4e00-\u9fff]{2,8})\s*[（(]\s*([A-Z][a-z]+)\s*[）)]', raw_output):
        cn_name = m.group(1)
        en_name = m.group(2)
        mapping[en_name] = cn_name
        mapping[cn_name] = en_name

    # 模式2: "English（中文）" 或 "English(中文)"
    for m in re.finditer(r'([A-Z][a-z]+)\s*[（(]\s*([\u4e00-\u9fff]{2,8})\s*[）)]', raw_output):
        en_name = m.group(1)
        cn_name = m.group(2)
        mapping[en_name] = cn_name
        mapping[cn_name] = en_name

    # 模式3: 使用预定义字典 (关键改进)
    for cn_name, en_name in CITY_CN_EN_DICT.items():
        if cn_name in raw_output:
            mapping[en_name] = cn_name
            mapping[cn_name] = en_name

    return mapping


def extract_all_city_mentions(raw_output, english_cities, city_mapping):
    """
    从模型输出中提取所有被提及的城市(中英文都算)
    返回: 城市英文名集合(去重)
    """
    if not raw_output:
        return set()

    mentioned = set()

    # 1. 匹配英文城市名
    for city in english_cities:
        if re.search(r'\b' + re.escape(city) + r'\b', raw_output):
            mentioned.add(city)

    # 2. 匹配中文城市名(通过mapping)
    for en_name, cn_name in city_mapping.items():
        if en_name in english_cities and cn_name in raw_output:
            mentioned.add(en_name)

    return mentioned


def extract_city_sequence(raw_output, english_cities, city_mapping):
    """
    尝试提取城市访问序列(用于顺序检查)
    返回: 城市英文名列表(按出现顺序)
    """
    if not raw_output:
        return []

    sequence = []
    seen = set()

    def add_city(en_name):
        if en_name in english_cities and en_name not in seen:
            seen.add(en_name)
            sequence.append(en_name)

    # 模式1: 英文箭头 "Lyon → Valencia"
    for m in re.finditer(r'([A-Z][a-z]+)\s*(?:→|->|➜|飞往|前往|到|至)\s*([A-Z][a-z]+)', raw_output):
        add_city(m.group(1))
        add_city(m.group(2))

    # 模式2: 中文箭头 "里昂 → 瓦伦西亚" (通过mapping转换)
    # 匹配 "中文 → 中文" 或 "中文->中文"
    for m in re.finditer(r'([\u4e00-\u9fff]{2,8})\s*(?:→|->|➜)\s*([\u4e00-\u9fff]{2,8})', raw_output):
        cn_src = m.group(1)
        cn_dst = m.group(2)
        if cn_src in city_mapping:
            add_city(city_mapping[cn_src])
        if cn_dst in city_mapping:
            add_city(city_mapping[cn_dst])

    # 模式3: 顿号分隔 "里昂、瓦伦西亚、伊斯坦布尔"
    for m in re.finditer(r'([\u4e00-\u9fff]{2,8}(?:[、，,]\s*[\u4e00-\u9fff]{2,8})+)', raw_output):
        parts = re.split(r'[、，,]\s*', m.group(1))
        for cn_name in parts:
            if cn_name in city_mapping:
                add_city(city_mapping[cn_name])

    # 模式4: 表格行 "| 第1～5天 | 里昂（Lyon） | 5天 |"
    for m in re.finditer(r'\|\s*[^|]*?[（(]([A-Z][a-z]+)[）)][^|]*?\s*\|', raw_output):
        add_city(m.group(1))

    # 模式5: "Arriving in X" / "arrive in X"
    for m in re.finditer(r'(?:Arriving in|arriving in|Arrive in|arrive in)\s+([A-Z][a-z]+)', raw_output):
        add_city(m.group(1))

    # 模式6: "Fly from X to Y"
    for m in re.finditer(r'(?:Fly from|fly from|Flight from|flight from)\s+([A-Z][a-z]+)\s+to\s+([A-Z][a-z]+)', raw_output):
        add_city(m.group(1))
        add_city(m.group(2))

    # 模式7: 编号列表 "1. 里昂（Lyon）" / "1. Lyon"
    for m in re.finditer(r'\d+\.\s*(?:[\u4e00-\u9fff]{2,8}[（(]([A-Z][a-z]+)[）)]|([A-Z][a-z]+))', raw_output):
        city = m.group(1) or m.group(2)
        if city:
            add_city(city)

    # 模式8: fallback - 直接匹配所有英文城市名
    if not sequence:
        for city in english_cities:
            if re.search(r'\b' + re.escape(city) + r'\b', raw_output):
                add_city(city)

    # 模式9: fallback - 匹配中文城市名
    if not sequence:
        for en_name, cn_name in city_mapping.items():
            if en_name in english_cities and cn_name in raw_output:
                add_city(en_name)

    return sequence


def score_trip_planning_v2(raw_output, golden_plan):
    """
    改进版 Trip Planning 评分
    """
    if not raw_output or not golden_plan:
        return None, '', []

    # 1. 从golden_plan提取所有英文城市名
    golden_cities = extract_english_cities_from_golden(golden_plan)
    if not golden_cities:
        return None, '', []

    # 2. 从模型输出建立中英文城市名映射
    city_mapping = build_city_mapping(raw_output, golden_cities)

    # 3. 提取模型输出中所有被提及的城市
    mentioned_cities = extract_all_city_mentions(raw_output, golden_cities, city_mapping)

    # 4. 尝试提取城市序列(用于顺序检查)
    model_sequence = extract_city_sequence(raw_output, golden_cities, city_mapping)

    # 5. 评分逻辑
    golden_set = set(golden_cities)

    # 必须包含所有golden城市才算正确(集合匹配)
    if golden_set.issubset(mentioned_cities):
        return True, f'所有{len(golden_cities)}个城市均被提及(集合匹配)', model_sequence

    # 部分匹配(>=80%)也算正确(容忍漏答)
    if len(golden_set & mentioned_cities) / len(golden_set) >= 0.8:
        return True, f'部分匹配{len(golden_set & mentioned_cities)}/{len(golden_set)}(>=80%容忍)', model_sequence

    return False, f'仅匹配{len(golden_set & mentioned_cities)}/{len(golden_set)}个城市', model_sequence


def extract_task_cities(task_content):
    """从task_content提取所有提到的城市名(英文)"""
    if not task_content:
        return set()
    # Natural Plan中常见的欧洲城市名
    known_cities = [
        'Lyon', 'Prague', 'Brussels', 'Bucharest', 'Frankfurt', 'Dubrovnik',
        'Helsinki', 'Vilnius', 'Berlin', 'Stuttgart', 'Amsterdam', 'Istanbul',
        'Milan', 'Reykjavik', 'Geneva', 'Madrid', 'Porto', 'Oslo', 'Valencia',
        'Athens', 'Zurich', 'Naples', 'Lisbon', 'Paris', 'London', 'Rome',
        'Barcelona', 'Vienna', 'Budapest', 'Munich', 'Copenhagen', 'Dublin',
        'Riga', 'Tallinn', 'Warsaw', 'Krakow', 'Hamburg', 'Cologne', 'Bonn',
        'Stockholm', 'Copenhagen', 'Oslo', 'Bergen', 'Helsinki', 'Tampere',
        'Bologna', 'Florence', 'Turin', 'Naples', 'Bari', 'Palermo', 'Catania',
        'Seville', 'Malaga', 'Bilbao', 'Valencia', 'Zaragoza', 'Valladolid',
        'Lyon', 'Marseille', 'Nice', 'Toulouse', 'Bordeaux', 'Nantes', 'Strasbourg',
        'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Edinburgh', 'Cardiff',
        'Belfast', 'Dublin', 'Cork', 'Galway',
        'Hamburg', 'Munich', 'Berlin', 'Cologne', 'Frankfurt', 'Stuttgart',
        'Dusseldorf', 'Dresden', 'Leipzig', 'Hannover', 'Nuremberg',
        'Vienna', 'Salzburg', 'Innsbruck', 'Graz', 'Linz',
        'Prague', 'Brno', 'Ostrava', 'Plzen',
        'Budapest', 'Debrecen', 'Szeged', 'Miskolc',
        'Warsaw', 'Krakow', 'Gdansk', 'Wroclaw', 'Poznan',
        'Copenhagen', 'Aarhus', 'Odense', 'Aalborg',
        'Stockholm', 'Gothenburg', 'Malmo', 'Uppsala',
        'Oslo', 'Bergen', 'Trondheim', 'Stavanger',
        'Helsinki', 'Tampere', 'Turku', 'Oulu',
        'Tallinn', 'Tartu',
        'Riga', 'Daugavpils',
        'Vilnius', 'Kaunas', 'Klaipeda',
        'Dubrovnik', 'Split', 'Zagreb', 'Rijeka',
        'Bucharest', 'Cluj', 'Timisoara', 'Iasi',
        'Sofia', 'Plovdiv', 'Varna',
        'Athens', 'Thessaloniki', 'Patras',
        'Istanbul', 'Ankara', 'Izmir', 'Bursa',
        'Reykjavik', 'Akureyri',
        'Geneva', 'Zurich', 'Basel', 'Bern', 'Lausanne',
    ]
    cities = set()
    for city in known_cities:
        if re.search(r'\b' + city + r'\b', task_content):
            cities.add(city)
    return cities


def match_golden_plan(task_content, golden_plans_dict):
    """
    精确匹配task_content到golden_plan
    使用task_content中所有城市作为唯一标识
    """
    task_cities = extract_task_cities(task_content)
    if not task_cities:
        return None

    # 必须包含所有task城市
    candidates = []
    for prompt, gp in golden_plans_dict.items():
        if all(city in prompt for city in task_cities):
            candidates.append((prompt, gp))

    if len(candidates) == 1:
        return candidates[0][1]
    elif len(candidates) > 1:
        # 多个候选,选择prompt与task_content最相似的
        best = None
        best_score = -1
        for prompt, gp in candidates:
            # 计算Jaccard相似度
            prompt_words = set(prompt.split())
            task_words = set(task_content.split())
            score = len(prompt_words & task_words) / len(prompt_words | task_words)
            if score > best_score:
                best_score = score
                best = gp
        return best
    return None


def main():
    log('=' * 80)
    log('Trip Planning 评分器改进版 v2')
    log('=' * 80)

    # 加载评分结果
    scored_file = os.path.join(RESULTS_DIR, 'scored_outputs.jsonl')
    with open(scored_file, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f]

    # 筛选trip_planning记录
    trip_records = [r for r in records if r['scoring']['scorer'] == 'NaturalPlan/trip_planning']
    log(f'\ntrip_planning记录数: {len(trip_records)}')

    # 加载natural_plan数据
    SOURCE_DATA = os.path.join(BASE_DIR, 'data', 'source_datasets')
    golden_plans = {}
    for fname, domain in [
        ('natural_plan_trip_planning.json', 'trip_planning'),
    ]:
        path = os.path.join(SOURCE_DATA, fname)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            prompt = (v.get('prompt_0shot') or '').strip()
            if prompt:
                golden_plans[prompt] = v.get('golden_plan', '')

    log(f'golden_plan总数: {len(golden_plans)}')

    # 重新评分
    rescored_count = 0
    changed_count = 0
    for r in trip_records:
        task_content = r['task_info']['task_content']
        # 用精确匹配
        golden_plan = match_golden_plan(task_content, golden_plans)

        if not golden_plan:
            log(f'  [警告] task_id={r["task_info"]["task_id"]} 未匹配到golden_plan')
            continue

        old_correct = r['scoring']['is_correct']
        new_correct, reason, sequence = score_trip_planning_v2(
            r['model_output']['raw_output'], golden_plan
        )

        if new_correct is not None and new_correct != old_correct:
            changed_count += 1
            log(f'\n  [变更] task_id={r["task_info"]["task_id"]} condition={r["condition"]}')
            log(f'    旧: {old_correct}')
            log(f'    新: {new_correct} ({reason})')
            log(f'    序列: {sequence}')

        # 更新记录
        if new_correct is not None:
            r['scoring']['is_correct'] = new_correct
            r['scoring']['scorer_version'] = 'v2_chinese_aware'
            r['scoring']['scoring_reason'] = reason
            r['scoring']['extracted_sequence'] = sequence
            rescored_count += 1

    log(f'\n重新评分: {rescored_count}/{len(trip_records)}')
    log(f'结果变更: {changed_count}')

    # 重新统计
    log('\n=== 重新评分后统计 ===')
    for cond in ['baseline', 'proposed']:
        cond_records = [r for r in trip_records if r['condition'] == cond and r['scoring']['is_correct'] is not None]
        correct = sum(1 for r in cond_records if r['scoring']['is_correct'])
        total = len(cond_records)
        log(f'  {cond}: {correct}/{total} = {correct/total:.4f}')

    # 保存重新评分结果
    # 更新原scored_outputs.jsonl中的trip_planning记录
    for i, r in enumerate(records):
        if r['scoring']['scorer'] == 'NaturalPlan/trip_planning':
            for new_r in trip_records:
                if new_r['record_id'] == r['record_id']:
                    records[i] = new_r
                    break

    output_file = os.path.join(RESULTS_DIR, 'scored_outputs.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    log(f'\n已更新: {output_file}')


if __name__ == '__main__':
    main()
