import os
import json
import sys
from openai import OpenAI
import random
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from utils import load_json_data, save_output

openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('BASE_URL')
names = [
    'Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu', 'Chongqing',
    'Wuhan', 'XiAn', 'Nanjing', 'Hangzhou', 'Suzhou', 'Qingdao', 'Tianjin',
    'Shenyang', 'Dalian', 'Jinan', 'Zhengzhou', 'Changsha', 'Fuzhou',
    'Xiamen', 'Ningbo', 'Wuxi', 'Foshan', 'Dongguan', 'Harbin', 'Kunming',
    'Lanzhou', 'Urumqi', 'Hohhot', 'Haikou', 'Sanya', 'Zhuhai', 'Shantou',
    'Taizhou', 'Quanzhou', 'Yantai', 'Weihai', 'Hefei', 'Nanchang',
    'Guiyang', 'Guilin', 'Liuzhou', 'Yinchuan', 'Taiyuan', 'Luoyang',
    'Anyang', 'Datong', 'Lianyungang', 'Zibo', 'Xuzhou', 'SuzhouAnhui',
    'Huizhou', 'Jiangmen', 'Zhongshan', 'Chaozhou', 'Jieyang', 'Shijiazhuang',
    'Langfang', 'Tangshan', 'Qinhuangdao', 'Baoding', 'Handan', 'Cangzhou',
    'Hengshui', 'Jiaxing', 'Huzhou', 'Shaoxing', 'Jinhua', 'Lishui',
    'Wenzhou', 'Linyi', 'Dezhou', 'Heze', 'Binzhou', 'Weifang', 'Rizhao',
    'Hulunbuir', 'Ordos', 'YanAn', 'Baoji', 'Hanzhong', 'Mianyang',
    'Deyang', 'Leshan', 'Meishan', 'Ziyang', 'Panzhihua', 'Zhoushan',
    'Changzhou', 'Maoming', 'Shaoguan', 'Heyuan', 'Loudi', 'Shaoyang',
    'Zhangjiajie', 'Yueyang', 'Zhuzhou', 'Yancheng'
]
surnames = [
    'Zhang', 'Wang', 'Li', 'Zhao', 'Liu', 'Chen', 'Yang', 'Huang', 'Wu',
    'Zhou', 'Xu', 'Sun', 'Ma', 'Zhu', 'Hu', 'Guo', 'He', 'Gao', 'Lin',
    'Lu', 'Tang', 'Feng', 'Yu', 'Cao', 'Liang', 'Song', 'Han', 'Dong',
    'Xiao', 'Yuan', 'Pan', 'Jiang', 'Cui', 'Qin', 'Shen', 'Deng', 'Bao',
    'Peng', 'Jin', 'Wei', 'Fan', 'Rao', 'Tan', 'Hou', 'Nie', 'Luo',
    'Bian', 'Shi', 'Pei', 'Gong', 'Wan', 'Yao', 'Tian', 'Qiao', 'Wen',
    'Liao', 'Ji', 'Du', 'Qiu', 'Qu', 'Xiong', 'Yin', 'Kang', 'Liang',
    'Mo', 'Lei', 'Hao', 'Meng', 'Shang', 'Kong', 'Ai', 'An', 'Bi',
    'Chai', 'Chu', 'FanYu', 'Fei', 'FengLei', 'Gu', 'Huangfu', 'Jiao',
    'Lian', 'Long', 'Miao', 'Ou', 'Ruan', 'Su', 'Tao', 'Xue', 'Yan',
    'Zeng', 'Zhong', 'Zou', 'Shu', 'HanYu'
]
characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

COURT_LEVELS = [
    "Basic People's Court",
    "Intermediate People's Court",
    "High People's Court"
]
CASE_COMPLEXITIES = ['General', 'Complex', 'Significant']
PROCEDURES_BY_TYPE = {
    'Civil': ['First-instance civil trial', 'Second-instance civil appeal', 'Civil mediation'],
    'Criminal': ['Public prosecution', 'First-instance criminal trial', 'Second-instance criminal appeal'],
    'Administrative': ['Administrative reconsideration', 'Administrative litigation', 'Judicial review']
}
PROCEDURE_STAGES = ['Filing', 'Investigation', 'Evidence exchange', 'Court hearing', 'Deliberation', 'Awaiting judgment']
REPRESENTATIVE_SUFFIXES = ['Law Firm', 'Attorneys', 'Legal Service Center']

LAW_BUCKETS = [
    ("Civil Code", 0, 9, 'Civil'),
    ("Criminal Law", 10, 19, 'Criminal'),
    ("Public Security Administration Punishments Law", 20, 29, 'Criminal'),
    ("Administrative Penalty Law", 30, 39, 'Administrative'),
    ("Labor Contract Law", 40, 49, 'Civil')
]


SCENARIO_ARCHETYPES = {
    'Civil': [
        'consumer finance dispute between an individual borrower and a commercial lender',
        'real estate transaction dispute involving purchase, mortgage or lease of property',
        'contract dispute between two companies over delivery, quality and payment',
        'family-related dispute involving marital property, support or inheritance issues',
        'tort dispute involving personal injury or property damage in daily life or work'
    ],
    'Criminal': [
        'street-level violent incident involving physical confrontation and weapons',
        'economic crime involving corruption, fraud, or misappropriation of funds',
        'organized group crime involving multiple offenders and repeated offenses',
        'public order incident occurring in crowded public spaces or transportation hubs',
        'cyber or technology-related crime involving online transactions or data misuse'
    ],
    'Administrative': [
        'administrative penalty dispute between a citizen and a local administrative bureau',
        'dispute over licensing, approval or registration by an administrative organ',
        'labor security or social insurance administrative enforcement dispute',
        'public security administrative case arising from minor violations of order',
        'dispute over land, planning or environmental regulation enforcement'
    ]
}


def get_bucket_info(crime_index):
    for bucket_name, start, end, case_type in LAW_BUCKETS:
        if start <= crime_index <= end:
            return bucket_name, start, end, case_type
    return None


def build_crime_context(crime_dict_list, crime_index, mode, pair_idx=None):
    crime_dict = crime_dict_list[crime_index]
    name = crime_dict['Name']
    detail = crime_dict['Details']

    # 默认 single 模式：保持原有行为
    if mode != 'pair':
        return name, detail

    bucket_info = get_bucket_info(crime_index)
    if not bucket_info:
        return name, detail

    _, start, end, _ = bucket_info

    # 如果显式指定了 pair_idx，且在同一 bucket 内，则优先使用该索引
    if pair_idx is not None and 0 <= pair_idx < len(crime_dict_list) and pair_idx != crime_index:
        if start <= pair_idx <= end:
            other_idx = pair_idx
        else:
            return name, detail
    else:
        # 否则在同一 LAW_BUCKET 范围内随机选择另一个不同的法条
        candidates = [
            idx
            for idx in range(start, min(end + 1, len(crime_dict_list)))
            if idx != crime_index
        ]
        if not candidates:
            return name, detail
        other_idx = random.choice(candidates)
    other = crime_dict_list[other_idx]

    combined_detail = (
        f"Primary charge or legal focus: {name}\n"
        f"{detail}\n\n"
        f"Additional related charge or legal focus from the same statute bucket: {other['Name']}\n"
        f"{other['Details']}"
    )

    # 名称仍然用主法条，保证目录结构与原流程一致
    return name, combined_detail


def generate_person_name():
    return random.choice(characters) + '. ' + random.choice(surnames)


def random_case_number():
    current_year = datetime.now().year
    year = random.randint(current_year - 6, current_year)
    suffix = ''.join(random.choices(characters, k=2))
    digits = ''.join(str(random.randint(0, 9)) for _ in range(3))
    return f"({year}) {suffix}{digits}"


def case_type_from_index(crime_index):
    for _, start, end, case_type in LAW_BUCKETS:
        if start <= crime_index <= end:
            return case_type
    return 'Civil'


def select_procedure(case_type):
    return random.choice(PROCEDURES_BY_TYPE.get(case_type, PROCEDURES_BY_TYPE['Civil']))


def random_stage():
    return random.choice(PROCEDURE_STAGES)


def random_law_firm(city):
    suffix = random.choice(REPRESENTATIVE_SUFFIXES)
    return f"{city} {random.choice(surnames)} {suffix}"


def build_actor(role, legal_status, city, representative=None):
    return {
        "role": role,
        "name": generate_person_name(),
        "description": "",
        "legal_status": legal_status,
        "representative_info": representative or random_law_firm(city)
    }


def build_actor_profiles(case_type, city):
    if case_type == 'Criminal':
        return [
            build_actor('Prosecutor', f"{city} People's Procuratorate", city, f"{city} People's Procuratorate"),
            build_actor('Defendant', 'Natural person', city),
            build_actor('Victim', 'Natural person', city)
        ]
    if case_type == 'Administrative':
        return [
            build_actor('Applicant', 'Natural person', city),
            build_actor('Respondent', f"{city} Administrative Bureau", city, f"{city} Administrative Bureau Legal Affairs Office")
        ]
    return [
        build_actor('Plaintiff', 'Natural person', city),
        build_actor('Defendant', 'Natural person', city)
    ]


def generate_article(
    model_name,
    data_for_complete,
    crime_name,
    crime_detail,
    case_type,
):
    time.sleep(random.random() * 1.5)
    system_prompt = 'You are an experienced court clerk. You produce structured, internally consistent case files with precise legal reasoning.'
    template_str = json.dumps(data_for_complete, ensure_ascii=False, indent=1)
    # 为同一罪名下的多次生成引入更明显的多样性：
    # 1) 随机场景类型 seed；2) 随机 key_events 数量（3-6 之间）
    archetypes = SCENARIO_ARCHETYPES.get(case_type, SCENARIO_ARCHETYPES['Civil'])
    scenario_seed = random.choice(archetypes)
    events_count = random.randint(3, 6)
    user_prompt = f"""Draft a comprehensive legal case record in JSON that follows the provided schema exactly.
Charge or focus: {crime_name}

Legal background you must rely on:
{crime_detail}

Scenario style seed (use this to shape the overall narrative so that repeated cases for the same charge can differ in pattern, industry and relationships):
{scenario_seed}

Instructions:
1. Fill every empty string, list, or placeholder in the template with concrete details. Preserve all field names and hierarchy.
2. `case_metadata` must include court level, court name, case number format, procedure, complexity level, cause of action, and current stage consistent with the facts.
3. Build a vivid `scenario`: craft a title, detailed background, timeline summary, and populate `actors` (roles and names are seeded) with rich descriptions, legal status notes, and representative information.
4. Provide {events_count} chronological `key_events`; each object must include `date`, `event`, and `details`, showing how the matter developed. Make the sequence of events clearly different across possible cases under the same charge.
5. In `remedy_request` and `legal_issue`, articulate the claims, defenses, and central questions.
6. Translate the statutes described in the legal background into structured entries under `legal_basis.statutes`, and add any related judicial interpretations or guiding cases when relevant.
7. Use `analysis.fact_finding`, `evidence_evaluation`, `legal_interpretation`, and `dispute_resolution` to synthesize facts, evidence types, and legal reasoning.
8. `answer.short_answer` should summarize the outcome succinctly, while `answer.final_explanation` must describe the remedy or sentence in detail.
9. Keep the tone professional and ensure the JSON is valid. Do not add commentary outside the JSON object.

Template to complete:
{template_str}
"""
    if base_url != '':
        client = OpenAI(api_key=openai_api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=openai_api_key)
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            response = response[response.find("{") : response.rfind("}") + 1]
            response = json.loads(response)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            time.sleep(1)
    return response
    

def set_value(data, crime_name, case_type):
    city = random.choice(names)
    court_level = random.choice(COURT_LEVELS)
    if court_level == "Basic People's Court":
        court_name = f"{city} People's Court"
    elif court_level == "Intermediate People's Court":
        court_name = f"{city} Intermediate People's Court"
    else:
        court_name = f"{city} High People's Court"

    data["case_metadata"]["jurisdiction"] = "China"
    data["case_metadata"]["court_level"] = court_level
    data["case_metadata"]["court_name"] = court_name
    data["case_metadata"]["case_type"] = case_type
    data["case_metadata"]["procedure"] = select_procedure(case_type)
    data["case_metadata"]["complexity_level"] = random.choice(CASE_COMPLEXITIES)
    data["case_metadata"]["case_number"] = random_case_number()
    data["case_metadata"]["cause_of_action"] = crime_name
    data["case_metadata"]["current_procedure_stage"] = random_stage()

    start_year = random.randint(datetime.now().year - 4, datetime.now().year - 1)
    end_year = start_year + random.randint(0, 2)
    data["scenario"]["title"] = f"{crime_name} case in {city}"
    data["scenario"]["background"] = ""
    data["scenario"]["timeline"] = f"{start_year} - {end_year}"
    data["scenario"]["actors"] = build_actor_profiles(case_type, city)
    data["scenario"]["key_events"] = []

    data["remedy_request"]["claimant_request"] = ""
    data["remedy_request"]["defendant_response"] = ""
    data["legal_issue"]["central_question"] = ""
    data["legal_issue"]["issue_type"] = ""
    data["legal_issue"]["sub_issues"] = []

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "pair"])
    parser.add_argument("--model_name", type=str, default='gpt-4o')
    parser.add_argument("--data_for_complete", type=str, default=None)
    parser.add_argument("--crime_names", type=str, default=None)
    parser.add_argument("--crime_name_idx", type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument("--json_idx", type=int, default=0)
    parser.add_argument("--pair_idx", type=int, default=None)
    args = parser.parse_args()

    json_idx = args.json_idx
    model_name = args.model_name

    crime_dict_list = load_json_data(args.crime_names)
    crime_name, crime_detail = build_crime_context(crime_dict_list, args.crime_name_idx, args.mode, args.pair_idx)
    case_type = case_type_from_index(args.crime_name_idx)
    data = load_json_data(args.data_for_complete)
    data = set_value(data, crime_name, case_type)
    response = generate_article(model_name, data, crime_name, crime_detail, case_type)
    save_output(args.output_dir, response, crime_name, json_idx, None, "json")

if __name__ == "__main__":
    main()
