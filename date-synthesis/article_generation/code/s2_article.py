import os
import json
import sys
from openai import OpenAI
import random
import argparse
import time
import pathlib
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from utils import load_json_data, save_output

openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('BASE_URL')

LAW_BUCKETS = [
    ("Civil Code", 0, 9, 'Civil'),
    ("Criminal Law", 10, 19, 'Criminal'),
    ("Public Security Administration Punishments Law", 20, 29, 'Criminal'),
    ("Administrative Penalty Law", 30, 39, 'Administrative'),
    ("Labor Contract Law", 40, 49, 'Civil')
]


def case_type_from_index(crime_index):
    for _, start, end, case_type in LAW_BUCKETS:
        if start <= crime_index <= end:
            return case_type
    return 'Civil'


def get_bucket_info(crime_index):
    for bucket_name, start, end, case_type in LAW_BUCKETS:
        if start <= crime_index <= end:
            return bucket_name, start, end, case_type
    return None


def build_crime_context(crime_dict_list, crime_index, mode, pair_idx=None):
    crime_dict = crime_dict_list[crime_index]
    name = crime_dict['Name']
    detail = crime_dict['Details']

    if mode != 'pair':
        return name, detail

    bucket_info = get_bucket_info(crime_index)
    if not bucket_info:
        return name, detail

    _, start, end, _ = bucket_info

    # 显式指定 pair_idx 时优先使用该索引（需在同一 bucket 内）
    if pair_idx is not None and 0 <= pair_idx < len(crime_dict_list) and pair_idx != crime_index:
        if start <= pair_idx <= end:
            other_idx = pair_idx
        else:
            return name, detail
    else:
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

    return name, combined_detail

def extract_article_numbers(detail_text):
    pattern = re.compile(r"Article\s+\d+")
    return list(dict.fromkeys(pattern.findall(detail_text)))


def generate_article(
    model_name,
    data_for_complete,
    crime_name,
    crime_detail,
    case_type,
):
    time.sleep(random.random() * 1.5)
    system_prompt = (
        "You are a senior clerk of a Chinese people's court. You draft long-form judgments in formal English, "
        "faithfully reflecting Chinese legal procedures, reasoning, and tone."
    )
    template_str = json.dumps(data_for_complete, ensure_ascii=False, indent=1)
    sections = [
        "Court and Prosecutorial Background",
        "Parties and Defense Counsel",
        "Procedural History",
        "Findings of Fact",
        "Charges and Applicable Law",
        "Evidence Evaluation",
        "Court's Analysis",
        "Sentencing Considerations" if case_type == 'Criminal' else "Relief Considerations",
        "Judgment Result",
        "Right to Appeal"
    ]
    statute_refs = '\n'.join(extract_article_numbers(crime_detail)) or "Article numbers unavailable"
    user_prompt = f"""Using the structured JSON case record below, draft an English judgment for a Chinese court.

Requirements:
1. Keep the structure in coherent prose, not JSON; follow the sections: {', '.join(sections)}.
2. Minimum length: 8,000 English words. Expand each part with concrete procedural and factual detail.
3. Adopt the perspective of "This Court" and keep the tone solemn, detailed, and consistent with PRC judicial writing.
4. All facts, evidence, and reasoning must align with the JSON content but you may elaborate plausibly.
5. Discuss each piece of evidence, how it supports or contradicts key elements, and why the court accepts or rejects arguments.
6. Cite statutory authority only by article numbers (e.g., Article 263) without quoting full text.
7. Conclude with a detailed disposition (sentence or civil/administrative relief) and inform parties of appeal rights.
8. After the main text, include a "Reference" section listing the relevant article numbers extracted from the legal background.

JSON case record:
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
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            time.sleep(1)
    response += '\n\nReference Articles:\n' + statute_refs + '\n' + crime_detail
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "pair"])
    parser.add_argument("--model_name", type=str, default='gpt-4o')
    parser.add_argument("--config_dir", type=str, default=None)
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
    base_crime_dict = crime_dict_list[args.crime_name_idx]
    case_type = base_crime_dict.get('CaseType') or case_type_from_index(args.crime_name_idx)
    data = load_json_data(pathlib.Path(args.config_dir) / crime_name / str(json_idx) / f'{str(json_idx)}.json')
    
    response = generate_article(model_name, data, crime_name, crime_detail, case_type)
    save_output(args.output_dir, response, crime_name, json_idx, None, "txt")


if __name__ == "__main__":
    main()
