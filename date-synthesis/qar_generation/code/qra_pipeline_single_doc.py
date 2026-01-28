import sys
import os
import glob
import json
import argparse
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Validate environment variables
if not openai_api_key:
    logging.error("OPENAI_API_KEY is not set in the environment.")
    sys.exit(1)

# Add root directory to system path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from client import OpenAIClient as Client
from data_processing import postprocess
from data_processing.postprocess import postprocess_en
from utils.utils import read_prompt, read_config_json, write_config_json


def safe_format(template: str, **values) -> str:
    """Safely replace placeholders without affecting JSON braces."""
    result = template
    for key, value in values.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def process_qra_document(model_name: str, file_path: str, prompts: List[Dict], input_dir: str, output_dir: str) -> None:
    """Process each document to generate QRA triples."""
    try:
        # 先计算输出路径，如已存在则直接跳过，方便断点续跑
        file_name = os.path.basename(file_path)
        rel_parent = os.path.dirname(os.path.relpath(file_path, input_dir))
        target_dir = os.path.join(output_dir, rel_parent)
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, file_name)

        if os.path.exists(output_path):
            logging.info(f"Output already exists, skip processing: {output_path}")
            return

        gpt_client = Client(openai_api_key=openai_api_key, model_name=model_name)
        logging.info(f"Using postprocess module from {postprocess.__file__}")
        config = read_config_json(file_path)

        # Prepare prompt types (包括 reference 抽取阶段用的 single document reference)
        prompt_types = {
            p["prompt_type"]: p
            for p in prompts
            if p["prompt_type"] in [
                "Factual Question",
                "Multi-hop Reasoning Question",
                "Summarization Question",
                "single document reference",
            ]
        }

        config_str = json.dumps(config, ensure_ascii=False)

        # 第一阶段：根据 config 生成三类 QA
        qa_tasks = [
            {
                "system_prompt": prompt_types[key]['system_prompt'],
                "user_prompt": safe_format(
                    prompt_types[key]['user_prompt'],
                    config=config_str,
                )
            }
            for key in ['Factual Question', 'Multi-hop Reasoning Question', 'Summarization Question']
        ]

        # Generate responses
        responses = gpt_client.generate(qa_tasks)

        # Postprocess and update the config
        for i, key in enumerate(['qa_fact_based', 'qa_multi_hop', 'qa_summary']):
            responses[i] = postprocess_en(
                response=responses[i],
                system_prompt=qa_tasks[i]['system_prompt'],
                user_prompt=qa_tasks[i]['user_prompt'],
                model_name=model_name
            )
            config[key] = responses[i]

        doc_path = file_path.replace('config', 'doc').replace('.json', '.txt')
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_content = f.read()

            config['Generated Article'] = doc_content

            qa_tasks = [
                {
                    "system_prompt": prompt_types['single document reference']['system_prompt'],
                    "user_prompt": safe_format(
                        prompt_types['single document reference']['user_prompt'],
                        doc=doc_content,
                        qa_pairs=json.dumps(config[key], ensure_ascii=False),
                    ),
                }
                for key in ['qa_fact_based', 'qa_multi_hop', 'qa_summary']
            ]

            responses = gpt_client.generate(qa_tasks)

            for i, key in enumerate(['qa_fact_based', 'qa_multi_hop', 'qa_summary']):
                responses[i] = postprocess_en(
                    response=responses[i],
                    system_prompt=qa_tasks[i]['system_prompt'],
                    user_prompt=qa_tasks[i]['user_prompt'],
                    model_name=model_name,
                )
                config[key] = responses[i]
        else:
            logging.warning(f"Doc file not found for config: {file_path} -> {doc_path}. Skip reference extraction stage.")

        write_config_json(output_path, config)
        logging.info(f"Finished processing {output_path}.")
    except Exception:
        logging.exception(f"Error while processing {file_path}")
        raise


def generate_qra(model_name: str, input_dir: str, output_dir: str, json_idx: int = None) -> None:
    """Generate QRA triples for each chapter in Law."""
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts.jsonl')
    prompts = read_prompt(
        file_path=prompt_path,
    )

    # Collect JSON files for processing (recursively)
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        # If json_idx is set, only keep files under the top-level folder
        # whose name (relative to input_dir) matches json_idx
        rel_root = os.path.relpath(root, input_dir)
        if json_idx is not None and rel_root != '.':
            top_level = rel_root.split(os.sep)[0]
            if top_level != str(json_idx):
                continue

        for fname in files:
            if fname.endswith('.json'):
                json_files.append(os.path.join(root, fname))

    # Process each file in parallel
    logging.info(f"Found {len(json_files)} JSON files to process")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_qra_document, model_name, file_path, prompts, input_dir, output_dir) for file_path in json_files]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing file: {e}")

    logging.info("All QRA generation tasks completed")


def main():
    parser = argparse.ArgumentParser(description='Generate QRA for documents.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the OpenAI model to use')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output JSON files')
    parser.add_argument('--json_idx', type=int, default=None, help='Index of the JSON files to process (if None, process all)')

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    generate_qra(
        model_name=args.model_name, 
        input_dir=args.input_dir, 
        output_dir=args.output_dir,
        json_idx=args.json_idx
    )


if __name__ == "__main__":
    main()