import os
import json
import argparse
from typing import List, Dict


def _extract_references_from_qa(qa: Dict) -> str:
 
    refs = qa.get("ref") or qa.get("refs") or []
    if isinstance(refs, list):
        return "\n".join(refs)
    return str(refs) if refs is not None else ""


def _should_keep_qa(qa: Dict) -> bool:

    # 显式跳过无关无解问题
    if qa.get("question type") == "Irrelevant Unsolvable Question":
        return False

    if qa.get("answer") == "Unable to answer.":
        return False

    refs = qa.get("ref") or qa.get("refs") or []
    if isinstance(refs, list) and len(refs) == 0:
        return False

    return True

def qa_format_single_doc(input_path: str, doc_root_dir: str) -> List[Dict]:
    key_list = ["qa_fact_based", "qa_multi_hop", "qa_summary"]
    output = []
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # 使用整篇判决文书作为 references，优先从 Generated Article 读取
                doc_content = json_data.get("Generated Article", "")
                
                for key in key_list:
                    if key in json_data:
                        for qa in json_data[key]:
                            if not _should_keep_qa(qa):
                                continue
                            # 回退为整篇 doc 作为引用
                            references = doc_content
                            # 仅保留指定字段
                            jsonl_obj = {
                                "query": {
                                    "query_type": qa['question type'],
                                    "content": qa['question'],
                                },
                                "ground_truth": {
                                    "content": qa['answer'],
                                    "references": references
                                }
                            }
                            output.append(jsonl_obj)
    return output

def qa_format_multi_doc(input_path: str, doc_root_dir: str) -> List[Dict]:
    output = []
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 仅处理多文档整合QA
                if 'qa_multi_document_information_integration' in json_data:
                    for qa in json_data['qa_multi_document_information_integration']:
                        if not _should_keep_qa(qa):
                            continue
                        references = _extract_references_from_qa(qa)
                        # 仅保留指定字段
                        jsonl_obj = {
                            "query": {
                                "query_type": qa['question type'],
                                "content": qa['question'],
                            },
                            "ground_truth": {
                                "content": qa['answer'],
                                "references": references
                            }
                        }
                        output.append(jsonl_obj)
    return output

def qa_format_irrelevant(input_path: str, doc_root_dir: str) -> List[Dict]:
    output = []
    
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for qa in json_data:
            if not _should_keep_qa(qa):
                continue
            references = _extract_references_from_qa(qa)
            jsonl_obj = {
                "query": {
                    "query_type": qa['question type'],
                    "content": qa['question'],
                },
                "ground_truth": {
                    "content": qa['answer'],
                    "references": references
                }
            }
            output.append(jsonl_obj)
    return output

def main():
    parser = argparse.ArgumentParser(description="Corporation of QRA data")
    parser.add_argument("--single_doc_dir", type=str, required=True, 
                        help="Directory containing single-doc QRA config JSON files")
    parser.add_argument("--doc_root_dir", type=str, required=True,
                        help="Root directory containing original judgment documents (for filling references)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for the final JSON dataset")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # doc 文件根目录通过参数传入
    doc_root_dir = args.doc_root_dir
    if not os.path.exists(doc_root_dir):
        print(f"Error: Doc root directory {doc_root_dir} does not exist!")
        return
    
    # 单文档 QA 的输入目录通过参数传入
    single_doc_input = args.single_doc_dir
    #multi_doc_input = os.path.join(args.input_root, "qra_multidoc_test")
    
    # 收集所有QRA数据
    output = []
    
    # 处理单文档QA
    if os.path.exists(single_doc_input):
        print(f"Processing single-doc QA from {single_doc_input}...")
        single_doc_qa = qa_format_single_doc(single_doc_input, doc_root_dir)
        output.extend(single_doc_qa)
        print(f"Added {len(single_doc_qa)} single-doc QRAs")
    else:
        print(f"Warning: Single-doc QA directory {single_doc_input} does not exist!")
    
    # 处理多文档QA
    #if os.path.exists(multi_doc_input):
    #    print(f"Processing multi-doc QA from {multi_doc_input}...")
    #    multi_doc_qa = qa_format_multi_doc(multi_doc_input, doc_root_dir)
    #    output.extend(multi_doc_qa)
    #    print(f"Added {len(multi_doc_qa)} multi-doc QRAs")
    #else:
    #    print(f"Warning: Multi-doc QA directory {multi_doc_input} does not exist!")
    
    # 无关QA（如需启用可取消注释）
    # irrelevant_input = os.path.join(args.input_root, "qra_irrelevant.json")
    # if os.path.exists(irrelevant_input):
    #     irrelevant_qa = qa_format_irrelevant(irrelevant_input, doc_root_dir)
    #     output.extend(irrelevant_qa)
    #     print(f"Added {len(irrelevant_qa)} irrelevant QRAs")
    # else:
    #     print(f"Warning: Irrelevant QA file {irrelevant_input} does not exist!")
    
    output_file_json = os.path.join(args.output_dir, "QA-summary-全文.json")
    with open(output_file_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Total QRAs generated: {len(output)}")
    print(f"Output saved to: {output_file_json}")

if __name__ == "__main__":
    main()