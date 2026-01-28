import os
import json
import argparse
from typing import List

def doc_corporation(domains: List, languages: List, output_dir: str) -> None:
    output_list = []
    for domain in domains:
        for language in languages:
            domain_dir = f'data/{domain}/{language}' 
            for root, dirs, files in os.walk(domain_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Extract meaningful title from path
                            # Path structure: data/law/zh/doc/[罪名]/0/0.txt
                            path_parts = file_path.split('/')
                            if len(path_parts) >= 4:
                                title = path_parts[-3]  # Extract the crime type (罪名)
                            else:
                                title = file.replace('.txt', '')
                            
                            jsonl_obj = {
                                "domain": domain.capitalize(),
                                "language": language,
                                "title": title,
                                "content": content
                            }
                            output_list.append(jsonl_obj)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    new_file_name = 'DRAGONBALL_docs.jsonl'
    output_path = os.path.join(output_dir, new_file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in output_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(output_list)} documents")
    print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Corporation of document data")
    parser.add_argument("--domains", type=str, required=True, help="Comma-separated list of domains")
    parser.add_argument("--languages", type=str, required=True, help="Comma-separated list of languages")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the JSONL file")
    
    args = parser.parse_args()
    
    domains = args.domains.split(',')
    languages = args.languages.split(',')
    
    doc_corporation(domains, languages, args.output_dir)

if __name__ == "__main__":
    main()
