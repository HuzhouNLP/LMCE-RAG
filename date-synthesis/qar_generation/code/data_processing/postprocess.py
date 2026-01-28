import sys
import os
import json
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Add root directory to system path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from client import OpenAIClient as Client

def postprocess_zh(response, system_prompt: str, user_prompt: str, model_name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    Modified for local model usage - no OpenAI API dependency.
    """
    # 检查是否是本地模型，如果是则跳过OpenAI客户端初始化
    is_local_model = "local" in model_name.lower() if model_name else False
    
    if not is_local_model:
        gpt_client = Client(
            openai_api_key=openai_api_key, 
            model_name=model_name,
        )
    else:
        gpt_client = None
    
    output = []
    flag = True
    while True:
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    if 'ref' in item:
                        try:
                            if "无法回答" in item['答案'] or item['ref'] == []:
                                item['问题类型'] = '无关无解问'
                                item['ref'] = []
                            json_obj = {
                                "问题类型": item['问题类型'],
                                "问题": item['问题'],
                                "答案": item['答案'],
                                "ref": item['ref'],
                            }
                            output.append(json_obj)
                        except KeyError:
                            flag = False
                            break
                    else:
                        try:
                            if "无法回答" in item['答案']:
                                item['问题类型'] = '无关无解问'
                            json_obj = {
                                "问题类型": item['问题类型'],
                                "问题": item['问题'],
                                "答案": item['答案'],
                            }
                            output.append(json_obj)
                        except KeyError:
                            flag = False
                            break    
            except json.JSONDecodeError:
                flag = False
        
            if not flag:
                response = gpt_client.generate([{"system_prompt": system_prompt, "user_prompt": user_prompt}])[0]
                flag = True
            else:
                return output

def postprocess_en(response, system_prompt: str, user_prompt: str, model_name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    """
    gpt_client = Client(
        openai_api_key=openai_api_key,
        model_name=model_name,
    )

    max_retries = 3
    attempts = 0

    while attempts < max_retries:
        # 每轮解析前清空输出
        output: List = []

        # 去掉可能的 ``` 包裹
        if "```" in response:
            response = response.replace("```json\n", "").replace("```", "")

        # 优先直接解析，如果失败再尝试截取首尾中括号包裹部分
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # 尝试只取第一段以 [ 开始、以 ] 结束的内容
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = response[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    parsed = None

        if parsed is not None and isinstance(parsed, list):
            flag = True
            for item in parsed:
                if 'ref' in item:
                    question_type = item.get('question type', 'Unknown Question Type')
                    question = item.get('question')
                    answer = item.get('answer')
                    refs = item.get('ref', [])
                    if question is None or answer is None:
                        flag = False
                        break
                    if "Unable to answer" in answer or refs == []:
                        question_type = 'Irrelevant Unsolvable Question'
                        refs = []
                    json_obj = {
                        "question type": question_type,
                        "question": question,
                        "answer": answer,
                        "ref": refs,
                    }
                    output.append(json_obj)
                else:
                    question_type = item.get('question type', 'Unknown Question Type')
                    question = item.get('question')
                    answer = item.get('answer')
                    if question is None or answer is None:
                        flag = False
                        break
                    if "Unable to answer" in answer:
                        question_type = 'Irrelevant Unsolvable Question'
                    json_obj = {
                        "question type": question_type,
                        "question": question,
                        "answer": answer,
                    }
                    output.append(json_obj)

            if flag:
                # 本轮解析成功
                return output

        # 走到这里说明本轮解析失败，需要重试或放弃
        attempts += 1
        if attempts >= max_retries:
            # 超过最大重试次数，抛出异常交给上层处理/日志
            raise ValueError("postprocess_en failed to parse valid JSON after max retries")

        # 调用大模型重新生成一次（保留原 system_prompt / user_prompt）
        response = gpt_client.generate([
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        ])[0]

def postprocess_irrelevant_zh(response: str, system_prompt: str, user_prompt: str, model_name: str, domain: str, name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    """
    gpt_client = Client(
        openai_api_key=openai_api_key, 
        model_name=model_name,
    )
    
    output = []
    flag = True
    while True:
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        if "无法回答" in item['答案'] or item['ref'] == []:
                            item['问题类型'] = '无关无解问'
                            item['ref'] = []
                        if domain == 'law':
                            if item['法院名'] != name:
                                flag = False
                                break
                            else:
                                json_obj = {
                                    "法院名": item['法院名'],
                                    "问题类型": item['问题类型'],
                                    "问题": item['问题'],
                                    "答案": item['答案'],
                                    "ref": item['ref'],
                                }
                                output.append(json_obj)
                        if domain == 'medical':
                            if item['医院_病人名'] != name:
                                flag = False
                                break
                            else:
                                json_obj = {
                                    "医院_病人名": item['医院_病人名'],
                                    "问题类型": item['问题类型'],
                                    "问题": item['问题'],
                                    "答案": item['答案'],
                                    "ref": item['ref'],
                                }
                                output.append(json_obj)
                    except KeyError:
                        flag = False
                        break 
            except json.JSONDecodeError:
                flag = False
        
            if not flag:
                response = gpt_client.generate([{"system_prompt": system_prompt, "user_prompt": user_prompt}])[0]
                flag = True
            else:
                return output

def postprocess_irrelevant_en(response: str, system_prompt: str, user_prompt: str, model_name: str, domain: str, name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    """
    gpt_client = Client(
        openai_api_key=openai_api_key, 
        model_name=model_name,
    )
    
    output = []
    flag = True
    while True:
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        if "Unable to answer" in item['answer'] or item['ref'] == []:
                            item['question type'] = 'Irrelevant Unsolvable Question'
                            item['ref'] = []
                        if domain == 'law':
                            json_obj = {
                                "court_name": item['court_name'],
                                "question type": item['question type'],
                                "question": item['question'],
                                "answer": item['answer'],
                                "ref": item['ref'],
                            }
                            output.append(json_obj)
                        if domain == 'medical':
                            if item['hospital_patient_name'] != name:
                                flag = False
                                break
                            else:
                                json_obj = {
                                    "hospital_patient_name": item['hospital_patient_name'],
                                    "question type": item['question type'],
                                    "question": item['question'],
                                    "answer": item['answer'],
                                    "ref": item['ref'],
                                }
                                output.append(json_obj)
                    except KeyError:
                        flag = False
                        break 
            except json.JSONDecodeError:
                flag = False
        
            if not flag:
                response = gpt_client.generate([{"system_prompt": system_prompt, "user_prompt": user_prompt}])[0]
                flag = True
            else:
                return output

def postprocess_reference_check_single_doc_zh(response: str, new_sentences: List, system_prompt: str, user_prompt: str, model_name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    gpt_client = Client(
        openai_api_key=openai_api_key, 
        model_name=model_name,
    )
    
    flag = True
    while True:
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    refs = []
                    try:
                        if item['ref'] == [] and item['问题类型'] != '无关无解问':
                            flag = False
                        else:
                            for ref in item['ref']:
                                refs.append(new_sentences[int(ref) - 1].split('] ')[-1])
                            item['ref'] = refs
                    except KeyError:
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False
        
            if not flag:
                response = gpt_client.generate([{"system_prompt": system_prompt, "user_prompt": user_prompt}])[0]
                flag = True
            else:
                return response

def postprocess_reference_check_single_doc_en(response: str, new_sentences: List, system_prompt: str, user_prompt: str, model_name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    gpt_client = Client(
        openai_api_key=openai_api_key, 
        model_name=model_name,
    )
    
    flag = True
    while True:
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    refs = []
                    try:
                        if item['ref'] == [] and item['question type'] != 'Irrelevant Unsolvable Question':
                            flag = False
                        else:
                            for ref in item['ref']:
                                refs.append(new_sentences[int(ref) - 1].split('] ')[-1])
                            item['ref'] = refs
                    except KeyError:
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False
        
            if not flag:
                response = gpt_client.generate([{"system_prompt": system_prompt, "user_prompt": user_prompt}])[0]
                flag = True
            else:
                return response

def postprocess_reference_check_multi_doc_zh(response: str, domain: str, new_sentences_1: List, new_sentences_2: List, system_prompt: str, user_prompt: str, model_name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    gpt_client = Client(
        openai_api_key=openai_api_key, 
        model_name=model_name,
    )
    
    flag = True
    while True:
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        for ref in item['ref']:
                            if ref['content'] == [] and item['问题类型'] != '无关无解问':
                                flag = False
                                break
                            else:
                                if domain == 'finance':
                                    for sentence in new_sentences_1:
                                        if ref['公司名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    
                                    for sentence in new_sentences_2:
                                        if ref['公司名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break

                                if domain == 'law':
                                    for sentence in new_sentences_1:
                                        if ref['法院名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['法院名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                
                                if domain == 'medical':
                                    for sentence in new_sentences_1:
                                        if ref['医院_病人名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['医院_病人名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                    except KeyError:
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False
        
            if not flag:
                response = gpt_client.generate([{"system_prompt": system_prompt, "user_prompt": user_prompt}])[0]
                flag = True
            else:
                return response

def postprocess_reference_check_multi_doc_en(response: str, domain: str, new_sentences_1: List, new_sentences_2: List, system_prompt: str, user_prompt: str, model_name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    gpt_client = Client(
        openai_api_key=openai_api_key, 
        model_name=model_name,
    )
    print(new_sentences_1)
    print(new_sentences_2)
    
    flag = True
    while True:
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        for ref in item['ref']:
                            if ref['content'] == [] and item['question type'] != 'Irrelevant Unanswerable Question':
                                flag = False
                                break
                            else:
                                if domain == 'finance':
                                    for sentence in new_sentences_1:
                                        if ref['company_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            print(ref['content'])
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['company_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            print(ref['content'])
                                            break

                                if domain == 'law':
                                    for sentence in new_sentences_1:
                                        if ref['court_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            print(ref['content'])
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['court_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            print(ref['content'])
                                            break
                                
                                if domain == 'medical':
                                    for sentence in new_sentences_1:
                                        if ref['hospital_patient_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            print(ref['content'])
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['hospital_patient_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            print(ref['content'])
                                            break
                    except KeyError:
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False
        
            if not flag:
                response = gpt_client.generate([{"system_prompt": system_prompt, "user_prompt": user_prompt}])[0]
                flag = True
            else:
                return response