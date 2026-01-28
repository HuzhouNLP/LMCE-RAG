import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import nltk
import numpy as np
import torch
import transformers
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from lkif.kb_encoder import LKIFEncoder
from lkif.models.kblam_config import LKIFConfig
from lkif.models.llama3_model import LkifLlamaForCausalLM
from lkif.models.phi3_model import LKIFPhi3ForCausalLM
from lkif.utils.data_utils import augment_row, generate_multi_entity_qa
from lkif.utils.eval_utils import (
    instruction_prompts,
    instruction_prompts_multi_entities,
    zero_shot_prompt,
    zero_shot_prompt_multi_entities,
    _format_Q_llama,
    _format_Q_phi3,
    model_prune_format_mapping,
    answer_question,
    softmax,
)
from kblam.utils.train_utils import get_kb_embd

NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    raise RuntimeError(f"未在 {NLTK_DATA_PATH} 找到wordnet，请先运行 nltk.download('wordnet')")

SENTENCE_TRANSFORMER_PATH = "/path/to/lkif/bge-large-en-v1.5"

logging.set_verbosity_warning()

from rouge_score import rouge_scorer


_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


class RougeCompute:
    def compute(self, predictions, references):
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(predictions, references):
            score = _rouge_scorer.score(str(ref), str(pred))
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        return {
            'rouge1': sum(scores['rouge1']) / len(scores['rouge1']),
            'rouge2': sum(scores['rouge2']) / len(scores['rouge2']),
            'rougeL': sum(scores['rougeL']) / len(scores['rougeL'])
        }

rouge = RougeCompute()
print("✓ Rouge评估器已加载")


class LocalEvaluator:
    def __init__(self):
        self.rouge = RougeCompute()
        self.st_model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)
        
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数（处理空字符串情况）"""
        if not predictions:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        predictions = [p.strip() if p.strip() else "empty" for p in predictions]
        references = [r.strip() if r.strip() else "empty" for r in references]
        
        rouge_scores = self.rouge.compute(predictions=predictions, references=references)
        return {
            "rouge-1": rouge_scores['rouge1'],
            "rouge-2": rouge_scores['rouge2'], 
            "rouge-l": rouge_scores['rougeL']
        }
    
    def compute_st_similarity(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """使用Sentence Transformer计算语义相似度（替代BERTScore）"""
        if not predictions:
            return {"st_mean_similarity": 0.0, "st_std_similarity": 0.0}
        # 处理空字符串
        predictions = [p.strip() if p.strip() else "empty" for p in predictions]
        references = [r.strip() if r.strip() else "empty" for r in references]
        
        # 编码文本
        pred_embeds = self.st_model.encode(predictions, convert_to_tensor=True)
        ref_embeds = self.st_model.encode(references, convert_to_tensor=True)
        
        # 计算余弦相似度
        cos_sim = util.cos_sim(pred_embeds, ref_embeds).diag().cpu().numpy()
        return {
            "st_mean_similarity": float(np.mean(cos_sim)),
            "st_std_similarity": float(np.std(cos_sim))
        }

local_evaluator = LocalEvaluator()


REFUSAL_PATTERNS = [
    "i cannot provide",
    "i can't provide",
    "i can’t provide",
    "i cannot give",
    "i can't give",
    "i can’t give",
    "i cannot help",
    "i can't help",
    "i can’t help",
    "i cannot answer",
    "i can't answer",
    "i can’t answer",
    "i cannot comply",
    "i can't comply",
    "i can’t comply",
    "i am unable to",
    "i'm unable to",
    "i am sorry",
    "i'm sorry",
    "sorry, but",
    "i cannot fulfill",
    "i can't fulfill",
    "i can’t fulfill",
    "i do not have access",
    "i don't have access",
    "i do not have the information",
    "i don't have the information",
    "cannot provide information",
    "can't provide information",
    "cannot provide details",
    "can't provide details",
    "i can never",
]


def is_refusal_response(text: str) -> bool:
    normalized = text.lower().replace("’", "'")
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


class KBRetriever:
    def __init__(
        self,
        encoder: LKIFEncoder,
        dataset: List[Dict],
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[np.ndarray] = None,
        dataset_type: str = "law",
    ):
        self.encoder = encoder
        self.dataset = dataset
        self.dataset_type = dataset_type
        
        if precomputed_embed_keys_path is not None:
            self.key_embds = np.load(precomputed_embed_keys_path).astype("float32")
        else:
            self.key_embds = None
        if precomputed_embed_values_path is not None:
            self.value_embds = np.load(precomputed_embed_values_path).astype("float32")
        else:
            self.value_embds = None

        if precomputed_embed_keys_path is not None:
            assert len(dataset) == len(self.key_embds)

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices):
        if self._use_cached_embd():
            return get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            # 根据数据集类型提取不同的字段
            if self.dataset_type == "law":
                # 对于法律数据集，key是query.content，value是ground_truth.references
                key_texts = []
                value_texts = []
                for idx in batch_indices:
                    data = self.dataset[idx]
                    key_texts.append(data.get("query", {}).get("content", ""))
                    
                    refs = data.get("ground_truth", {}).get("references", [])
                    if isinstance(refs, list) and refs:
                        if all(isinstance(ref, str) for ref in refs):
                            refs_str = "\n".join(refs)
                        elif all(isinstance(ref, dict) for ref in refs):
                            refs_str = "\n".join([ref.get('content', str(ref)) for ref in refs])
                        else:
                            refs_str = "\n".join([str(ref) for ref in refs])
                    else:
                        refs_str = ""
                    value_texts.append(refs_str)
                
                kb_dict = [{"key_string": k, "description": v} for k, v in zip(key_texts, value_texts)]
            else:
                # 原始游戏数据集格式
                kb_dict = [self.dataset[idx] for idx in batch_indices]
            
            return get_kb_embd(self.encoder, batch_indices, kb_dict=kb_dict)


def perform_eval(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    encoder_model_spec: str,
    kb_config: LKIFConfig,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    topk_size: int = -1,
    multi_entites: int = -1,
    remove_sorry: bool = False,
    dataset_type: str = "law",
    max_new_tokens: int = 300,
):
    np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    
    # 根据数据集类型提取不同字段
    if dataset_type == "law":
        key_str = [row.get("query", {}).get("content", "") for row in test_kb]
        value_str = []
        for row in test_kb:
            refs = row.get("ground_truth", {}).get("references", [])
            if isinstance(refs, list) and refs:
                if all(isinstance(ref, str) for ref in refs):
                    refs_str = "\n".join(refs)
                elif all(isinstance(ref, dict) for ref in refs):
                    refs_str = "\n".join([ref.get('content', str(ref)) for ref in refs])
                else:
                    refs_str = "\n".join([str(ref) for ref in refs])
            else:
                refs_str = ""
            value_str.append(refs_str)

    prompt_strs = ""
    for question, reference in zip(key_str, value_str):
        prompt_strs += f'Question: "{question}"\nReference: "{reference}"\n\n'  

    kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

    model_outputs = []
    answers = []
    full_outputs = []
    # answer_question
    subset_size = min(400, len(test_kb))  # 控制测试规模
    total_examples = subset_size
    skipped_count = 0

    for idx, row in enumerate(tqdm(test_kb[:subset_size])):
        if multi_entites == -1:
            if dataset_type == "law":
                Q = row.get("query", {}).get("content", "")
                answer = row.get("ground_truth", {}).get("content", "")
      
        else:
            kb_subset_idx = np.random.randint(0, len(test_kb), multi_entites)
            if dataset_type == "law":
                # 对于法律数据集的多实体处理
                names = [test_kb[i].get("domain", "") for i in kb_subset_idx]
                desc_types = [test_kb[i].get("language", "") for i in kb_subset_idx]
                descriptions = [test_kb[i].get("ground_truth", {}).get("content", "") for i in kb_subset_idx]
            else:
                names = [test_kb[i]["name"] for i in kb_subset_idx]
                desc_types = [test_kb[i]["description_type"] for i in kb_subset_idx]
                descriptions = [test_kb[i]["description"] for i in kb_subset_idx]
            
            Q, A = generate_multi_entity_qa(names, desc_types, descriptions)
            answer = A

        if eval_mode == "kb":
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                kb_config=kb_config,
                max_new_tokens=max_new_tokens,
            )
            # 安全地分割输出，防止IndexError
            split_output = model_output.split(Q)
            if len(split_output) > 1:
                model_output = split_output[1]
            else:
                model_output = model_output  # 如果分割失败，使用原始输出
        elif eval_mode == "icl":
            if multi_entites != -1:
                ins_prompt = instruction_prompts_multi_entities
            else:
                ins_prompt = instruction_prompts
            model_output = answer_question(
                tokenizer,
                model,
                ins_prompt + prompt_strs + Q,
                kb=None,
                kb_config=kb_config,
                max_new_tokens=max_new_tokens,
            ).split(Q)[2]
        elif eval_mode == "zeroshot":
            if multi_entites != -1:
                ins_prompt = zero_shot_prompt_multi_entities
            else:
                ins_prompt = zero_shot_prompt
            model_output = answer_question(
                tokenizer,
                model,
                ins_prompt + Q,
                kb=None,
                kb_config=kb_config,
                max_new_tokens=max_new_tokens,
            ).split(Q)[1]
        
        if remove_sorry and is_refusal_response(model_output):
            skipped_count += 1
            continue
        full_outputs.append((model_output, answer))
        
        # 每处理10个样本清理一次显存，防止内存溢出
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
        # 提取预测结果和答案
        if multi_entites == -1:
            if dataset_type == "law":
                # 对于法律数据集，直接使用答案内容
                answers.append(row.get("ground_truth", {}).get("content", ""))
                model_outputs.append(model_output.strip())
            else:
                # 原始游戏数据集的处理逻辑
                pattern = r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)'
                match = re.search(pattern, model_output)
                answers.append(row["description"])
                if match:
                    model_output = match.group(1)
                model_outputs.append(model_output)
        else:
            pattern = r"(?:is|are) (.*?)(?:\.|;)"
            matches = re.findall(pattern, model_output)
            model_output = "; ".join(matches)
            model_outputs.append(model_output)
            answers.append(";".join(re.findall(r"(?:is|are) (.*?);", answer)))

    print(f"KB size: {kb_size}, mode: {eval_mode}, dataset_type: {dataset_type}")
    
    if model_outputs:
        rouge_scores = local_evaluator.compute_rouge(model_outputs, answers)
        print("ROUGE Scores:", rouge_scores)
        
        st_scores = local_evaluator.compute_st_similarity(model_outputs, answers)
        print("Sentence Transformer Similarity Scores:", st_scores)
    else:
        rouge_scores = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        st_scores = {"st_mean_similarity": 0.0, "st_std_similarity": 0.0}
        print("ROUGE Scores:", rouge_scores)
        print("Sentence Transformer Similarity Scores:", st_scores)

    # 整合结果
    results_dict = {**rouge_scores, **st_scores}

    mem_bytes = torch.cuda.max_memory_reserved("cuda")
    mem_gb = mem_bytes / (1024 ** 3)
    results_dict["mem_cost_gb"] = mem_gb
    results_dict["total_examples"] = total_examples
    results_dict["evaluated_examples"] = len(model_outputs)
    results_dict["skipped_examples"] = skipped_count

    results = ""
    if skipped_count:
        results += f"Skipped {skipped_count} refusal-style responses before scoring.\n-------\n"
    for a, A in full_outputs:
        results += f"Model output: {a}\nTrue answer: {A}\n-------\n"
    if eval_mode == "kb":
        eval_mode = encoder_model_spec + eval_mode

    return results, results_dict


def perform_eval_refusal(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    kb_config: Optional[LKIFConfig] = None,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    outlier_ratio: float = 0.2,
    topk_size: int = -1,
    question_size: int = 100,
    dataset_type: str = "law",
    max_new_tokens: int = 300,
):
    instruction_prompts = (
    """ Please follow the steps below to answer the final question strictly based on the provided Knowledge Base (composed of multiple Question-Reference pairs):
Step 1: Understand the Knowledge Base Structure
The Knowledge Base provided below contains multiple Question-Reference pairs (each pair consists of a "Question" and its corresponding "Reference"—the Reference is the authoritative information used to answer that Question). You need to first read and memorize all Question-Reference pairs in the Knowledge Base.
Step 2: Match the Final Question to the Knowledge Base
After reading the Knowledge Base.You need to:
Find the most relevant Question in the Knowledge Base (i.e., the Question in the Knowledge Base that has the same core meaning or directly corresponds to the final question);
Extract the Reference corresponding to this matched Question (this Reference is the only valid information source for answering the final question—do not use any information outside the Knowledge Base).
Step 3: Step 3: Generate the answer. Use the matching reference to answer the question. The final answer only needs the answer part and should not output any redundant content.
     The example is as follows
"Question": "According to the court judgment of Oakwood, Orchard, Court, what is the defendant Li Jiawei's occupation?",
"References": 
"At the time of the commission of the offenses and prior to his apprehension, the defendant Li Jiawei was employed in the capacity of a Senior Financial Analyst at Global Finance Group."
Your output results strictly adhere to the following format："Senior Financial Analyst at Global Finance Group."
     """
    ' if relevant information cannot be found in the text, please respond "I am sorry I cannot find relevant information in the KB".'
    )
    zero_shot_prompt = """
    Please answer the question in a very compact manner """

    np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    
    # 根据数据集类型提取不同字段
    if dataset_type == "law":
        key_str = [row.get("query", {}).get("content", "") for row in test_kb]
        value_str = []
        for row in test_kb:
            refs = row.get("ground_truth", {}).get("references", [])
            if isinstance(refs, list) and refs:
                if all(isinstance(ref, str) for ref in refs):
                    refs_str = "\n".join(refs)
                elif all(isinstance(ref, dict) for ref in refs):
                    refs_str = "\n".join([ref.get('content', str(ref)) for ref in refs])
                else:
                    refs_str = "\n".join([str(ref) for ref in refs])
            else:
                refs_str = ""
            value_str.append(refs_str)
    else:
        key_str = [row["key_string"] for row in test_kb]
        value_str = [row["description"] for row in test_kb]
    
    prompt_strs = ""
    for question, reference in zip(key_str, value_str):
        prompt_strs += f'Question: "{question}"\nReference: "{reference}"\n\n'  


    kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

    model_outputs = []
    answers = []
    # 准备问题集（包含异常值）
    outlier_idx = np.arange(len(kb_retriever.dataset))
    outlier_idx = outlier_idx[~np.isin(outlier_idx, kb_idx)]
    np.random.shuffle(outlier_idx)
    question_size = min(kb_size, question_size)
    outlier_idx = outlier_idx[: int(question_size * outlier_ratio)]
    test_kb = test_kb[: int(question_size * (1 - outlier_ratio))] + [
        kb_retriever.dataset[idx] for idx in outlier_idx
    ]
    change_point = int(question_size * (1 - outlier_ratio))
    
    for i, row in tqdm(enumerate(test_kb)):
        if dataset_type == "law":
            Q = row.get("query", {}).get("content", "")
        else:
            Q = row["Q"]
            
        if eval_mode == "kb":
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                kb_config=kb_config,
                max_new_tokens=max_new_tokens,
            ).split(Q)[1]

        elif eval_mode == "icl":
            model_output = answer_question(
                tokenizer,
                model,
                instruction_prompts + prompt_strs + Q,
                kb=None,
                kb_config=kb_config,
                max_new_tokens=max_new_tokens,
            ).split(Q)[2]
        elif eval_mode == "zeroshot":
            model_output = answer_question(
                tokenizer,
                model,
                zero_shot_prompt + Q,
                kb=None,
                kb_config=kb_config,
                max_new_tokens=max_new_tokens,
            ).split(Q)[1]
        model_outputs.append(model_output)
        if i < change_point:
            if dataset_type == "law":
                answers.append(row.get("ground_truth", {}).get("content", ""))
            else:
                answers.append(row["description"])
        else:
            answers.append("Cannot find relevant information in the KB")
    
    true_label = [0] * change_point + [1] * int(question_size * outlier_ratio)
    prediction = [int("sorry" in model_output.lower()) for model_output in model_outputs]
    print(f"KB size: {kb_size}, mode: {eval_mode}, outlier ratio: {outlier_ratio}, dataset_type: {dataset_type}")
    
    results = ""
    for a, A in zip(model_outputs, answers):
        results += f"Model output: {a}\nTrue answer: {A}\n-------\n"
    return results, np.array([prediction, true_label])


parser = argparse.ArgumentParser(description="Evaluation script (离线版本)")

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--dataset_dir", type=str, help="Directory containing the dataset"
)
parent_parser.add_argument(
    "--encoder_dir", type=str, help="Directory containing the encoder model"
)
parent_parser.add_argument(
    "--encoder_spec",
    type=str,
    default="OAI",
    help="Specification for the encoder model",
)
parent_parser.add_argument(
    "--fancy_instruction",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use fancy instructions",
)
parent_parser.add_argument(
    "--kb_layer_frequency",
    type=int,
    default=3,
    help="Frequency of knowledge base layers",
)
parent_parser.add_argument(
    "--kb_scale_factor",
    type=int,
    default=None,
    help="Scaling factor for knowledge base",
)
parent_parser.add_argument(
    "--kb_size", type=int, default=200, help="Size of the knowledge base"
)
parent_parser.add_argument(
    "--llm_base_dir",
    type=str,
    help="本地LLM模型路径（如Llama-3.2-1B-Instruct）",
)
parent_parser.add_argument(
    "--llm_type",
    type=str,
    default="phi3",
    choices=["llama3", "phi3"],
    help="Type of language model to use",
)
parent_parser.add_argument(
    "--model_dir", type=str, help="本地模型 checkpoint 路径"
)
parent_parser.add_argument("--save_dir", type=str, help="Directory to save outputs")
parent_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
parent_parser.add_argument(
    "--test_dataset", type=str, help="测试数据集路径（KV格式JSON）"
)
parent_parser.add_argument(
    "--precomputed_embed_keys_path", type=str, help="预计算的key嵌入路径"
)
parent_parser.add_argument(
    "--precomputed_embed_values_path",
    type=str,
    help="预计算的value嵌入路径",
)
parent_parser.add_argument(
    "--query_head_path", type=str, default="", help="KB head加载路径"
)
parent_parser.add_argument(
    "--dataset_type",
    type=str,
    default="game",
    choices=["game", "law"],
    help="Type of dataset (game or law)",
)

# 子命令解析器
subparsers = parser.add_subparsers(dest="command", required=True)

# generation子命令
gen_parser = subparsers.add_parser(
    "generation", parents=[parent_parser], help="Evaluate generation"
)
gen_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
)
gen_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="generation_results",
    help="实验配置名称",
)
gen_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="KB token层频率",
)
gen_parser.add_argument(
    "--multi_entites",
    type=int,
    default=-1,
    help="多实体数量（-1表示不限）",
)
gen_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="使用无异常值训练的checkpoint",
)
gen_parser.add_argument(
    "--remove_sorry",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='过滤包含"sorry"的回答',
)
gen_parser.add_argument(
    "--topk_size", type=int, default=-1, help="top-k选择大小（-1表示全部）"
)
gen_parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=300,
    help="LLM 生成的新 token 上限（长答案可以适当调大）",
)


# accuracy子命令
acc_parser = subparsers.add_parser(
    "accuracy", parents=[parent_parser], help="Evaluate accuracy"
)

acc_parser.add_argument(
    "--attn_save_dir", type=str, default="", help="注意力掩码保存目录"
)
acc_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="accuracy_results",
    help="实验配置名称",
)
acc_parser.add_argument(
    "--fancy_question",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="启用复杂问题格式",
)
acc_parser.add_argument(
    "--log_save_dir", type=str, help="准确率结果保存目录"
)
acc_parser.add_argument(
    "--test_batch_size", type=int, default=50, help="测试批次大小"
)
acc_parser.add_argument(
    "--use_shift_match",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="启用偏移匹配",
)

# acc_results子命令
acc_results_parser = subparsers.add_parser(
    "acc_results", parents=[acc_parser], help="运行准确率评估", add_help=False
)


# refusal子命令
ref_parser = subparsers.add_parser(
    "refusal", parents=[parent_parser], help="Evaluate refusal"
)
ref_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="评估模式",
)
ref_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="refusal_results",
    help="实验配置名称",
)
ref_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="KB token层频率",
)
ref_parser.add_argument(
    "--multi_entites",
    type=int,
    default=-1,
    help="多实体数量",
)
ref_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="使用无异常值训练的checkpoint",
)
ref_parser.add_argument(
    "--remove_sorry",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='过滤"sorry"回答',
)
ref_parser.add_argument(
    "--topk_size", type=int, default=-1, help="top-k选择大小"
)
ref_parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=300,
    help="LLM 生成的新 token 上限（用于拒答评估）",
)

# standard子命令
basic_parser = subparsers.add_parser(
    "standard", parents=[parent_parser], help="Evaluate basic performance"
)
basic_parser.add_argument(
    "--attn_summary_save_dir",
    type=str,
    default="",
    help="注意力摘要保存目录",
)
basic_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="评估模式",
)
basic_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="basic_results",
    help="实验配置名称",
)
basic_parser.add_argument(
    "--exp_config_str", type=str, help="实验配置字符串"
)
basic_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="KB token层频率",
)
basic_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="使用无异常值训练的checkpoint",
)
basic_parser.add_argument(
    "--sample_size", default=5, type=int, help="样本数量"
)
basic_parser.add_argument(
    "--subset_size", default=100, type=int, help="子集大小"
)
basic_parser.add_argument(
    "--topk_size", type=int, default=-1, help="top-k选择大小"
)


def eval_generate():
    """评估生成能力"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    eval_mode = args.eval_mode
    exp_config = args.exp_config_name
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    query_head_path = args.query_head_path
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    # 兼容 JSON 数组和 JSONL，两种格式都支持
    dataset_path = os.path.join(dataset_dir, test_dataset)
    with open(dataset_path, "r", encoding="utf-8") as f:
        try:
            dataset = json.load(f)
            if isinstance(dataset, dict):
                dataset = [dataset]
        except json.JSONDecodeError:
            f.seek(0)
            lines = [ln.strip() for ln in f if ln.strip()]
            dataset = [json.loads(ln) for ln in lines]

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
        dataset_type=args.dataset_type,
    )

    gen_results, score_results = perform_eval(
        model,
        tokenizer,
        kb_retriever,
        encoder_model_spec,
        kb_config,
        eval_mode,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        multi_entites=args.multi_entites,
        remove_sorry=args.remove_sorry,
        dataset_type=args.dataset_type,
        max_new_tokens=args.max_new_tokens,
    )

    (Path(args.save_dir) / exp_config).mkdir(exist_ok=True, parents=True)
    write_to_json(score_results, Path(args.save_dir) / f"{exp_config}.json")
    print(score_results)
    with open(os.path.join(args.save_dir, exp_config + ".txt"), "w") as text_file:
        text_file.write(gen_results)


def _prepare_models(
    encoder_spec,
    encoder_path,
    llm_type,
    llm_base_dir,
    model_path,
    query_head_path,
    kb_layer_frequency,
    kb_scale_factor,
):

    tokenizer = AutoTokenizer.from_pretrained(
        llm_base_dir, 
        trust_remote_code=True, 
        padding_side="left",
        local_files_only=True  # 强制使用本地文件，不联网
    )
    tokenizer.pad_token = "^"

    if llm_type == "llama3":
        if query_head_path:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                local_files_only=True  # 强制本地加载
            )
            model.load_query_head(query_head_path)
        else:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                local_files_only=True  # 强制本地加载
            )
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True  # 强制本地加载
        )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    kb_config = LKIFConfig(
        sep_query_head=True,
        kb_layer_frequency=kb_layer_frequency,
        kb_scale_factor=kb_scale_factor,
    )

    # 加载编码器
    encoder = LKIFEncoder(
        encoder_name=encoder_spec,  # 移除.upper()，保持原始路径
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size
        * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
    )
    encoder.load_state_dict(torch.load(encoder_path))
    return tokenizer, encoder, model, kb_config


def eval_accuracy(
    tokenizer,
    kb_retriever,
    model,
    dataset,
    exp_config,
    fancy_question,
    kb_config,
    kb_size,
    llm_type,
    test_batch_size,
    save_dir,
    attn_save_dir,
    dataset_type="game",
):
    """评估准确率"""
    if kb_size == len(dataset):
        dataset_subset_idx = range(len(dataset))
    elif kb_size > len(dataset):
        raise IndexError(
            f"KB大小 {kb_size} 大于数据集大小 {len(dataset)}"
        )
    else:
        dataset_subset_idx = np.random.choice(len(dataset), kb_size, replace=False)

    dataset_subset = [dataset[i] for i in dataset_subset_idx]
    kb_embedding_real = kb_retriever.get_key_embeddings(dataset_subset_idx)

    format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}

    if not fancy_question:
        if dataset_type == "law":
            input_strs_gen = (dataset_subset[i].get("query", {}).get("content", "") for i in range(test_batch_size))
        else:
            input_strs_gen = (dataset_subset[i]["Q"] for i in range(test_batch_size))
    else:
        input_strs_gen = (augment_row(dataset_subset[i], dataset_type) for i in range(test_batch_size))
    input_strs = [format_func_map[llm_type](ex) for ex in input_strs_gen]

    tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
        "cuda"
    )
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb_embedding_real,
            max_new_tokens=60,
            tokenizer=tokenizer,
            output_attentions=True,
            save_attention_weights=True,
            kb_config=kb_config,
            attention_save_loc=attn_save_dir,
            attention_file_base_name=exp_config,
        )
        outputs = tokenizer.batch_decode(outputs.squeeze(), skip_special_tokens=False)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    with open(save_path / f"{exp_config}_acc.txt", "w+") as text_file:
        for output in outputs:
            output_string = output.strip("^")
            text_file.write(f"{str(output_string)}\n")

    accs = []
    with torch.autograd.no_grad():
        for idx in range(0, 32, kb_config.kb_layer_frequency):
            weight = np.load(os.path.join(attn_save_dir, f"{exp_config}_{idx}.npy"))
            weight = weight[..., :kb_size]
            label = np.arange(test_batch_size)
            weight = weight.reshape(test_batch_size, -1, kb_size)
            acc = (weight.sum(1).argmax(1) == label).mean()
            top_5_predictions = torch.topk(torch.from_numpy(weight.sum(1)), 5, dim=1)[1]
            top_5_acc = (top_5_predictions.numpy() == label[:, None]).any(1).mean()
            if idx == 15:
                print(f"ACC & TOP 5 ACC: {idx} {(acc, top_5_acc)}")
                print(f"min: {np.min(weight)}  max: {np.max(weight)}")
            accs.append(
                {
                    "idx": idx,
                    "acc": float(acc),
                    "top5acc": float(top_5_acc),
                }
            )

    np.save(
        save_path / f"{exp_config}_acc.npy",
        np.array([(a["acc"], a["top5acc"]) for a in accs]),
    )

    return accs


def eval_accuracy_cli():
    """命令行调用准确率评估"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_path = args.encoder_dir
    encoder_spec = args.encoder_spec
    exp_config = args.exp_config_name
    fancy_question = args.fancy_question
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    test_batch_size = args.test_batch_size
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    query_head_path = args.query_head_path
    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )
    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
        dataset_type=args.dataset_type,
    )

    eval_accuracy(
        tokenizer,
        kb_retriever,
        model,
        dataset,
        exp_config,
        fancy_question,
        kb_config,
        kb_size,
        llm_type,
        test_batch_size,
        args.log_save_dir,
        args.attn_save_dir,
        dataset_type=args.dataset_type,
    )


def write_to_json(
    data: Any, filepath: str, indent: int = 4, encoding: str = "utf-8"
) -> bool:
    """写入JSON文件"""
    try:
        file_path = Path(filepath)
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=True,
                default=str,
            )
        return True
    except Exception as e:
        print(f"写入JSON错误: {str(e)}")
        return False


def run_accuracy_evalution():
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_path = args.encoder_dir
    encoder_spec = args.encoder_spec
    exp_config = args.exp_config_name
    fancy_question = args.fancy_question
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    test_dataset = args.test_dataset

    query_head_path = args.query_head_path
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))
    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
        dataset_type=args.dataset_type,
    )

    xs = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    accuracy_results = []
    for x in xs:
        print(f"kb_size {x}")

        accs = eval_accuracy(
            tokenizer,
            kb_retriever,
            model,
            dataset,
            exp_config,
            fancy_question,
            kb_config,
            x,
            llm_type,
            min(x, 200),
            args.log_save_dir,
            args.attn_save_dir,
            dataset_type=args.dataset_type,
        )
        shutil.rmtree(args.attn_save_dir, ignore_errors=True)
        os.makedirs(args.attn_save_dir, exist_ok=True)
        accuracy_results.append({"kb_size": x, "accuracy_results": accs})
    write_to_json(
        accuracy_results, os.path.join(args.log_save_dir, "accuracy_results.json")
    )


def eval_refusal():
    """评估拒绝回答能力"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    eval_mode = args.eval_mode
    exp_config = args.exp_config_name
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path
    query_head_path = args.query_head_path

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
        dataset_type=args.dataset_type,
    )

    gen_results, refusal_results = perform_eval_refusal(
        model,
        tokenizer,
        kb_retriever,
        eval_mode=eval_mode,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        kb_config=kb_config,
        dataset_type=args.dataset_type,
        max_new_tokens=args.max_new_tokens,
    )

    np.save(os.path.join(args.save_dir, "OutLierTest" + exp_config), refusal_results)
    with open(
        os.path.join(args.save_dir, "OutLierTest" + exp_config + ".txt"), "w"
    ) as text_file:
        text_file.write(gen_results)


def eval():
    """标准评估流程"""
    args = parser.parse_args()

    attn_summary_save_dir = args.attn_summary_save_dir
    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    exp_config_str = args.exp_config_str
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    output_dir = args.save_dir
    sample_size = args.sample_size
    seed = args.seed
    subset_size = args.subset_size
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path
    query_head_path = args.query_head_path
    sep_query_head = True
    actual_kb_token_layer_frequency = 3

    if kb_size == -1:
        kb_size = None

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    if sep_query_head:
        print("使用独立的KB查询头!")

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["ATTN_SAVE_DIR"] = output_dir
    os.environ["EVAL_MODE"] = "1"

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    for param in model.parameters():
        param.requires_grad = False

    # 初始化编码器
    encoder = KBEncoder(
        encoder_name=encoder_model_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // actual_kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=torch.device("cuda"),
    )
    encoder.load_state_dict(torch.load(encoder_path))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
        dataset_type=args.dataset_type,
    )
    no_kb_predictions = []
    predictions = []
    answer = []

    for _ in range(sample_size):
        print("******")
        dataset_subset_idx = np.random.choice(len(dataset), subset_size, replace=False)
        dataset_subset = [dataset[i] for i in dataset_subset_idx]
        encoder.eval()
        with torch.autograd.no_grad():
            kb_embedding_real = kb_retriever.get_key_embeddings(dataset_subset_idx)
            kb_embedding_key, kb_embedding_val = kb_embedding_real
            kb_embedding_real = (kb_embedding_key, kb_embedding_val)

        format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}

        input_strs = [
            format_func_map[llm_type](dataset_subset[i]["Q"])
            for i in range(subset_size)
        ]

        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
            "cuda"
        )
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        kb_embedding_real = (kb_embedding_real[0], kb_embedding_real[1])

        config_str = f"{exp_config_str}__kb_{subset_size}__seed_{seed}"
        with torch.autograd.no_grad():
            outputs_no_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=None,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=False,
                kb_config=kb_config,
            )

            outputs_true_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=kb_embedding_real,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=True,
                save_attention_weights=True,
                attention_save_loc=output_dir,
                attention_file_base_name=config_str,
                kb_config=kb_config,
            )
        print("解码中")
        outputs_no_kb = tokenizer.batch_decode(outputs_no_kb, skip_special_tokens=False)
        outputs_true_kb = tokenizer.batch_decode(
            outputs_true_kb, skip_special_tokens=False
        )
        print("KB内容:")
        for i in range(subset_size):
            print(
                "{} : {}".format(
                    dataset_subset[i]["name"], dataset_subset[i]["description"]
                )
            )

        for m in model_prune_format_mapping:
            if isinstance(model, m):
                prune_str = model_prune_format_mapping[m]

        print("------------------")
        for i in range(subset_size):
            print("带KB的输出", prune_str(outputs_true_kb[i]))
            print("真实答案: ", dataset_subset[i]["A"])
            no_kb_predictions.append(
                prune_str(outputs_no_kb[i]).split(dataset_subset[i]["Q"])[1]
            )
            predictions.append(
                prune_str(outputs_true_kb[i]).split(dataset_subset[i]["Q"])[1]
            )
            answer.append(dataset_subset[i]["A"])
            print("--------------------")
        print("******")

    rouge_scores = local_evaluator.compute_rouge(predictions, answer)
    np.savez(
        os.path.join(attn_summary_save_dir, f"{config_str}_rouge.npy"), **rouge_scores
    )

    rouge_scores_no_kb = local_evaluator.compute_rouge(no_kb_predictions, answer)
    np.savez(
        os.path.join(attn_summary_save_dir, f"{config_str}_rouge_no_kb.npy"),** rouge_scores_no_kb,
    )

    # 计算语义相似度
    st_scores = local_evaluator.compute_st_similarity(predictions, answer)
    np.savez(
        os.path.join(attn_summary_save_dir, f"{config_str}_st_scores.npy"), **st_scores
    )

    # 注意力分析
    ranges = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 32)]

    save_dir = output_dir
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    accs, confidences = [], []
    for left, right in ranges:
        weights = []
        kb_size = subset_size
        for idx in range(32)[left:right]:
            if idx % 3 == 0:
                weight = np.load(os.path.join(save_dir, f"{config_str}_{idx}.npy"))
                weights.append(weight[..., :kb_size].reshape(kb_size, -1, kb_size))
        print(f"注意力权重数量: {len(weights)}")
        weights = np.stack(weights)
        weights = weights.transpose(1, 0, 2, 3).reshape(kb_size, -1, kb_size)
        acc = (weights.sum(1).argmax(1) == np.arange(kb_size)).mean()
        top_5_predictions = torch.topk(torch.from_numpy(weights.sum(1)), 5, dim=1)[1]
        top_5_acc = (
            (top_5_predictions == torch.arange(kb_size)[:, None]).any(1).float().mean()
        )
        accs.append((acc, top_5_acc))
        confidence = softmax(weights.mean(1), -1).max()
        confidences.append(confidence)
    np.save(
        os.path.join(attn_summary_save_dir, f"{config_str}_acc.npy"), np.array(accs)
    )
    np.save(
        os.path.join(attn_summary_save_dir, f"{config_str}_conf.npy"),
        np.array(confidences),
    )


def main():
    args = parser.parse_args()
    print("参数配置:", args)
    
    # 根据子命令执行对应评估
    if args.command == "generation":
        eval_generate()
    elif args.command == "accuracy":
        eval_accuracy_cli()
    elif args.command == "acc_results":
        run_accuracy_evalution()
    elif args.command == "refusal":
        eval_refusal()
    elif args.command == "standard":
        eval()
    else:
        raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
