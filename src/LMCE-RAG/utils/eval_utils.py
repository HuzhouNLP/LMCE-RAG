from typing import Optional

import numpy as np
import torch
import transformers

from lkif.models.kblam_config import LKIFConfig
from lkif.models.llama3_model import LkifLlamaForCausalLM
from lkif.models.phi3_model import LKIFPhi3ForCausalLM
from lkif.models.Qwen_modle import LKIFQwen2ForCausalLM

instruction_prompts =""" Please follow the steps below to answer the final question strictly based on the provided Knowledge Base (composed of multiple Question-Reference pairs):
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
Your output results strictly adhere to the following format："Answer":"Senior Financial Analyst at Global Finance Group."
     """

instruction_prompts_multi_entities = """
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
"""

zero_shot_prompt = """
Please answer the question in a very compact manner """

zero_shot_prompt_multi_entities = """
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
"""


def _prune_for_llama(S: str) -> str:
    S = S.replace("<|eot_id|>", "")
    S = S.replace("<|start_header_id|>assistant<|end_header_id|>", "\n\n")
    S = S.replace("<|start_header_id|>user<|end_header_id|>", "")
    S = S.replace("<|end_of_text|>", "")
    return S


def _prune_for_phi3(S: str) -> str:
    S = S.replace("<|end|>", "")
    S = S.replace("<|assistant|>", "\n\n")
    S = S.replace("<|user|>", "")
    return S


def _prune_for_qwen(S: str) -> str:
    S = S.replace("<|im_end|>", "")
    S = S.replace("<|im_start|>assistant", "\n\n")
    S = S.replace("<|im_start|>user", "")
    return S


def softmax(x: np.array, axis: int) -> np.array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def _format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )


def _format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"


def _format_Q_qwen(Q: str):
    return "<|im_start|>user\n" + Q + "<|im_end|>\n" + "<|im_start|>assistant\n"


model_question_format_mapping = {
    LkifLlamaForCausalLM: _format_Q_llama,
    LKIFPhi3ForCausalLM: _format_Q_phi3,
    LKIFQwen2ForCausalLM: _format_Q_qwen,
}
model_prune_format_mapping = {
    LkifLlamaForCausalLM: _prune_for_llama,
    LKIFPhi3ForCausalLM: _prune_for_phi3,
    LKIFQwen2ForCausalLM: _prune_for_qwen,
}


def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    model: LKIFPhi3ForCausalLM | LkifLlamaForCausalLM | LKIFQwen2ForCausalLM,
    Q: str,
    kb=None,
    kb_config: Optional[LKIFConfig] = None,
    attention_save_loc: Optional[str] = None,
    save_attention_weights: bool = False,
    attention_file_base_name: Optional[str] = None,
    topk_size: int = 100,  # 目前未使用，仅为兼容旧接口
    max_new_tokens: int = 300,
):
    for m in model_question_format_mapping:
        if isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)
    tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        try:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=kb,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                output_attentions=True,
                kb_config=kb_config,
                pad_token_id=tokenizer.eos_token_id,
                save_attention_weights=save_attention_weights,
                attention_file_base_name=attention_file_base_name,
                attention_save_loc=attention_save_loc,
                do_sample=False,
                num_beams=1,
                temperature=None,
            ).squeeze()
        except RuntimeError as e:
            if "probability tensor contains" in str(e):
                print(f"🚨 Numerical instability detected, falling back to basic generation")
                # 降级到最基本的生成模式
                fallback_max_new_tokens = min(max_new_tokens, 128)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    max_new_tokens=fallback_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                ).squeeze()
            else:
                raise e
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)
    print(f"🔍 Raw model output: '{outputs}'")

    for m in model_prune_format_mapping:
        if isinstance(model, m):
            pruned_output = model_prune_format_mapping[m](outputs)
            print(f"🔍 Pruned output: '{pruned_output}'")
    return pruned_output
