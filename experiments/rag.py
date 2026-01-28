import json
from dataclasses import dataclass
from typing import List, Tuple

import argparse
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


"""
A minimal local RAG demo:
- Knowledge items: each entry's `ground_truth.references` in QA-summary.jsonl
- Encoder: /path/to/lkif/bge-large-en-v1.5
- LLM: /path/to/lkif/Meta-Llama-3-8B-Instruct

Usage (run on Linux):
	python 11.py
	Then enter your question in English when prompted.
"""


KB_PATH = "/path/to/lkif/date-synthesis/qar_generation/results/QA-summary.jsonl"
ENCODER_PATH = "/path/to/lkif/bge-large-en-v1.5"
LLM_PATH = "/path/to/lkif/Meta-Llama-3-8B-Instruct"
TOP_K = 3
MAX_NEW_TOKENS = 256


@dataclass
class KBEntry:
	query: str
	reference: str
	answer: str


def load_kb(path: str) -> List[KBEntry]:
	entries: List[KBEntry] = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			q = obj["query"]["content"]
			ref = obj["ground_truth"]["references"]
			ans = obj["ground_truth"]["content"]
			entries.append(KBEntry(query=q, reference=ref, answer=ans))
	return entries


def deduplicate_kb_by_reference(entries: List[KBEntry]) -> List[KBEntry]:
	"""按 reference 文本去重，保留每个唯一文档的一条代表样本。

	这样可以避免同一篇判决在检索结果中重复出现多次、且得分完全相同。
	"""
	seen_refs = set()
	unique_entries: List[KBEntry] = []
	for e in entries:
		if e.reference in seen_refs:
			continue
		seen_refs.add(e.reference)
		unique_entries.append(e)
	return unique_entries


def build_kb_embeddings(model: SentenceTransformer, entries: List[KBEntry]) -> torch.Tensor:
	texts = [e.reference for e in entries]
	embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
	return embeddings


def retrieve(
	encoder: SentenceTransformer,
	kb_embeddings: torch.Tensor,
	entries: List[KBEntry],
	query: str,
	top_k: int,
) -> List[Tuple[float, KBEntry]]:
	q_emb = encoder.encode(query, convert_to_tensor=True)
	scores = util.cos_sim(q_emb, kb_embeddings)[0]
	top_results = torch.topk(scores, k=min(top_k, len(entries)))
	results: List[Tuple[float, KBEntry]] = []
	for score, idx in zip(top_results.values, top_results.indices):
		results.append((float(score), entries[int(idx)]))
	return results


def build_prompt(
	tokenizer: AutoTokenizer,
	user_query: str,
	retrieved: List[Tuple[float, KBEntry]],
) -> str:
	"""根据检索结果构造提示词。

	只使用检索到的第 1 篇文档，保留其全文，不做截断。
	"""
	context_parts = []
	if retrieved:
		score, entry = retrieved[0]
		context_parts.append(f"[DOC 1 | score={score:.4f}]\n{entry.reference}")
	context = "\n\n".join(context_parts)

	system_instructions = (
		"You are a legal assistant. Answer the user's question based only on the given documents. "
		"If the documents do not contain the answer, say you cannot find it. Answer briefly."
	)

	user_content = (
		f"Context:\n{context}\n\n"
		f"Question: {user_query}\n\n"
		"Answer in English."
	)

	# 优先使用 Llama3 的 chat 模板
	if hasattr(tokenizer, "apply_chat_template"):
		messages = [
			{"role": "system", "content": system_instructions},
			{"role": "user", "content": user_content},
		]
		prompt = tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=True,
			tokenize=False,
		)
	else:
		# 兼容非 chat 模型的简单拼接
		prompt = (
			f"System: {system_instructions}\n\n"
			f"{user_content}"
		)

	return prompt


def load_models():
	device = "cuda" if torch.cuda.is_available() else "cpu"

	encoder = SentenceTransformer(ENCODER_PATH, device=device)

	tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
	model = AutoModelForCausalLM.from_pretrained(
		LLM_PATH,
		torch_dtype=torch.float16 if device == "cuda" else torch.float32,
		device_map="auto" if device == "cuda" else None,
	)

	return encoder, tokenizer, model, device


def generate_answer(
	tokenizer: AutoTokenizer,
	model: AutoModelForCausalLM,
	device: str,
	prompt: str,
	max_new_tokens: int,
) -> str:
	inputs = tokenizer(prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=True,
			pad_token_id=tokenizer.eos_token_id,
		)
	# 只解码新生成的部分，避免把整个 prompt 一起解码
	generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
	answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
	return answer.strip()


def main():
	parser = argparse.ArgumentParser(description="Simple RAG demo over QA-summary.jsonl")
	parser.add_argument("--num_samples", type=int, default=5, help="随机抽取的 query 数量")
	parser.add_argument("--seed", type=int, default=42, help="随机种子")
	parser.add_argument("--top_k", type=int, default=TOP_K, help="检索返回的文档条数")
	parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="生成答案的最大 token 数")
	args = parser.parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)

	print("加载知识库...")
	qa_entries = load_kb(KB_PATH)
	print(f"原始共有 {len(qa_entries)} 条 QA 样本。")
	kb_entries = deduplicate_kb_by_reference(qa_entries)
	print(f"按 reference 去重后，共 {len(kb_entries)} 条唯一 KB 文档。")

	print("加载编码器和大模型（可能较慢）...")
	encoder, tokenizer, model, device = load_models()

	print("构建知识库向量...")
	kb_embeddings = build_kb_embeddings(encoder, kb_entries)
	print("向量构建完成。可以开始问问题了。\n")

	# 根据 seed 随机抽取若干条 query 作为测试问题
	indices = list(range(len(qa_entries)))
	if args.num_samples < len(indices):
		indices = random.sample(indices, args.num_samples)

	for run_id, idx in enumerate(indices, start=1):
		entry = qa_entries[idx]
		user_query = entry.query

		print(f"\n====== 测试样本 {run_id}/{len(indices)} (idx={idx}) ======")
		print("问题:", user_query)
		print("参考答案:", entry.answer)

		retrieved = retrieve(encoder, kb_embeddings, kb_entries, user_query, top_k=args.top_k)
		print("\n检索到的文档得分：")
		for i, (score, r_entry) in enumerate(retrieved, start=1):
			print(f"  {i}. score={score:.4f} | query_example={r_entry.query[:80]}...")

		prompt = build_prompt(tokenizer, user_query, retrieved)
		print("\n调用大模型生成答案中...\n")
		answer = generate_answer(tokenizer, model, device, prompt, max_new_tokens=args.max_new_tokens)
		print("【RAG 回答】", answer)
		print("-" * 80)


if __name__ == "__main__":
	main()
