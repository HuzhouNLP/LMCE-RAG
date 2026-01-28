import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


# Default paths (can be overridden via command-line arguments)
# Replace these with local paths in your environment before running.
KB_ORIG_PATH = "/path/to/lkif/qar_generation/results/QA-summary.jsonl"
KB_REF_PATH = "/path/to/lkif/qar_generation/results/QA-summary-ref.json"
ENCODER_PATH = "/path/to/lkif/encoders/bge-large-en-v1.5"
LLM_PATH = "/path/to/lkif/llms/Meta-Llama-3-8B-Instruct"
TOP_K = 3
MAX_NEW_TOKENS = 256


@dataclass
class KBEntry:
    query: str
    reference: str
    answer: str


def _init_nvml(device_index: int):
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return pynvml, handle
    except Exception:
        return None, None


def _shutdown_nvml(nvml_module) -> None:
    if nvml_module is None:
        return
    try:
        nvml_module.nvmlShutdown()
    except Exception:
        pass


def _gpu_util_percent(nvml_module, handle) -> Optional[float]:
    if nvml_module is None or handle is None:
        return None
    try:
        util_rates = nvml_module.nvmlDeviceGetUtilizationRates(handle)
        return float(util_rates.gpu)
    except Exception:
        return None


def load_kb(path: str) -> List[KBEntry]:
    """Load KB file, supporting two formats:

    1) JSON array: entire content is List[Object]
    2) JSONL: one JSON object per line
    """
    entries: List[KBEntry] = []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return entries

    # If starts with '[', parse as JSON array
    if text.lstrip().startswith("["):
        try:
            objects = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON array KB file: {path}: {e}")
    else:
        # Otherwise parse as JSONL, one JSON per line
        objects = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            objects.append(json.loads(line))

    for obj in objects:
        q = obj["query"]["content"]
        ref = obj["ground_truth"]["references"]
        ans = obj["ground_truth"]["content"]
        # references can be list or str, unify to str
        if isinstance(ref, list):
            ref_text = "\n".join(str(r) for r in ref)
        else:
            ref_text = str(ref)
        entries.append(KBEntry(query=q, reference=ref_text, answer=ans))

    return entries


def deduplicate_kb_by_reference(entries: List[KBEntry]) -> List[KBEntry]:
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
    """与 experiments/rag.py 保持一致的提示词构造方式。"""
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
        prompt = (
            f"System: {system_instructions}\n\n"
            f"{user_content}"
        )

    return prompt


def build_zero_shot_prompt(
    tokenizer: AutoTokenizer,
    user_query: str,
) -> str:
    system_instructions = (
        "You are a legal assistant. Answer the user's question using your own knowledge. "
        "If you are uncertain, say you are unsure. Answer briefly."
    )

    user_content = f"Question: {user_query}\n\nAnswer in English."

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
        prompt = (
            f"System: {system_instructions}\n\n"
            f"{user_content}"
        )

    return prompt


def load_models(encoder_path: str, llm_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = SentenceTransformer(encoder_path, device=device)

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
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
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


# ===== 简单 ROUGE 实现（1/2/L，基于 token 序列） =====

import re


def _tokenize_for_rouge(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _precision_recall_f1(matches: int, pred_total: int, ref_total: int) -> Tuple[float, float, float]:
    if pred_total == 0 or ref_total == 0 or matches == 0:
        return 0.0, 0.0, 0.0
    p = matches / pred_total
    r = matches / ref_total
    f1 = 2 * p * r / (p + r)
    return p, r, f1


def rouge_n(pred: str, ref: str, n: int) -> Tuple[float, float, float]:
    pred_toks = _tokenize_for_rouge(pred)
    ref_toks = _tokenize_for_rouge(ref)
    pred_counts = _ngram_counts(pred_toks, n)
    ref_counts = _ngram_counts(ref_toks, n)

    matches = 0
    for ng, c_ref in ref_counts.items():
        c_pred = pred_counts.get(ng, 0)
        matches += min(c_ref, c_pred)

    return _precision_recall_f1(matches, sum(pred_counts.values()), sum(ref_counts.values()))


def lcs(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(pred: str, ref: str) -> Tuple[float, float, float]:
    pred_toks = _tokenize_for_rouge(pred)
    ref_toks = _tokenize_for_rouge(ref)
    lcs_len = lcs(pred_toks, ref_toks)
    return _precision_recall_f1(lcs_len, len(pred_toks), len(ref_toks))


def evaluate_rag_on_kb(
    kb_path: str,
    encoder: SentenceTransformer,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    num_samples: int,
    seed: int,
    top_k: int,
    max_new_tokens: int,
    sample_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], List[int]]:
    print(f"\n=== Evaluating KB: {kb_path} ===")
    eval_start = time.perf_counter()
    qa_entries = load_kb(kb_path)
    print(f"Loaded {len(qa_entries)} QA entries.")

    kb_entries = deduplicate_kb_by_reference(qa_entries)
    print(f"After reference dedup: {len(kb_entries)} unique KB docs.")

    print("Building KB embeddings...")
    kb_embeddings = build_kb_embeddings(encoder, kb_entries)

    if sample_indices is not None:
        indices = sample_indices
    else:
        indices = list(range(len(qa_entries)))
        if num_samples < len(indices):
            random.seed(seed)
            indices = random.sample(indices, num_samples)

    sum_r1, sum_r2, sum_rl, sum_sim = 0.0, 0.0, 0.0, 0.0
    total_latency = 0.0
    peak_memories: List[float] = []
    gpu_utils: List[float] = []

    device_index = torch.cuda.current_device() if torch.cuda.is_available() and device.startswith("cuda") else None
    nvml_module, nvml_handle = _init_nvml(device_index) if device_index is not None else (None, None)
    total_latency = 0.0
    peak_memories: List[float] = []
    gpu_utils: List[float] = []

    device_index = torch.cuda.current_device() if torch.cuda.is_available() and device.startswith("cuda") else None
    nvml_module, nvml_handle = _init_nvml(device_index) if device_index is not None else (None, None)

    for i, idx in enumerate(indices, start=1):
        entry = qa_entries[idx]
        gt_answer = entry.answer
        user_query = entry.query

        if device_index is not None:
            torch.cuda.synchronize(device_index)
            torch.cuda.reset_peak_memory_stats(device_index)

        sample_start = time.perf_counter()
        retrieved = retrieve(encoder, kb_embeddings, kb_entries, user_query, top_k=top_k)
        prompt = build_prompt(tokenizer, user_query, retrieved)
        pred_answer = generate_answer(tokenizer, model, device, prompt, max_new_tokens=max_new_tokens)

        if device_index is not None:
            torch.cuda.synchronize(device_index)
        elapsed = time.perf_counter() - sample_start
        total_latency += elapsed
        peak_mem = None
        gpu_util = None
        if device_index is not None:
            peak_mem = torch.cuda.max_memory_reserved(device_index) / (1024 ** 3)
            peak_memories.append(peak_mem)
            gpu_util = _gpu_util_percent(nvml_module, nvml_handle)
            if gpu_util is not None:
                gpu_utils.append(gpu_util)

        # ROUGE F1
        _, _, r1_f = rouge_n(pred_answer, gt_answer, 1)
        _, _, r2_f = rouge_n(pred_answer, gt_answer, 2)
        _, _, rl_f = rouge_l(pred_answer, gt_answer)

        # 语义相似度（使用同一个 encoder）
        emb_pred = encoder.encode(pred_answer, convert_to_tensor=True)
        emb_ref = encoder.encode(gt_answer, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_pred, emb_ref)[0][0])

        sum_r1 += r1_f
        sum_r2 += r2_f
        sum_rl += rl_f
        sum_sim += sim

        # 打印当前样本的指标和答案，方便人工检查
        print(f"Sample {i}/{len(indices)} | idx={idx} | ROUGE-1={r1_f:.4f}, ROUGE-2={r2_f:.4f}, ROUGE-L={rl_f:.4f}, sim={sim:.4f}")
        print("  Query        :", user_query)
        print("  Ground truth :", gt_answer)
        print("  Pred answer  :", pred_answer)
        metrics_line = f"  Metrics      : latency={elapsed:.3f}s"
        if peak_mem is not None:
            metrics_line += f", peak_mem={peak_mem:.2f}GB"
        if gpu_util is not None:
            metrics_line += f", gpu_util={gpu_util:.1f}%"
        print(metrics_line)
        print("  -----")

    n = len(indices)
    if device_index is not None:
        torch.cuda.synchronize(device_index)
    total_wall = time.perf_counter() - eval_start

    metrics = {
        "rouge1_f": sum_r1 / n,
        "rouge2_f": sum_r2 / n,
        "rougeL_f": sum_rl / n,
        "semantic_sim": sum_sim / n,
        "num_eval": n,
        "avg_latency_s": total_latency / n if n else 0.0,
        "throughput_samples_per_s": (n / total_latency) if total_latency > 0 else 0.0,
        "total_latency_s": total_latency,
        "wall_time_s": total_wall,
    }
    if peak_memories:
        metrics["avg_gpu_mem_gb"] = sum(peak_memories) / len(peak_memories)
        metrics["max_gpu_mem_gb"] = max(peak_memories)
    if gpu_utils:
        metrics["avg_gpu_util_percent"] = sum(gpu_utils) / len(gpu_utils)
        metrics["max_gpu_util_percent"] = max(gpu_utils)

    _shutdown_nvml(nvml_module)
    return metrics, indices


def evaluate_zero_shot_on_kb(
    kb_path: str,
    encoder: SentenceTransformer,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    num_samples: int,
    seed: int,
    max_new_tokens: int,
    sample_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], List[int]]:
    print(f"\n=== Evaluating Zero-Shot on KB: {kb_path} ===")
    eval_start = time.perf_counter()
    qa_entries = load_kb(kb_path)
    print(f"Loaded {len(qa_entries)} QA entries.")

    if sample_indices is not None:
        indices = sample_indices
    else:
        indices = list(range(len(qa_entries)))
        if num_samples < len(indices):
            random.seed(seed)
            indices = random.sample(indices, num_samples)

    sum_r1, sum_r2, sum_rl, sum_sim = 0.0, 0.0, 0.0, 0.0
    total_latency = 0.0
    peak_memories: List[float] = []
    gpu_utils: List[float] = []

    device_index = torch.cuda.current_device() if torch.cuda.is_available() and device.startswith("cuda") else None
    nvml_module, nvml_handle = _init_nvml(device_index) if device_index is not None else (None, None)

    for i, idx in enumerate(indices, start=1):
        entry = qa_entries[idx]
        gt_answer = entry.answer
        user_query = entry.query

        if device_index is not None:
            torch.cuda.synchronize(device_index)
            torch.cuda.reset_peak_memory_stats(device_index)

        sample_start = time.perf_counter()
        prompt = build_zero_shot_prompt(tokenizer, user_query)
        pred_answer = generate_answer(tokenizer, model, device, prompt, max_new_tokens=max_new_tokens)

        if device_index is not None:
            torch.cuda.synchronize(device_index)
        elapsed = time.perf_counter() - sample_start
        total_latency += elapsed
        peak_mem = None
        gpu_util = None
        if device_index is not None:
            peak_mem = torch.cuda.max_memory_reserved(device_index) / (1024 ** 3)
            peak_memories.append(peak_mem)
            gpu_util = _gpu_util_percent(nvml_module, nvml_handle)
            if gpu_util is not None:
                gpu_utils.append(gpu_util)

        _, _, r1_f = rouge_n(pred_answer, gt_answer, 1)
        _, _, r2_f = rouge_n(pred_answer, gt_answer, 2)
        _, _, rl_f = rouge_l(pred_answer, gt_answer)

        emb_pred = encoder.encode(pred_answer, convert_to_tensor=True)
        emb_ref = encoder.encode(gt_answer, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_pred, emb_ref)[0][0])

        sum_r1 += r1_f
        sum_r2 += r2_f
        sum_rl += rl_f
        sum_sim += sim

        print(f"Sample {i}/{len(indices)} | idx={idx} | ROUGE-1={r1_f:.4f}, ROUGE-2={r2_f:.4f}, ROUGE-L={rl_f:.4f}, sim={sim:.4f}")
        print("  Query        :", user_query)
        print("  Ground truth :", gt_answer)
        print("  Pred answer  :", pred_answer)
        metrics_line = f"  Metrics      : latency={elapsed:.3f}s"
        if peak_mem is not None:
            metrics_line += f", peak_mem={peak_mem:.2f}GB"
        if gpu_util is not None:
            metrics_line += f", gpu_util={gpu_util:.1f}%"
        print(metrics_line)
        print("  -----")

    n = len(indices)
    if device_index is not None:
        torch.cuda.synchronize(device_index)
    total_wall = time.perf_counter() - eval_start

    metrics = {
        "rouge1_f": sum_r1 / n,
        "rouge2_f": sum_r2 / n,
        "rougeL_f": sum_rl / n,
        "semantic_sim": sum_sim / n,
        "num_eval": n,
        "avg_latency_s": total_latency / n if n else 0.0,
        "throughput_samples_per_s": (n / total_latency) if total_latency > 0 else 0.0,
        "total_latency_s": total_latency,
        "wall_time_s": total_wall,
    }
    if peak_memories:
        metrics["avg_gpu_mem_gb"] = sum(peak_memories) / len(peak_memories)
        metrics["max_gpu_mem_gb"] = max(peak_memories)
    if gpu_utils:
        metrics["avg_gpu_util_percent"] = sum(gpu_utils) / len(gpu_utils)
        metrics["max_gpu_util_percent"] = max(gpu_utils)

    _shutdown_nvml(nvml_module)
    return metrics, indices


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG on original vs refined QA datasets.")
    parser.add_argument("--orig_kb", type=str, default=KB_ORIG_PATH, help="Path to original QA-summary.jsonl")
    parser.add_argument("--ref_kb", type=str, default=KB_REF_PATH, help="Path to refined QA-summary-ref.json/jsonl")
    parser.add_argument("--encoder_path", type=str, default=ENCODER_PATH, help="SentenceTransformer model path")
    parser.add_argument("--llm_path", type=str, default=LLM_PATH, help="Causal LM model path")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of QA examples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k documents to retrieve")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Max new tokens for generation")
    parser.add_argument("--evaluate_zero_shot", action="store_true", help="Also evaluate zero-shot (no retrieval) baseline")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading encoder and LLM (may take a while)...")
    encoder, tokenizer, model, device = load_models(args.encoder_path, args.llm_path)

    # 评估原始 KB
    metrics_orig, orig_indices = evaluate_rag_on_kb(
        args.orig_kb,
        encoder,
        tokenizer,
        model,
        device,
        num_samples=args.num_samples,
        seed=args.seed,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    # 评估精简 KB
    metrics_ref, ref_indices = evaluate_rag_on_kb(
        args.ref_kb,
        encoder,
        tokenizer,
        model,
        device,
        num_samples=args.num_samples,
        seed=args.seed,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    if args.evaluate_zero_shot:
        metrics_orig_zero, _ = evaluate_zero_shot_on_kb(
            args.orig_kb,
            encoder,
            tokenizer,
            model,
            device,
            num_samples=args.num_samples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            sample_indices=orig_indices,
        )

        metrics_ref_zero, _ = evaluate_zero_shot_on_kb(
            args.ref_kb,
            encoder,
            tokenizer,
            model,
            device,
            num_samples=args.num_samples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            sample_indices=ref_indices,
        )

    print("\n=== SUMMARY (averaged over samples) ===")
    def fmt(m):
        parts = [
            f"R1={m['rouge1_f']:.4f}",
            f"R2={m['rouge2_f']:.4f}",
            f"RL={m['rougeL_f']:.4f}",
            f"sim={m['semantic_sim']:.4f}",
        ]
        if "avg_latency_s" in m:
            parts.append(f"lat={m['avg_latency_s']:.3f}s")
        if "throughput_samples_per_s" in m:
            parts.append(f"thr={m['throughput_samples_per_s']:.2f}/s")
        if "wall_time_s" in m:
            parts.append(f"wall={m['wall_time_s']:.1f}s")
        if "avg_gpu_mem_gb" in m:
            parts.append(f"gpu_mem={m['avg_gpu_mem_gb']:.2f}GB")
        if "avg_gpu_util_percent" in m:
            parts.append(f"gpu_util={m['avg_gpu_util_percent']:.1f}%")
        parts.append(f"N={m['num_eval']}")
        return ", ".join(parts)

    print(f"Original KB: {args.orig_kb}")
    print("  RAG   :", fmt(metrics_orig))
    if args.evaluate_zero_shot:
        print("  Zero  :", fmt(metrics_orig_zero))
    print(f"Refined  KB: {args.ref_kb}")
    print("  RAG   :", fmt(metrics_ref))
    if args.evaluate_zero_shot:
        print("  Zero  :", fmt(metrics_ref_zero))


if __name__ == "__main__":
    main()
