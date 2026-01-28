import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np


_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return _WORD_RE.findall(text)


def _split_sentences(text: str) -> List[str]:
    """Very simple sentence/segment splitter.

    We intentionally keep it lightweight (no extra deps) and
    robust to judgment-style long paragraphs by splitting on
    blank lines first, then on punctuation.
    """
    if not text:
        return []

    # First split by blank lines to preserve paragraph structure
    paragraphs: List[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block:
            paragraphs.append(block)

    sentences: List[str] = []
    for para in paragraphs:
        # Rough split on Chinese/English sentence end markers
        tmp = re.split(r"([.!?\u3002\uff01\uff1f])", para)
        buf = ""
        for piece in tmp:
            if not piece:
                continue
            buf += piece
            if piece in [".", "!", "?", "\u3002", "\uff01", "\uff1f"]:
                s = buf.strip()
                if s:
                    sentences.append(s)
                buf = ""
        if buf.strip():
            sentences.append(buf.strip())

    # Fallback: if nothing was split, treat whole text as one segment
    if not sentences:
        sentences = [text.strip()]

    return sentences


@dataclass
class BM25Index:
    sentences: List[str]
    tokenized: List[List[str]]
    df: Dict[str, int]
    idf: Dict[str, float]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, sentences: List[str]) -> "BM25Index":
        tokenized: List[List[str]] = []
        df: Dict[str, int] = {}
        doc_lens: List[int] = []

        for sent in sentences:
            toks = _tokenize(sent)
            tokenized.append(toks)
            doc_lens.append(len(toks))
            uniq = set(toks)
            for t in uniq:
                df[t] = df.get(t, 0) + 1

        N = max(len(sentences), 1)
        avgdl = float(np.mean(doc_lens)) if doc_lens else 0.0
        idf: Dict[str, float] = {}
        for t, f in df.items():
            # Standard BM25 idf with small smoothing
            idf[t] = math.log(1 + (N - f + 0.5) / (f + 0.5))

        return cls(sentences=sentences, tokenized=tokenized, df=df, idf=idf, avgdl=avgdl)

    def score(self, query: str) -> List[float]:
        q_toks = _tokenize(query)
        scores: List[float] = []
        for i, doc in enumerate(self.tokenized):
            score = 0.0
            doc_len = len(doc) or 1
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            for t in q_toks:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                score += idf * f * (self.k1 + 1) / denom
            scores.append(score)
        return scores


def _extract_numbers(text: str) -> List[str]:
    """Extract numeric patterns like amounts, dates parts, etc."""
    return re.findall(r"[0-9]+(?:\.[0-9]+)?", text)


def _extract_upper_tokens(text: str) -> List[str]:
    """Very rough proxy for named entities / institutions.

    Note: We cannot reuse `_tokenize` because it lowercases text and loses
    information about capitalized tokens. We use a regex on the original
    text to capture tokens that start with an uppercase letter (e.g., names,
    institutions, clause names).
    """
    return re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", text)


def _extract_keywords_for_recall(text: str) -> Set[str]:
    """Keywords used for high-recall retrieval: numbers + capitalized tokens
    + simple monetary patterns.

    These keywords are used in BM25 scoring and for rule-based fallbacks
    (direct string matching in the original text to avoid missing key evidence
    sentences).
    """
    nums = set(_extract_numbers(text))
    uppers = set(_extract_upper_tokens(text))

    # 简单金额/币种模式，比如 "500 RMB", "22,233 yuan"
    money_like = set(re.findall(r"[0-9][0-9,\.]*\s*(rmb|yuan|元)", text, flags=re.IGNORECASE))

    return nums | uppers | money_like


def score_snippets(
    query: str,
    sentences: List[str],
    bm25_scores: List[float],
    query_type: str = "",
) -> List[Tuple[int, float]]:
    """Combine BM25 with a few simple heuristic features.

    Returns list of (index, score) sorted by descending score.
    """
    q_type = (query_type or "").lower()
    is_multi_hop_type = "multi-hop" in q_type
    is_summarization_type = "summarization" in q_type
    q_numbers = set(_extract_numbers(query))
    q_upper = set(_extract_upper_tokens(query))

    q_lower = (query or "").lower()

    def _contains_any(text: str, subs: List[str]) -> bool:
        return any(s in text for s in subs)

    is_amount_q = _contains_any(
        q_lower,
        [
            "how much",
            "what is the amount",
            "total amount",
            "fine",
            "double wages",
            "litigation fee",
            "litigation fees",
            "compensation amount",
            "overtime wages",
        ],
    ) or bool(re.search(r"\b(rmb|yuan|¥)\b", q_lower))

    is_date_q = _contains_any(
        q_lower,
        [
            "on what date",
            "what date",
            "when did",
            "when was",
            "date of",
            "filing date",
            "issued on",
            "rendered on",
        ],
    )

    is_evidence_q = _contains_any(
        q_lower,
        [
            "what type of evidence",
            "what evidence",
            "key evidence",
            "evidentiary support",
            "evidence was used",
        ],
    )

    is_legal_q = _contains_any(
        q_lower,
        [
            "under which legal provision",
            "under which provision",
            "under which article",
            "legal basis",
            "according to article",
            "in accordance with article",
        ],
    )

    is_identity_q = _contains_any(
        q_lower,
        [
            "legal status",
            "status of the applicant",
            "status of the defendant",
            "how old",
            "what is the age",
            "minor under fourteen",
            "under fourteen",
            "natural person",
            "employer",
            "employee",
        ],
    )

    is_cause_q = _contains_any(
        q_lower,
        [
            "cause of action",
            "core legal cause of action",
            "core cause of action",
            "what is the core cause",
            "legal dispute in this case",
            "dispute involving",
            "what is the dispute over",
        ],
    )

    is_procedure_q = _contains_any(
        q_lower,
        [
            "trial process",
            "procedural history",
            "procedural objections",
            "pre-litigation",
            "filing of lawsuit",
            "filed an administrative lawsuit",
            "filed the lawsuit",
            "investigation hearing",
            "civil mediation",
            "mediation",
        ],
    )

    is_court_q = _contains_any(
        q_lower,
        [
            "full official name of the court",
            "name of the court",
            "which court",
            "what court",
            "trial level",
        ],
    )

    boilerplate_phrases = [
        "within the jurisdiction of",
        "is a local people's court at the",
        "this case arises within the jurisdiction of",
        "the present case, registered under the docket number",
        "is a local people's court at the high level",
    ]

    fact_phrases = [
        "facts pertaining to",
        "fact pertaining to",
        "the court finds",
        "the court found",
        "this court finds",
        "this court found",
    ]

    legal_phrases = [
        "the court holds",
        "this court holds",
        "the court determines",
        "the court concludes",
        "in accordance with",
        "according to article",
    ]

    evidence_tokens = [
        "evidence",
        "household register",
        "bank transfer",
        "wechat",
        "witness statement",
        "testimony",
        "on-site photo",
        "on-site photos",
        "investigation record",
        "reconsideration decision",
        "penalty decision",
        "administrative penalty decision",
    ]

    law_tokens = [
        "law of the people's republic of china on",
        "labor contract law",
        "administrative penalties",
        "administrative penalty",
        "administrative procedure law",
        "labor law",
        "judicial interpretation",
        "guiding case",
    ]

    amount_tokens = [
        "rmb",
        "yuan",
        "¥",
        "double wages",
        "fine",
        "litigation fee",
        "litigation fees",
        "compensation",
        "overtime wages",
    ]

    identity_tokens = [
        "years old",
        "born on",
        "minor",
        "under fourteen",
        "natural person",
        "legal person",
        "employer",
        "employee",
        "individual business operator",
        "sole proprietor",
        "student",
    ]

    cause_tokens = [
        "cause of action",
        "dispute over",
        "labor dispute",
        "dispute involving",
        "dispute concerning",
    ]

    procedure_tokens = [
        "pre-litigation",
        "filing of",
        "filed an administrative lawsuit",
        "filed the lawsuit",
        "case acceptance notice",
        "accepted the case",
        "docketing the case",
        "reconsideration",
        "hearing",
        "mediation",
        "civil mediation",
        "procedural history",
    ]

    court_tokens = [
        "people's court",
        "intermediate people's court",
        "high people's court",
        "basic people's court",
        "civil judgment",
        "administrative judgment",
    ]

    scored: List[Tuple[int, float]] = []
    for idx, (sent, bm25) in enumerate(zip(sentences, bm25_scores)):
        s_numbers = set(_extract_numbers(sent))
        s_upper = set(_extract_upper_tokens(sent))
        sent_lower = sent.lower()

        num_overlap = len(q_numbers & s_numbers)
        upper_overlap = len(q_upper & s_upper)

        length_penalty = 0.0
        # Multi-hop and summarization queries often need long paragraphs;
        # for other types we penalize overly long sentences.
        if len(sent) > 800 and not (is_multi_hop_type or is_summarization_type):
            length_penalty = 0.2

        score = bm25
        score += 0.8 * num_overlap
        score += 0.5 * upper_overlap
        score -= length_penalty

        # 对多跳 / 总结题，额外偏好“包含多个数字”的句子，
        # 即便这些数字不一定都直接出现在 query 中。
        num_count = len(s_numbers)
        if (is_multi_hop_type or is_summarization_type) and num_count >= 2:
            score += 0.4
            if num_count >= 4:
                score += 0.4

        if is_amount_q:
            has_currency = _contains_any(sent_lower, amount_tokens)
            if s_numbers and has_currency:
                score += 1.0
            elif s_numbers:
                score += 0.4

        if is_date_q:
            has_year = bool(re.search(r"\b(19|20)\d{2}\b", sent_lower))
            has_month_word = bool(
                re.search(
                    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                    sent_lower,
                )
            )
            has_iso_date = bool(re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", sent_lower))
            if has_iso_date or (has_year and has_month_word):
                score += 1.0

        if is_evidence_q:
            if _contains_any(sent_lower, evidence_tokens):
                score += 1.0

        if is_legal_q:
            has_article = bool(re.search(r"\barticle\s+\d+\b", sent_lower))
            if has_article or _contains_any(sent_lower, law_tokens):
                score += 1.0

        if is_identity_q:
            if _contains_any(sent_lower, identity_tokens):
                score += 1.0

        if is_cause_q:
            if _contains_any(sent_lower, cause_tokens):
                score += 1.0

        if is_procedure_q:
            if _contains_any(sent_lower, procedure_tokens):
                score += 1.0

        if is_court_q:
            if _contains_any(sent_lower, court_tokens):
                score += 1.0

        if not is_court_q and not is_procedure_q:
            if _contains_any(sent_lower, boilerplate_phrases):
                score -= 0.3

        if _contains_any(sent_lower, fact_phrases):
            score += 0.3

        if _contains_any(sent_lower, legal_phrases):
            score += 0.2

        scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _high_recall_select_indices(
    query: str,
    sentences: List[str],
    bm25_scores: List[float],
    large_top_k: int = 80,
    window_size: int = 2,
    query_type: str = "",
) -> List[int]:
    """高召回的句子 index 选择：

    1) 先用现有 BM25+rule 得到排序结果；
    2) 取一个偏大的 top K （如 80 条）；
    3) 再对 query 的关键实体做字符串匹配回补；
    4) 对所有命中句做窗口扩展（±window_size 句）；
    5) 去重并按原文顺序排序，保证不漏关键信息且上下文完整。
    """

    if not sentences:
        return []

    ranked = score_snippets(query, sentences, bm25_scores, query_type=query_type)

    # 1) BM25 大召回：取前 large_top_k 个 index
    bm25_top_indices: Set[int] = set(idx for idx, _ in ranked[:large_top_k])

    # 2) 关键实体规则回补：直接字符串搜索
    keywords = _extract_keywords_for_recall(query)
    rule_indices: Set[int] = set()
    if keywords:
        lowered_sentences = [s.lower() for s in sentences]
        for kw in keywords:
            if not kw:
                continue
            kw_l = str(kw).lower()
            for i, s_l in enumerate(lowered_sentences):
                if kw_l in s_l:
                    rule_indices.add(i)

    # 3) 合并 BM25 & 规则 的命中句
    base_indices: Set[int] = bm25_top_indices | rule_indices

    # 如果啥都没有命中，fallback 到 BM25 第一句
    if not base_indices and ranked:
        base_indices.add(ranked[0][0])

    # 4) 对命中句做窗口扩展（前后各 window_size 句）
    expanded_indices: Set[int] = set()
    n = len(sentences)
    for idx in base_indices:
        start = max(0, idx - window_size)
        end = min(n - 1, idx + window_size)
        for j in range(start, end + 1):
            expanded_indices.add(j)

    # 5) 按原文顺序返回
    return sorted(expanded_indices)


def select_snippets_for_example(
    query: str,
    full_ref: str,
    max_chars: int = 800,
    max_sentences: int = 5,
    query_type: str = "",
) -> List[str]:
    """Select a compact list of supporting snippets for one (q, D).

    - Split full_ref into sentences
    - BM25 over sentences within this document
    - Add simple rule-based bonuses for numbers / names overlap
    - Greedily pick top sentences until reaching `max_chars`
    """
    sentences = _split_sentences(full_ref)
    if not sentences:
        return []

    bm25 = BM25Index.build(sentences)
    bm25_scores = bm25.score(query)

    # 高召回句子 index（BM25 大 K + 规则回补 + 窗口扩展）
    candidate_indices = _high_recall_select_indices(
        query, sentences, bm25_scores, query_type=query_type
    )

    # 在高召回集合上按“覆盖信息尽量多 + 长度受限”做截断
    selected: List[str] = []
    total_chars = 0

    # 为了让更相关的句子优先进入，在候选 index 内再用 score_snippets 排序一次
    sub_sentences = [sentences[i] for i in candidate_indices]
    sub_scores = [bm25_scores[i] for i in candidate_indices]
    ranked_local = score_snippets(query, sub_sentences, sub_scores, query_type=query_type)

    # ranked_local 的 index 是相对于 sub_sentences 的，需要映射回原 index
    for local_idx, _ in ranked_local:
        idx = candidate_indices[local_idx]
        s = sentences[idx].strip()
        if not s:
            continue
        if s in selected:
            continue
        if total_chars + len(s) > max_chars and selected:
            break
        selected.append(s)
        total_chars += len(s)
        if len(selected) >= max_sentences:
            break

    # 兜底：如果因为各种限制没选出来，就至少返回一个最相关句
    if not selected:
        # 直接用 BM25 全局最优句
        global_ranked = score_snippets(query, sentences, bm25_scores, query_type=query_type)
        best_idx = global_ranked[0][0]
        return [sentences[best_idx].strip()]

    return selected


def prune_full_text_for_example(
    query: str,
    full_ref: str,
    query_type: str = "",
) -> str:
    """Prune clearly irrelevant sentences while keeping most of the document.

    目标：做"减法"而不是强压缩——删除一部分与问题明显无关、得分极低的句子，
    但整体长度仍然接近原文，从而尽量保持答案可推导性。
    """

    sentences = _split_sentences(full_ref)
    if not sentences:
        return full_ref or ""

    bm25 = BM25Index.build(sentences)
    bm25_scores = bm25.score(query)

    scored = score_snippets(query, sentences, bm25_scores, query_type=query_type)
    if not scored:
        return full_ref or ""

    # 按原句顺序还原打分
    scores_by_idx: List[float] = [0.0] * len(sentences)
    for idx, s in scored:
        scores_by_idx[idx] = s

    q_type_lower = (query_type or "").lower()
    if "multi-hop" in q_type_lower or "summarization" in q_type_lower:
        prune_ratio = 0.15
    else:
        prune_ratio = 0.3

    # 低分阈值：仅考虑分布中靠后的部分
    sorted_scores = sorted(scores_by_idx)
    cut_idx = int(len(sorted_scores) * prune_ratio)
    if cut_idx <= 0:
        return full_ref or ""
    threshold = sorted_scores[cut_idx]
    max_score = max(scores_by_idx) or 1e-6

    # 高相关窗口：与 top-K 相邻的句子一律保留
    top_k = sorted(range(len(sentences)), key=lambda i: scores_by_idx[i], reverse=True)[:20]
    top_k_set = set(top_k)

    keep_flags: List[bool] = []
    for i, sent in enumerate(sentences):
        s_score = scores_by_idx[i]

        # 靠近高分句：保留
        if i in top_k_set or (i - 1) in top_k_set or (i + 1) in top_k_set:
            keep_flags.append(True)
            continue

        has_num = bool(_extract_numbers(sent))
        has_upper = bool(_extract_upper_tokens(sent))

        # 极低分、且没有数字和大写实体的句子，认为是可以安全删除的套话/背景
        if (
            s_score <= threshold
            and s_score < 0.2 * max_score
            and (not has_num)
            and (not has_upper)
        ):
            keep_flags.append(False)
        else:
            keep_flags.append(True)

    pruned_sentences = [s for s, keep in zip(sentences, keep_flags) if keep]
    if not pruned_sentences:
        return full_ref or ""
    if len(pruned_sentences) == len(sentences):
        return full_ref or ""

    return "\n".join(pruned_sentences)


def _load_json_or_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return [data]
        except json.JSONDecodeError:
            items: List[Dict] = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return items


def _dump_json_or_jsonl(data: List[Dict], path: str) -> None:
    # Follow train.py: accept both JSON/JSONL; here we always write JSONL
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_dataset(
    input_path: str,
    output_path: str,
    max_chars: int,
    max_sentences: int,
    mode: str = "snippets",
) -> None:
    dataset = _load_json_or_jsonl(input_path)

    processed: List[Dict] = []
    for ex in dataset:
        q_meta = ex.get("query", {})
        query = q_meta.get("content", "")
        q_type = (q_meta.get("query_type") or "").strip()
        gt = ex.get("ground_truth", {})
        refs = gt.get("references", "")

        # 标准化成字符串全文
        if isinstance(refs, list):
            full_ref = "\n".join(str(r) for r in refs)
        elif isinstance(refs, dict):
            full_ref = refs.get("content", str(refs))
        elif isinstance(refs, str):
            full_ref = refs
        else:
            full_ref = str(refs)

        new_ex = dict(ex)
        new_gt = dict(gt)

        if mode == "snippets":
            # 针对多跳 / 总结问题，放宽长度限制以保留更多上下文
            eff_max_chars = max_chars
            eff_max_sentences = max_sentences
            q_type_lower = q_type.lower()
            if "multi-hop" in q_type_lower:
                eff_max_chars = int(max_chars * 2)
                eff_max_sentences = max_sentences + 3
            elif "summarization" in q_type_lower:
                eff_max_chars = int(max_chars * 2.5)
                eff_max_sentences = max_sentences + 4

            # 在线检索同构：只使用 question 作为查询（但离线阶段可利用 query_type
            # 调整截断长度和打分偏好，不把答案 / config 信息泄露给检索器）。
            snippets = select_snippets_for_example(
                query,
                full_ref,
                max_chars=eff_max_chars,
                max_sentences=eff_max_sentences,
                query_type=q_type,
            )

            # 将精简片段写回 ground_truth.references，使用字符串列表形式
            new_gt["references"] = snippets
        else:
            # 减法模式：删除明显无关的句子，保留精简长文
            pruned = prune_full_text_for_example(query, full_ref, query_type=q_type)
            new_gt["references"] = pruned

        new_ex["ground_truth"] = new_gt
        processed.append(new_ex)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _dump_json_or_jsonl(processed, output_path)


def main():
    parser = argparse.ArgumentParser(description="BM25 + rule-based snippet selection for QA datasets")
    parser.add_argument("--input_path", type=str, required=True, help="Path to original QA dataset (JSON/JSONL)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write processed dataset")
    parser.add_argument("--max_chars", type=int, default=800, help="Max total characters for concatenated snippets")
    parser.add_argument("--max_sentences", type=int, default=5, help="Max number of snippets per example")
    parser.add_argument(
        "--mode",
        type=str,
        default="snippets",
        choices=["snippets", "prune"],
        help="snippets: 返回少量高相关片段；prune: 删除明显无关句，仅做减法精简",
    )

    args = parser.parse_args()
    process_dataset(args.input_path, args.output_path, args.max_chars, args.max_sentences, mode=args.mode)


if __name__ == "__main__":
    main()
