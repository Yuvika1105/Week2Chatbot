import re

# Standard list of English stop words to exclude during tokenization
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", 
    "being", "to", "from", "in", "on", "at", "by", "for", "with", "about", "of", "it", 
    "its", "they", "them", "their", "he", "him", "his", "she", "her", "you", "your", 
    "we", "us", "our", "what", "which", "who", "whom", "this", "that", "these", "those"
}

def _tokenize(text: str) -> set[str]:
    """Lowercase text, remove punctuation, split into words, and filter out stop words."""
    if not text or not isinstance(text, str):
        return set()
    # Replace any punctuation character with a space
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = cleaned.split()
    return {w for w in words if w not in STOP_WORDS and len(w) > 0}

def _jaccard(a: set, b: set) -> float:
    """Compute the Jaccard index (similarity) between two sets."""
    if not a or not b:
        return 0.0
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / len(union)

def context_precision(retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
    """Fraction of retrieved chunks that are relevant (Jaccard similarity >= 0.3 with any relevant chunk)."""
    if not retrieved_chunks:
        return 1.0 if not relevant_chunks else 0.0
    relevant_tokens_list = [_tokenize(rc) for rc in relevant_chunks if rc]
    if not relevant_tokens_list:
        return 0.0

    relevant_retrieved_count = 0
    for chunk in retrieved_chunks:
        chunk_tokens = _tokenize(chunk)
        is_relevant = False
        for rt in relevant_tokens_list:
            if _jaccard(chunk_tokens, rt) >= 0.3:
                is_relevant = True
                break
        if is_relevant:
            relevant_retrieved_count += 1

    return relevant_retrieved_count / len(retrieved_chunks)

def context_recall(retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
    """Fraction of required ground truth chunks that were retrieved (Jaccard similarity >= 0.3)."""
    if not relevant_chunks:
        return 1.0
    retrieved_tokens_list = [_tokenize(c) for c in retrieved_chunks if c]
    if not retrieved_tokens_list:
        return 0.0

    found_count = 0
    for rc in relevant_chunks:
        rc_tokens = _tokenize(rc)
        is_retrieved = False
        for rt in retrieved_tokens_list:
            if _jaccard(rc_tokens, rt) >= 0.3:
                is_retrieved = True
                break
        if is_retrieved:
            found_count += 1

    return found_count / len(relevant_chunks)

def answer_faithfulness(answer: str, retrieved_chunks: list[str], overlap_threshold: float = 0.4) -> float:
    """Fraction of claims in generated answer that are supported by the retrieved context chunks (Jaccard similarity >= threshold)."""
    if not answer or not answer.strip():
        return 1.0

    # Refusals are always faithful by definition
    ans_lower = answer.lower()
    refusal_keywords = [
        "i don't have", "not enough information", "cannot access", "don't have access",
        "do not have access", "i'm sorry, i can't", "access denied", "not permitted",
        "technical issue", "system inference fault", "blocked:", "refuse", "not found"
    ]
    for kw in refusal_keywords:
        if kw in ans_lower:
            return 1.0

    if not retrieved_chunks:
        return 0.0

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", answer) if s.strip()]
    if not sentences:
        return 1.0

    # Split retrieved chunks into sentences for fair size-matched Jaccard comparison
    chunk_sentences = []
    for chunk in retrieved_chunks:
        if not chunk:
            continue
        c_sents = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", chunk) if s.strip()]
        for cs in c_sents:
            cs_tokens = _tokenize(cs)
            if cs_tokens:
                chunk_sentences.append(cs_tokens)

    if not chunk_sentences:
        return 0.0

    supported_sentences = 0
    for s in sentences:
        s_tokens = _tokenize(s)
        if not s_tokens:
            supported_sentences += 1
            continue
        
        is_supported = False
        for cs_tokens in chunk_sentences:
            if _jaccard(s_tokens, cs_tokens) >= overlap_threshold:
                is_supported = True
                break
        if is_supported:
            supported_sentences += 1

    return supported_sentences / len(sentences)

def answer_relevance(question: str, answer: str) -> float:
    """Fraction of question non-stop-word keywords that are found in the generated answer (case-insensitive)."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 1.0
    ans_lower = answer.lower()
    match_count = sum(1 for token in q_tokens if token in ans_lower)
    return match_count / len(q_tokens)

def source_accuracy(predicted_sources: list[str], ground_truth_sources: list[str]) -> float:
    """Set overlap recall over sources: fraction of correct source files that were actually cited."""
    if not ground_truth_sources:
        return 1.0
    pred_set = {p.strip().lower() for p in predicted_sources if p}
    gt_set = {g.strip().lower() for g in ground_truth_sources if g}
    if not gt_set:
        return 1.0
    return len(pred_set.intersection(gt_set)) / len(gt_set)

def answer_f1(generated_answer: str, ground_truth_answer: str) -> dict:
    """Harmonic mean F1 of precision and recall at individual token level."""
    gen_tokens = _tokenize(generated_answer)
    gt_tokens = _tokenize(ground_truth_answer)
    
    if not gen_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gen_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = gen_tokens.intersection(gt_tokens)
    precision = len(intersection) / len(gen_tokens)
    recall = len(intersection) / len(gt_tokens)
    
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    return {"precision": precision, "recall": recall, "f1": f1}
