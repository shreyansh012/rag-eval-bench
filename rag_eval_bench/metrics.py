"""Metrics for one question. Each takes a ranked list of doc ids."""


def recall_at_k(ranked_ids, expected_ids, k):
    """Share of expected documents found in the first k results. 1.0 means all of them."""
    if not expected_ids:
        return 0.0
    first = set(ranked_ids[:k])
    return sum(1 for e in expected_ids if e in first) / len(expected_ids)


def reciprocal_rank(ranked_ids, expected_ids, depth):
    """1 / rank of the first expected document inside the first `depth` results. 0 if none."""
    wanted = set(expected_ids)
    for pos, doc_id in enumerate(ranked_ids[:depth], start=1):
        if doc_id in wanted:
            return 1.0 / pos
    return 0.0


def must_mention_hit(answer_text, terms):
    """True when every must-mention term appears in the answer text (ignoring case)."""
    low = answer_text.lower()
    return all(t.lower() in low for t in terms)
