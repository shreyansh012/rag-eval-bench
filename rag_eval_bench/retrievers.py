"""Three small retrievers. All use only the standard library.

Every retriever has the same shape:

    retriever.rank(query, rng) -> [(doc_id, score), ...]   best first

`rng` is a random.Random. It only breaks ties between equal scores, so the
same seed always gives the same order.
"""

import math
from collections import Counter

from .text import tokenize


class Retriever:
    name = "base"

    def __init__(self, documents):
        self.ids = [d["id"] for d in documents]
        self.tokens = {d["id"]: tokenize(d["title"] + " " + d["text"]) for d in documents}
        self.n = len(self.ids)
        self.df = Counter()
        for toks in self.tokens.values():
            self.df.update(set(toks))

    def scores(self, query_tokens):
        raise NotImplementedError

    def rank(self, query, rng):
        scores = self.scores(tokenize(query))
        tie_key = {i: rng.random() for i in self.ids}
        order = sorted(self.ids, key=lambda i: (-scores[i], tie_key[i]))
        return [(i, scores[i]) for i in order]


class BM25(Retriever):
    """Okapi BM25 with the usual k1 and b."""

    name = "bm25"

    def __init__(self, documents, k1=1.5, b=0.75):
        super().__init__(documents)
        self.k1 = k1
        self.b = b
        self.counts = {i: Counter(t) for i, t in self.tokens.items()}
        self.avg_len = sum(len(t) for t in self.tokens.values()) / self.n

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens):
        out = {}
        for doc_id in self.ids:
            counts = self.counts[doc_id]
            length = len(self.tokens[doc_id])
            total = 0.0
            for term in set(query_tokens):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf(term) * tf * (self.k1 + 1) / norm
            out[doc_id] = total
        return out


class TfidfCosine(Retriever):
    """Log-scaled term frequency times inverse document frequency, cosine score."""

    name = "tfidf"

    def __init__(self, documents):
        super().__init__(documents)
        self.vectors = {i: self._vector(Counter(t)) for i, t in self.tokens.items()}

    def _idf(self, term):
        return math.log((1 + self.n) / (1 + self.df.get(term, 0))) + 1.0

    def _vector(self, counts):
        vec = {t: (1.0 + math.log(c)) * self._idf(t) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return {}
        return {t: v / norm for t, v in vec.items()}

    def scores(self, query_tokens):
        q = self._vector(Counter(t for t in query_tokens if t in self.df))
        out = {}
        for doc_id in self.ids:
            doc = self.vectors[doc_id]
            out[doc_id] = sum(w * doc.get(t, 0.0) for t, w in q.items())
        return out


class WordOverlap(Retriever):
    """A deliberately coarse retriever: score = number of query words in the document.

    Many documents get the same whole-number score, so the seeded tie-breaker
    changes the order from run to run. It exists to show the majority-of-runs rule.
    """

    name = "overlap"

    def scores(self, query_tokens):
        wanted = set(query_tokens)
        return {i: float(len(wanted & set(t))) for i, t in self.tokens.items()}


def default_retrievers(documents):
    return [BM25(documents), TfidfCosine(documents), WordOverlap(documents)]
