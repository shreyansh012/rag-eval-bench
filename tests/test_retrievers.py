import random
import unittest

import helpers
from rag_eval_bench.retrievers import BM25, TfidfCosine, WordOverlap


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.questions, cls.cfg = helpers.load_all()
        cls.bm25 = BM25(cls.docs)
        cls.tfidf = TfidfCosine(cls.docs)
        cls.overlap = WordOverlap(cls.docs)

    def best(self, retriever, query):
        return retriever.rank(query, random.Random(1))[0][0]

    def test_bm25_finds_obvious_documents(self):
        self.assertEqual(self.best(self.bm25, "tune the B string with a fret"), "d14")
        self.assertEqual(self.best(self.bm25, "how to sharpen a knife on a whetstone"), "d09")

    def test_tfidf_finds_obvious_documents(self):
        self.assertEqual(self.best(self.tfidf, "tide table high and low"), "d04")
        self.assertEqual(self.best(self.tfidf, "freshwater aquarium ammonia"), "d11")

    def test_every_document_is_ranked_once(self):
        for retriever in (self.bm25, self.tfidf, self.overlap):
            ids = [i for i, _ in retriever.rank("coffee water", random.Random(1))]
            self.assertEqual(sorted(ids), sorted(d["id"] for d in self.docs))

    def test_scores_are_sorted_best_first(self):
        for retriever in (self.bm25, self.tfidf, self.overlap):
            scores = [s for _, s in retriever.rank("seeds dry winter", random.Random(1))]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unknown_words_score_zero(self):
        for retriever in (self.bm25, self.tfidf, self.overlap):
            scores = [s for _, s in retriever.rank("zzzz qqqq", random.Random(1))]
            self.assertTrue(all(s == 0 for s in scores))

    def test_tfidf_cosine_is_at_most_one(self):
        scores = [s for _, s in self.tfidf.rank("sourdough starter flour water", random.Random(1))]
        self.assertLessEqual(max(scores), 1.0 + 1e-9)

    def test_same_seed_gives_same_order(self):
        a = self.overlap.rank("kitchen tool", random.Random(7))
        b = self.overlap.rank("kitchen tool", random.Random(7))
        self.assertEqual(a, b)

    def test_different_seeds_change_tied_order(self):
        orders = {
            tuple(i for i, _ in self.overlap.rank("kitchen tool condition", random.Random(s)))
            for s in range(10)
        }
        self.assertGreater(len(orders), 1)

    def test_seed_does_not_change_untied_order(self):
        orders = {
            tuple(i for i, sc in self.bm25.rank("tune the B string with a fret", random.Random(s)) if sc > 0)
            for s in range(10)
        }
        self.assertEqual(len(orders), 1)


if __name__ == "__main__":
    unittest.main()
