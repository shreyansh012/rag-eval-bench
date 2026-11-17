import unittest

import helpers  # noqa: F401  (sets up the import path)
from rag_eval_bench.metrics import must_mention_hit, recall_at_k, reciprocal_rank
from rag_eval_bench.text import tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_drops_stopwords(self):
        self.assertEqual(tokenize("How do I Feed the Starter?"), ["feed", "starter"])

    def test_splits_on_punctuation(self):
        self.assertEqual(tokenize("pour-over, coffee!"), ["pour", "over", "coffee"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class MetricTests(unittest.TestCase):
    def test_recall_all_found(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["a", "b"], 3), 1.0)

    def test_recall_half_found(self):
        self.assertEqual(recall_at_k(["a", "x", "y"], ["a", "b"], 3), 0.5)

    def test_recall_respects_k(self):
        self.assertEqual(recall_at_k(["x", "y", "a"], ["a"], 2), 0.0)
        self.assertEqual(recall_at_k(["x", "y", "a"], ["a"], 3), 1.0)

    def test_recall_no_expected(self):
        self.assertEqual(recall_at_k(["a"], [], 3), 0.0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["a", "b"], ["a"], 10), 1.0)
        self.assertEqual(reciprocal_rank(["x", "b"], ["b"], 10), 0.5)
        self.assertEqual(reciprocal_rank(["x", "y", "z"], ["a"], 10), 0.0)

    def test_reciprocal_rank_depth_cuts_off(self):
        self.assertEqual(reciprocal_rank(["x", "y", "a"], ["a"], 2), 0.0)

    def test_reciprocal_rank_uses_first_expected(self):
        self.assertEqual(reciprocal_rank(["x", "b", "a"], ["a", "b"], 10), 0.5)

    def test_must_mention_all_terms(self):
        self.assertTrue(must_mention_hit("Feed it ONCE A DAY with flour", ["once a day", "flour"]))

    def test_must_mention_missing_term(self):
        self.assertFalse(must_mention_hit("Feed it once a day", ["once a day", "flour"]))

    def test_must_mention_no_terms_is_a_hit(self):
        self.assertTrue(must_mention_hit("anything", []))


if __name__ == "__main__":
    unittest.main()
