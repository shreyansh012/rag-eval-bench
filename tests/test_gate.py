import dataclasses
import json
import os
import shutil
import tempfile
import unittest

import helpers
from rag_eval_bench.gate import (ConfigError, build_answer, check_citations, load_config,
                                 majority_passes, run_gate)
from rag_eval_bench.retrievers import BM25, TfidfCosine, WordOverlap, default_retrievers


ALL_BARS = {"recall_at_k": 0.5, "mrr": 0.5, "must_mention_hit_rate": 0.5}


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.questions, cls.cfg = helpers.load_all()

    # --- the healthy bench passes ---

    def test_shipped_retrievers_pass_the_shipped_bars(self):
        gate = run_gate(self.docs, self.questions, self.cfg, default_retrievers(self.docs))
        self.assertTrue(gate.passed, [(r.name, r.run_failures) for r in gate.results])

    def test_same_seed_gives_same_numbers(self):
        a = run_gate(self.docs, self.questions, self.cfg, [WordOverlap(self.docs)])
        b = run_gate(self.docs, self.questions, self.cfg, [WordOverlap(self.docs)])
        self.assertEqual([r["recall_at_k"] for r in a.results[0].runs],
                         [r["recall_at_k"] for r in b.results[0].runs])

    # --- planted regressions: the gate must catch them ---

    def test_gate_catches_a_reversed_ranking(self):
        gate = run_gate(self.docs, self.questions, self.cfg, [helpers.ReversedBM25(self.docs)])
        res = gate.results[0]
        self.assertFalse(gate.passed)
        self.assertEqual(res.passed_runs, 0)
        self.assertLess(res.mean("recall_at_k"), 0.5)

    def test_gate_catches_a_made_up_document_id(self):
        gate = run_gate(self.docs, self.questions, self.cfg, [helpers.MadeUpIdBM25(self.docs)])
        res = gate.results[0]
        self.assertFalse(gate.passed)
        self.assertEqual(res.fake_total, len(self.questions) * self.cfg.runs)
        self.assertEqual(res.passed_runs, 0)

    def test_shipped_bars_catch_a_modest_regression(self):
        # BM25 is wrong on only 3 of the 16 questions in every run.
        gate = run_gate(self.docs, self.questions, self.cfg,
                        [helpers.PartlyReversedBM25(self.docs, bad_questions=3,
                                                    questions_per_run=len(self.questions))])
        res = gate.results[0]
        self.assertFalse(gate.passed)
        self.assertEqual(res.passed_runs, 0)
        self.assertGreater(res.mean("recall_at_k"), 0.5)  # a modest drop, not a collapse

    def test_strict_bars_fail_a_healthy_retriever(self):
        strict = dataclasses.replace(
            self.cfg, bars={"bm25": {"recall_at_k": 1.0, "mrr": 1.0, "must_mention_hit_rate": 1.0}})
        gate = run_gate(self.docs, self.questions, strict, [BM25(self.docs)])
        self.assertFalse(gate.passed)

    def test_no_retrievers_is_an_error(self):
        with self.assertRaises(ConfigError):
            run_gate(self.docs, self.questions, self.cfg, [])

    # --- fake citation check ---

    def test_check_citations(self):
        ids = {"d01", "d02"}
        self.assertEqual(check_citations(["d01", "d02"], ids), [])
        self.assertEqual(check_citations(["d01", "d77"], ids), ["d77"])

    def test_answer_with_unknown_id_has_no_text_from_it(self):
        by_id = {d["id"]: d for d in self.docs}
        answer = build_answer([("zzz", 1.0), ("d01", 0.5)], by_id, 2)
        self.assertEqual(answer.cites, ["zzz", "d01"])
        self.assertIn("sourdough", answer.text.lower())
        self.assertEqual(check_citations(answer.cites, set(by_id)), ["zzz"])

    def test_one_fake_citation_fails_even_when_most_runs_pass(self):
        class OneBadQuestion(BM25):
            name = "bm25"
            calls = 0

            def rank(self, query, rng):
                OneBadQuestion.calls += 1
                ranked = super().rank(query, rng)
                if OneBadQuestion.calls == 1:  # first question of first run only
                    return [("d99", 9.0)] + ranked
                return ranked

        gate = run_gate(self.docs, self.questions, self.cfg, [OneBadQuestion(self.docs)])
        res = gate.results[0]
        self.assertEqual(res.fake_total, 1)
        self.assertTrue(majority_passes(res.passed_runs, self.cfg.runs))
        self.assertFalse(res.passed)

    # --- majority-of-runs rule ---

    def test_majority_rule_needs_more_than_half(self):
        self.assertTrue(majority_passes(3, 5))
        self.assertFalse(majority_passes(2, 5))
        self.assertFalse(majority_passes(2, 4))
        self.assertTrue(majority_passes(1, 1))

    def test_retriever_passes_when_most_runs_pass(self):
        flaky = helpers.FlakyBM25(self.docs, bad_runs={0, 1}, questions_per_run=len(self.questions))
        res = run_gate(self.docs, self.questions, self.cfg, [flaky]).results[0]
        self.assertEqual(res.run_ok, [False, False, True, True, True])
        self.assertTrue(res.passed)

    def test_retriever_fails_when_most_runs_fail(self):
        flaky = helpers.FlakyBM25(self.docs, bad_runs={0, 1, 2}, questions_per_run=len(self.questions))
        res = run_gate(self.docs, self.questions, self.cfg, [flaky]).results[0]
        self.assertEqual(res.passed_runs, 2)
        self.assertFalse(res.passed)

    def test_seeded_tie_breaker_demo_passes_by_majority_not_unanimity(self):
        res = run_gate(self.docs, self.questions, self.cfg, [WordOverlap(self.docs)]).results[0]
        self.assertTrue(res.passed)
        self.assertGreater(res.passed_runs, self.cfg.runs // 2)
        self.assertLess(res.passed_runs, self.cfg.runs)

    # --- config ---

    def test_missing_bars_for_a_retriever_is_an_error(self):
        cfg = dataclasses.replace(self.cfg, bars={})
        with self.assertRaises(ConfigError):
            run_gate(self.docs, self.questions, cfg, [TfidfCosine(self.docs)])

    def test_bad_config_values_are_rejected(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "c.json")
        good = {"k": 3, "answer_k": 3, "rank_depth": 10, "runs": 5, "seed": 1,
                "bars": {"bm25": ALL_BARS}}
        for change in ({"runs": 0}, {"bars": {"bm25": dict(ALL_BARS, mrr=1.5)}},
                       {"bars": {"bm25": dict(ALL_BARS, speed=0.5)}}, {"seed": None},
                       {"bars": {"bm25": 0.9}}, {"bars": {"bm25": ["mrr"]}},
                       {"bars": {"bm25": {}}}, {"bars": {"bm25": {"mrr": 0.5}}}):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dict(good, **change), f)
            with self.assertRaises(ConfigError, msg=str(change)):
                load_config(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(good, f)
        self.assertEqual(load_config(path).runs, 5)


if __name__ == "__main__":
    unittest.main()
