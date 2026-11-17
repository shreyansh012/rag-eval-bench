import contextlib
import io
import json
import os
import shutil
import unittest

import helpers
from rag_eval_bench import cli


def run_cli(argv, factory=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv, retriever_factory=factory)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_healthy_run_exits_zero_and_prints_a_table(self):
        code, out, _ = run_cli([])
        self.assertEqual(code, 0)
        self.assertIn("GATE: PASS", out)
        for name in ("bm25", "tfidf", "overlap"):
            self.assertIn(name, out)

    def test_planted_regression_exits_non_zero(self):
        def broken(docs):
            return [helpers.ReversedBM25(docs)]

        code, out, _ = run_cli([], factory=broken)
        self.assertEqual(code, 1)
        self.assertIn("GATE: FAIL", out)
        self.assertIn("below the bar", out)

    def test_strict_example_config_fails_the_gate(self):
        code, out, _ = run_cli(["--config", os.path.join(helpers.DATA_DIR, "config_strict.json")])
        self.assertEqual(code, 1)
        self.assertIn("GATE: FAIL", out)

    def test_made_up_id_exits_non_zero_and_names_it(self):
        code, out, _ = run_cli([], factory=lambda docs: [helpers.MadeUpIdBM25(docs)])
        self.assertEqual(code, 1)
        self.assertIn("d99-not-real", out)

    def test_changed_questions_stop_the_run_before_scoring(self):
        tmp, d = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(d, "questions.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["questions"][0]["must_mention"] = ["anything easy"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        code, out, err = run_cli(["--data-dir", d])
        self.assertEqual(code, 2)
        self.assertIn("changed since it was frozen", err)
        self.assertNotIn("GATE", out)

    def test_freeze_then_run_works_after_a_deliberate_change(self):
        tmp, d = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(d, "questions.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["questions"].append({"id": "q99", "question": "How do I tune a guitar string?",
                                 "expected_ids": ["d14"], "must_mention": ["fret"]})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        self.assertEqual(run_cli(["--data-dir", d])[0], 2)
        self.assertEqual(run_cli(["freeze", "--data-dir", d])[0], 0)
        self.assertEqual(run_cli(["--data-dir", d])[0], 0)

    def test_missing_config_exits_two(self):
        code, _, err = run_cli(["--config", os.path.join(helpers.ROOT, "no-such.json")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_deleted_data_file_exits_two(self):
        tmp, d = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        os.remove(os.path.join(d, "corpus.json"))
        self.assertEqual(run_cli(["--data-dir", d])[0], 2)
        self.assertEqual(run_cli(["freeze", "--data-dir", d])[0], 2)
        self.assertEqual(run_cli(["freeze", "--data-dir", os.path.join(d, "nope")])[0], 2)

    def test_checksum_file_that_is_not_text_exits_two(self):
        tmp, d = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with open(os.path.join(d, "questions.sha256"), "wb") as f:
            f.write(bytes([0xff, 0xfe, 0x80, 0x81]))
        self.assertEqual(run_cli(["--data-dir", d])[0], 2)

    def test_wrong_shaped_data_exits_two(self):
        bad = [("corpus.json", {"documents": ["x"]}),
               ("questions.json", {"questions": ["x"]}),
               ("questions.json", {"questions": [{"id": "q1", "question": "a", "must_mention": [],
                                                  "expected_ids": [["d01"]]}]})]
        for name, content in bad:
            tmp, d = helpers.copy_data_dir()
            self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
            with open(os.path.join(d, name), "w", encoding="utf-8", newline="") as f:
                json.dump(content, f)
            run_cli(["freeze", "--data-dir", d])
            code, _, err = run_cli(["--data-dir", d])
            self.assertEqual(code, 2, (name, content))
            self.assertIn("ERROR", err)

    def test_wrong_shaped_bars_exit_two(self):
        tmp, d = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with open(os.path.join(helpers.DATA_DIR, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        path = os.path.join(d, "config.json")
        for bars in ({"bm25": 0.9}, {"bm25": ["mrr"]}, {"bm25": {}, "tfidf": {}, "overlap": {}}):
            cfg["bars"] = bars
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            self.assertEqual(run_cli(["--data-dir", d])[0], 2, bars)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[1, 2]")
        self.assertEqual(run_cli(["--data-dir", d])[0], 2)


if __name__ == "__main__":
    unittest.main()
