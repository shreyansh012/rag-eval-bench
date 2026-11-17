import json
import os
import shutil
import unittest

import helpers
from rag_eval_bench import data


class FrozenDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.dir = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def qpath(self):
        return os.path.join(self.dir, "questions.json")

    def test_shipped_data_matches_checksum(self):
        data.verify_frozen(helpers.DATA_DIR)

    def test_changed_question_fails(self):
        with open(self.qpath(), encoding="utf-8") as f:
            raw = json.load(f)
        raw["questions"][0]["question"] += " Please."
        with open(self.qpath(), "w", encoding="utf-8", newline="\n") as f:
            json.dump(raw, f, indent=2)
        with self.assertRaises(data.DataError) as ctx:
            data.verify_frozen(self.dir)
        self.assertIn("questions.json changed", str(ctx.exception))

    def test_changed_corpus_fails(self):
        path = os.path.join(self.dir, "corpus.json")
        with open(path, "a", encoding="utf-8") as f:
            f.write(" ")
        with self.assertRaises(data.DataError):
            data.verify_frozen(self.dir)

    def test_freeze_accepts_a_deliberate_change(self):
        with open(self.qpath(), "a", encoding="utf-8") as f:
            f.write("\n")
        with self.assertRaises(data.DataError):
            data.verify_frozen(self.dir)
        data.freeze(self.dir)
        data.verify_frozen(self.dir)

    def test_windows_line_endings_give_the_same_digest(self):
        with open(self.qpath(), "rb") as f:
            raw = f.read()
        before = data.file_digest(self.qpath())
        with open(self.qpath(), "wb") as f:
            f.write(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertEqual(data.file_digest(self.qpath()), before)
        data.verify_frozen(self.dir)

    def test_missing_checksum_entry_fails(self):
        with open(os.path.join(self.dir, data.CHECKSUM_FILE), "w", encoding="utf-8") as f:
            f.write("")
        with self.assertRaises(data.DataError):
            data.verify_frozen(self.dir)


class ValidationTests(unittest.TestCase):
    def test_shipped_questions_only_expect_real_documents(self):
        docs, questions, _ = helpers.load_all()
        ids = {d["id"] for d in docs}
        self.assertGreaterEqual(len(questions), 10)
        for q in questions:
            self.assertTrue(set(q["expected_ids"]) <= ids)
            self.assertTrue(q["must_mention"])

    def test_question_with_unknown_expected_id_is_rejected(self):
        tmp, d = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(d, "questions.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["questions"][0]["expected_ids"] = ["d404"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        docs = data.load_corpus(d)
        with self.assertRaises(data.DataError):
            data.load_questions(d, {x["id"] for x in docs})

    def test_duplicate_document_id_is_rejected(self):
        tmp, d = helpers.copy_data_dir()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(d, "corpus.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["documents"].append(dict(raw["documents"][0]))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with self.assertRaises(data.DataError):
            data.load_corpus(d)


if __name__ == "__main__":
    unittest.main()
