"""Shared test setup. Puts the repo root on the import path and builds fixtures."""

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_eval_bench import data  # noqa: E402
from rag_eval_bench.gate import load_config  # noqa: E402
from rag_eval_bench.retrievers import BM25  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")


def load_all():
    docs = data.load_corpus(DATA_DIR)
    questions = data.load_questions(DATA_DIR, {d["id"] for d in docs})
    cfg = load_config(os.path.join(DATA_DIR, "config.json"))
    return docs, questions, cfg


def copy_data_dir():
    """Copy the data folder into a fresh temp folder. Caller must clean up."""
    tmp = tempfile.mkdtemp(prefix="rag_eval_bench_test_")
    dest = os.path.join(tmp, "data")
    shutil.copytree(DATA_DIR, dest)
    return tmp, dest


class ReversedBM25(BM25):
    """Planted regression: a ranking that puts the worst documents first."""

    name = "bm25"

    def rank(self, query, rng):
        return list(reversed(super().rank(query, rng)))


class MadeUpIdBM25(BM25):
    """Planted regression: puts an id that is not in the corpus at rank 1."""

    name = "bm25"

    def rank(self, query, rng):
        return [("d99-not-real", 99.0)] + super().rank(query, rng)


class FlakyBM25(BM25):
    """Good in most runs, reversed (bad) in the runs listed in `bad_runs`.

    The bench asks one question after another within a run, so the run number
    is the call count divided by the number of questions.
    """

    name = "bm25"

    def __init__(self, documents, bad_runs, questions_per_run):
        super().__init__(documents)
        self.bad_runs = set(bad_runs)
        self.questions_per_run = questions_per_run
        self.calls = 0

    def rank(self, query, rng):
        run = self.calls // self.questions_per_run
        self.calls += 1
        ranked = super().rank(query, rng)
        return list(reversed(ranked)) if run in self.bad_runs else ranked


class PartlyReversedBM25(BM25):
    """Planted regression: reversed ranking for only the first few questions of each run."""

    name = "bm25"

    def __init__(self, documents, bad_questions, questions_per_run):
        super().__init__(documents)
        self.bad_questions = bad_questions
        self.questions_per_run = questions_per_run
        self.calls = 0

    def rank(self, query, rng):
        position = self.calls % self.questions_per_run
        self.calls += 1
        ranked = super().rank(query, rng)
        return list(reversed(ranked)) if position < self.bad_questions else ranked
