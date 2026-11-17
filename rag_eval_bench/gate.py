"""The CI gate: run each retriever several times, check the pass bars."""

import json
import random
from dataclasses import dataclass, field

from .metrics import must_mention_hit, recall_at_k, reciprocal_rank

BAR_METRICS = ("recall_at_k", "mrr", "must_mention_hit_rate")


class ConfigError(Exception):
    """The config file is missing something or has a bad value."""


@dataclass
class Config:
    k: int
    answer_k: int
    rank_depth: int
    runs: int
    seed: int
    bars: dict


def load_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as err:
        raise ConfigError("cannot read config %s: %s" % (path, err))
    if not isinstance(raw, dict):
        raise ConfigError("bad config: the top level must be an object")
    try:
        cfg = Config(
            k=int(raw["k"]),
            answer_k=int(raw["answer_k"]),
            rank_depth=int(raw["rank_depth"]),
            runs=int(raw["runs"]),
            seed=int(raw["seed"]),
            bars=dict(raw["bars"]),
        )
    except (KeyError, TypeError, ValueError) as err:
        raise ConfigError("bad config: %r" % (err,))
    if min(cfg.k, cfg.answer_k, cfg.rank_depth, cfg.runs) < 1:
        raise ConfigError("k, answer_k, rank_depth and runs must be at least 1")
    for name, bars in cfg.bars.items():
        if not isinstance(bars, dict):
            raise ConfigError("bars for %s must be an object" % name)
        unknown = set(bars) - set(BAR_METRICS)
        if unknown:
            raise ConfigError("unknown bar(s) for %s: %s" % (name, sorted(unknown)))
        missing = set(BAR_METRICS) - set(bars)
        if missing:
            raise ConfigError("missing bar(s) for %s: %s" % (name, sorted(missing)))
        for metric, value in bars.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ConfigError("bar %s.%s must be a number from 0 to 1" % (name, metric))
    return cfg


# --- answers and the fake citation check -----------------------------------------

@dataclass
class Answer:
    text: str
    cites: list


def build_answer(ranked, docs_by_id, answer_k):
    """A plain answer: the text of the best-ranked documents, citing their ids.

    An id the corpus does not know contributes no text but is still cited.
    That is how a made-up id gets caught.
    """
    cites = [doc_id for doc_id, _ in ranked[:answer_k]]
    parts = []
    for doc_id in cites:
        doc = docs_by_id.get(doc_id)
        if doc:
            parts.append(doc["title"] + ". " + doc["text"])
    return Answer(text="\n".join(parts), cites=cites)


def check_citations(cites, corpus_ids):
    """Return the cited ids that do not exist in the corpus. Empty list means clean."""
    return [c for c in cites if c not in corpus_ids]


# --- one run, many runs ------------------------------------------------------

def evaluate_run(retriever, docs_by_id, questions, cfg, rng):
    recalls, rrs, hits, fakes = [], [], [], []
    for q in questions:
        ranked = retriever.rank(q["question"], rng)
        ids = [doc_id for doc_id, _ in ranked]
        recalls.append(recall_at_k(ids, q["expected_ids"], cfg.k))
        rrs.append(reciprocal_rank(ids, q["expected_ids"], cfg.rank_depth))
        answer = build_answer(ranked, docs_by_id, cfg.answer_k)
        hits.append(1.0 if must_mention_hit(answer.text, q["must_mention"]) else 0.0)
        for bad in check_citations(answer.cites, docs_by_id):
            fakes.append((q["id"], bad))
    n = len(questions)
    return {
        "recall_at_k": sum(recalls) / n,
        "mrr": sum(rrs) / n,
        "must_mention_hit_rate": sum(hits) / n,
        "fake_citations": fakes,
    }


def failed_bars(metrics, bars):
    """List of (metric, value, bar) for each bar this run missed."""
    out = []
    for metric, bar in sorted(bars.items()):
        if metrics[metric] < bar:
            out.append((metric, metrics[metric], bar))
    return out


def majority_passes(passed_runs, total_runs):
    """True when strictly more than half of the runs passed."""
    return passed_runs * 2 > total_runs


@dataclass
class RetrieverResult:
    name: str
    runs: list = field(default_factory=list)       # metrics dict per run
    run_ok: list = field(default_factory=list)     # bool per run
    run_failures: list = field(default_factory=list)
    fake_total: int = 0
    passed: bool = False

    @property
    def passed_runs(self):
        return sum(self.run_ok)

    def mean(self, metric):
        return sum(r[metric] for r in self.runs) / len(self.runs)


@dataclass
class GateResult:
    results: list
    cfg: Config

    @property
    def passed(self):
        return all(r.passed for r in self.results)


def run_gate(documents, questions, cfg, retrievers):
    """Run every retriever cfg.runs times and apply the bars.

    Rules:
      * A run passes when every bar is met in that run.
      * A retriever passes when most of its runs pass (majority).
      * Any made-up citation in any run fails the retriever. No majority vote.
    """
    if not retrievers:
        raise ConfigError("no retrievers to run")
    docs_by_id = {d["id"]: d for d in documents}
    results = []
    for retriever in retrievers:
        bars = cfg.bars.get(retriever.name)
        if bars is None:
            raise ConfigError("no pass bars in config for retriever %r" % retriever.name)
        res = RetrieverResult(name=retriever.name)
        for run_index in range(cfg.runs):
            rng = random.Random(cfg.seed * 1000 + run_index)
            metrics = evaluate_run(retriever, docs_by_id, questions, cfg, rng)
            misses = failed_bars(metrics, bars)
            res.runs.append(metrics)
            res.run_failures.append(misses)
            res.run_ok.append(not misses and not metrics["fake_citations"])
            res.fake_total += len(metrics["fake_citations"])
        res.passed = majority_passes(res.passed_runs, cfg.runs) and res.fake_total == 0
        results.append(res)
    return GateResult(results=results, cfg=cfg)


# --- printing ----------------------------------------------------------------

def format_report(gate):
    cfg = gate.cfg
    head = ["retriever", "runs ok", "recall@%d" % cfg.k, "mrr", "must-mention", "fake cites", "verdict"]
    rows = []
    for r in gate.results:
        rows.append([
            r.name,
            "%d/%d" % (r.passed_runs, cfg.runs),
            "%.3f" % r.mean("recall_at_k"),
            "%.3f" % r.mean("mrr"),
            "%.3f" % r.mean("must_mention_hit_rate"),
            str(r.fake_total),
            "PASS" if r.passed else "FAIL",
        ])
    widths = [max(len(head[i]), *(len(row[i]) for row in rows)) for i in range(len(head))]

    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = [
        "Scores are the mean over %d runs (seed %d). Table covers %s."
        % (cfg.runs, cfg.seed, "a tiny synthetic corpus"),
        "",
        line(head),
        line(["-" * w for w in widths]),
    ]
    out.extend(line(row) for row in rows)
    problems = []
    for r in gate.results:
        for i, misses in enumerate(r.run_failures):
            for metric, value, bar in misses:
                problems.append("  %s run %d: %s %.3f is below the bar %.3f" % (r.name, i + 1, metric, value, bar))
        seen = set()
        for i, m in enumerate(r.runs):
            for qid, bad in m["fake_citations"]:
                if (qid, bad) not in seen:
                    seen.add((qid, bad))
                    problems.append("  %s run %d: question %s cites %r, which is not in the corpus" % (r.name, i + 1, qid, bad))
    if problems:
        out += ["", "Details:"] + problems
    out += ["", "GATE: " + ("PASS" if gate.passed else "FAIL")]
    return "\n".join(out)
