# rag-eval-bench

A small bench that scores two search methods (BM25 and TF-IDF cosine) on a frozen question set, then fails the build if a score drops below a pass bar.

## What this demo shows

1. Two search methods, written from scratch with only the Python standard library, are scored with recall@k, MRR and a "must-mention" hit rate on 16 frozen questions over 14 short synthetic documents.
2. A runner prints a table and exits with a non-zero code when a pass bar is missed, so a CI job can block a bad change. The question set is locked by a checksum, so you cannot quietly make the questions easier.
3. Extra guards cover made-up citations (an answer that cites a document id that is not in the corpus fails the run) and noisy results (each method runs several times with a seeded tie-breaker, and must pass in most runs).

## How to run

You need Python 3.11 or newer. There is nothing to install. The bench uses no packages, no network and no API key.

```
python -m unittest discover -s tests -v
python -m rag_eval_bench
```

On Linux or macOS, check the exit code with `echo $?`. In PowerShell, use `$LASTEXITCODE`.

Expected output of the second command:

```
Scores are the mean over 5 runs (seed 20260101). Table covers a tiny synthetic corpus.

retriever  runs ok  recall@3  mrr    must-mention  fake cites  verdict
---------  -------  --------  -----  ------------  ----------  -------
bm25       5/5      0.969     0.958  0.938         0           PASS
tfidf      5/5      0.969     0.969  0.938         0           PASS
overlap    3/5      0.956     0.960  0.938         0           PASS

Details:
  overlap run 1: recall_at_k 0.938 is below the bar 0.950
  overlap run 4: recall_at_k 0.938 is below the bar 0.950

GATE: PASS
```

Exit codes: `0` all bars met, `1` a bar was missed, `2` the data or config is bad (for example, the frozen files changed).

### See the gate fail

```
python -m rag_eval_bench --config data/config_strict.json
```

This config asks for a perfect score. The run prints the misses and exits with code 1.

### Change the question set on purpose

Edit `data/questions.json`, then record the new checksum and commit both files together:

```
python -m rag_eval_bench freeze
```

If you edit the file and skip this step, the run stops with exit code 2 before any scoring.

## How it works

| Piece | File |
| --- | --- |
| BM25, TF-IDF cosine, and a coarse word-overlap method | `rag_eval_bench/retrievers.py` |
| recall@k, MRR, must-mention check | `rag_eval_bench/metrics.py` |
| Corpus loading and the checksum freeze | `rag_eval_bench/data.py` |
| Runs, pass bars, majority rule, fake-citation check | `rag_eval_bench/gate.py` |
| Command line | `rag_eval_bench/cli.py` |
| Pass bars and run settings | `data/config.json` |

The "answer" in this bench is simple. It is the text of the first three retrieved documents, citing their ids. There is no language model. The bench checks two things about it: do the must-mention terms appear, and does every cited id exist in the corpus.

Tests include planted regressions. One retriever returns its ranking upside down. Another puts an id that does not exist at rank 1. The tests prove the gate fails for both.

## The majority-of-runs rule

Each method runs 5 times. Between runs, only the random tie-breaker changes (it orders documents that have equal scores). A run passes when it meets every bar. A method passes when more than half of its runs pass. In the table above, the coarse `overlap` method passes in 3 of 5 runs, so it passes the gate even though two runs missed the recall bar. Its bar was set to show this rule, which is explained in `DECISIONS.md`.

One rule has no vote: any made-up citation in any run fails the method.

## What this does NOT prove

- The corpus is tiny: 14 short synthetic documents. The scores say nothing about real systems, real documents or real users.
- The questions are easy. Both methods score above 0.9, so the numbers cannot rank one method above the other in any useful way.
- 16 questions is too few for a small score gap to mean anything.
- The pass bars were chosen after looking at the scores. They guard against a drop from this baseline. They do not say a score is "good".
- Word-matching methods like these cannot handle meaning. A question that shares no words with its document will miss.
- There is no stemming, so "springs" will not match "spring", and spelling variants such as "tyre" and "tire" do not match.
- The pass bars in `data/config.json` are not frozen. Only the questions and the corpus are. A change to the bars needs a code review to be caught.
- The fake-citation check only catches ids that are not in the corpus. It does not check that a citation is the right one.
- The must-mention check is plain text matching. It does not judge whether an answer is correct.
- No language model is used, so nothing here measures answer writing.

AI-assisted, with tests and review gates.

## Layout

```
rag_eval_bench/   source
data/             synthetic corpus, frozen questions, checksum file, configs
tests/            unit tests (standard library unittest)
DECISIONS.md      why things are the way they are
DATA.md           where the data comes from
```

Licence: MIT (see `LICENSE`).
