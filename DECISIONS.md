# Decision log

Each entry has the decision, the reason, and the option I did not take.

## 1. Standard library only

- **Decision:** No third-party packages. BM25, TF-IDF and the tests use only the standard library.
- **Why:** The code runs anywhere Python 3.11 runs, with nothing to install. A reader can see every line of the scoring.
- **Rejected:** Using a search library for BM25. It would hide the formulas, which are the point of the demo.

## 2. Synthetic corpus

- **Decision:** I wrote 14 short documents myself and labelled them synthetic.
- **Why:** There is no licence question, no download, and the data is small enough to read in a few minutes.
- **Rejected:** Public-domain book text. It needs a source and licence record, and bigger text would not make the demo clearer.

## 3. Freeze the questions with a checksum, and the corpus too

- **Decision:** `questions.sha256` stores a digest of `questions.json` and `corpus.json`. A mismatch stops the run with exit code 2.
- **Why:** The easiest way to pass a gate is to change the test. A frozen file makes that a visible, deliberate act that needs a second step (`freeze`) and shows up in a code review. The corpus is frozen too, because changing a document moves the scores in the same way.
- **Rejected:** Freezing only the questions. Rejected because a corpus edit can hide a regression just as well.

## 4. Line endings are normalised before hashing

- **Decision:** `\r\n` becomes `\n` before the digest is computed.
- **Why:** Git on Windows can rewrite line endings on checkout. Without this, the same data would pass on Linux and fail on Windows.
- **Rejected:** Hashing raw bytes. Rejected because it makes CI fail for a reason that has nothing to do with the data.

## 5. The "answer" is the retrieved text

- **Decision:** An answer is the text of the first 3 retrieved documents plus the list of ids it cites. There is no language model.
- **Why:** The demo is about retrieval and gating, and it must run with no API key, no GPU and no cost. The two checks (must-mention terms, citations exist) work the same on any answer text.
- **Rejected:** A stub that writes a fluent answer. It would add code but no new test of the gate.

## 6. Fake-citation check is a hard rule

- **Decision:** An answer that cites an id not in the corpus fails the method. Any such citation in any run fails it, even if most runs pass.
- **Why:** A made-up source is a different kind of fault from a low score. It should not be averaged away by a majority vote. The planted-regression test shows a single bad citation in 1 of 80 answers still fails the gate.
- **Rejected:** Treating fake citations as one more bar with a threshold. Rejected because "no more than 2 percent fake sources" is not a rule I would defend.

## 7. Majority of runs, with a seeded tie-breaker

- **Decision:** Each retriever runs 5 times. The only thing that changes between runs is a random tie-breaker, seeded from config seed * 1000 + the run number. A retriever passes when more than half of its runs pass.
- **Why:** Equal scores have no natural order, so results can wobble. A fixed seed makes the wobble repeatable (the same command prints the same table), while the majority rule stops one unlucky ordering from failing the build.
- **Rejected:** Requiring every run to pass. Rejected because it fails on ordering noise. Also rejected: an unseeded tie-breaker, because CI would give different answers on different days.

## 8. A coarse third method, with a bar chosen to show the rule

- **Decision:** I added `overlap`, a method that scores a document by how many query words it contains. It has many ties. Its recall bar (0.95) sits between its worst run (0.938) and its best (0.969), so the shipped config shows 3 of 5 runs passing.
- **Why:** BM25 and TF-IDF almost never tie on this corpus, so they give the same result in every run and cannot show the majority rule. The third method can.
- **Honest note:** I picked that bar after seeing the scores, on purpose, to make the rule visible. A different seed gives different run counts. A test checks only that the count is above half and below all, not the exact number.
- **Rejected:** Adding random noise to BM25 scores to fake instability. That would be a made-up problem.

## 9. Pass bars are minimums set just under the baseline

- **Decision:** Bars live in `data/config.json`, one set per retriever. BM25 and TF-IDF need at least 0.90 recall@3, 0.90 MRR and 0.85 must-mention hit rate.
- **Why:** A bar that sits close under today's score catches a real drop. A bar far under it would let a large drop through.
- **Rejected:** One shared bar for every retriever. Rejected because a coarse method and a weighted one do not have the same baseline.

## 10. Metric definitions

- **Decision:** recall@k is the share of expected documents found in the first k results (k = 3). MRR is 1 divided by the rank of the first expected document in the first 10 results. The must-mention hit rate is the share of questions where every term appears in the answer text, ignoring case.
- **Why:** These are the common definitions, and they are easy to check by hand. The unit tests do exactly that.
- **Rejected:** A partial-credit must-mention score (share of terms found). Rejected because one missing term usually means the answer is wrong.

## 11. Three exit codes

- **Decision:** 0 pass, 1 a bar failed, 2 bad data or config.
- **Why:** CI can tell "the quality dropped" from "the setup is broken". Both stop the build.
- **Rejected:** Exit 1 for everything. Rejected because it hides which kind of failure happened.

## 12. Planted regressions in the tests

- **Decision:** Tests use a retriever that returns its ranking upside down, a retriever that puts an unknown id first, and one that fails only in chosen runs. The gate must fail (or, for the flaky one, follow the majority rule) for each.
- **Why:** A gate that has never been seen to fail proves little. These tests are the evidence that it can fail.
- **Rejected:** Only testing that the good path passes.

## 13. No stemming, a short stopword list

- **Decision:** Tokens are lowercase words with a small stopword list. No stemming.
- **Why:** It keeps the text step short and readable. The cost is that "tyre" will not match "tire" and "springs" will not match "spring". The README lists this as a limit.
- **Rejected:** Adding a stemmer. It would take more code than the retrievers themselves.
