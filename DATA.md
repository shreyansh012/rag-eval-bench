# Data

All data in this repository is synthetic. I wrote it for this demo. None of it comes from a real source, so there is no source URL and no outside licence to record.

| File | What it is |
| --- | --- |
| `data/corpus.json` | 14 short invented how-to documents (baking, bikes, tides, knitting and similar). Each has an id (`d01` to `d14`), a title and a few sentences. |
| `data/questions.json` | 16 invented questions. Each lists the expected document ids and the terms an answer must mention. |
| `data/questions.sha256` | SHA-256 checksums of the two files above. See below. |
| `data/config.json` | Run settings and pass bars. |
| `data/config_strict.json` | The same settings with every bar at 1.0. It exists so you can watch the gate fail. |

## Do not use the documents as advice

The documents are plain sketches, not checked guidance. Do not rely on them for baking, bike repair, hiking or anything else.

## Licence of the data

The synthetic data is released under the same MIT licence as the code (see `LICENSE`).

## The checksum file

`questions.sha256` holds one line per frozen file: the SHA-256 digest, two spaces, then the file name. Before scoring, the runner recomputes each digest. Windows line endings are turned into Unix ones first, so a Windows checkout and a Linux checkout give the same digest. If a digest differs, the run stops with exit code 2.

To change the data on purpose, edit the file and run `python -m rag_eval_bench freeze`.

## Why the questions are easy

The questions were written by the same person who wrote the documents, so they share many words with their target documents. That makes the scores high. See "What this does NOT prove" in `README.md`.
