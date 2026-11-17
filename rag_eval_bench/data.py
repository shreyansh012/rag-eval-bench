"""Load the corpus and the frozen question set, and check the freeze."""

import hashlib
import json
import os


class DataError(Exception):
    """The data on disk is not what the bench expects."""


CHECKSUM_FILE = "questions.sha256"
FROZEN_FILES = ("questions.json", "corpus.json")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as err:
        raise DataError("cannot read %s: %s" % (path, err))


def file_digest(path):
    """SHA-256 of the file with Windows line endings turned into Unix ones.

    That way a checkout on Windows gives the same digest as one on Linux.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as err:
        raise DataError("cannot read %s: %s" % (path, err))
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def read_checksums(path):
    sums = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                digest, _, name = line.partition("  ")
                sums[name.strip()] = digest.strip()
    except (OSError, ValueError) as err:
        raise DataError("cannot read checksum file %s: %s" % (path, err))
    return sums


def verify_frozen(data_dir):
    """Raise DataError if a frozen file differs from the recorded checksum."""
    sums = read_checksums(os.path.join(data_dir, CHECKSUM_FILE))
    for name in FROZEN_FILES:
        want = sums.get(name)
        if want is None:
            raise DataError("no checksum recorded for %s" % name)
        have = file_digest(os.path.join(data_dir, name))
        if have != want:
            raise DataError(
                "%s changed since it was frozen (recorded %s..., now %s...). "
                "If the change is on purpose, run: python -m rag_eval_bench freeze"
                % (name, want[:12], have[:12])
            )


def freeze(data_dir):
    """Write the checksum file for the current data. Returns the lines written."""
    lines = ["%s  %s" % (file_digest(os.path.join(data_dir, n)), n) for n in FROZEN_FILES]
    with open(os.path.join(data_dir, CHECKSUM_FILE), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def load_corpus(data_dir):
    raw = _read_json(os.path.join(data_dir, "corpus.json"))
    docs = raw.get("documents") if isinstance(raw, dict) else None
    if not docs:
        raise DataError("corpus.json has no documents")
    seen = set()
    if not isinstance(docs, list):
        raise DataError("corpus.json: documents must be a list")
    for d in docs:
        if not isinstance(d, dict):
            raise DataError("document must be an object: %r" % (d,))
        for key in ("id", "title", "text"):
            if not isinstance(d.get(key), str) or not d[key]:
                raise DataError("document is missing %r: %r" % (key, d))
        if d["id"] in seen:
            raise DataError("duplicate document id %s" % d["id"])
        seen.add(d["id"])
    return docs


def load_questions(data_dir, corpus_ids):
    raw = _read_json(os.path.join(data_dir, "questions.json"))
    qs = raw.get("questions") if isinstance(raw, dict) else None
    if not qs:
        raise DataError("questions.json has no questions")
    seen = set()
    if not isinstance(qs, list):
        raise DataError("questions.json: questions must be a list")
    for q in qs:
        if not isinstance(q, dict):
            raise DataError("question must be an object: %r" % (q,))
        if not isinstance(q.get("id"), str) or q["id"] in seen:
            raise DataError("bad or duplicate question id: %r" % (q.get("id"),))
        seen.add(q["id"])
        if not isinstance(q.get("question"), str) or not q["question"]:
            raise DataError("question %s has no text" % q["id"])
        exp = q.get("expected_ids")
        if not isinstance(exp, list) or not exp:
            raise DataError("question %s has no expected_ids" % q["id"])
        if not all(isinstance(e, str) for e in exp):
            raise DataError("question %s: expected_ids must be strings" % q["id"])
        missing = [e for e in exp if e not in corpus_ids]
        if missing:
            raise DataError("question %s expects unknown documents: %s" % (q["id"], missing))
        terms = q.get("must_mention")
        if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
            raise DataError("question %s needs a must_mention list" % q["id"])
    return qs
