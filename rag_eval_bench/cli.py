"""Command line entry point.

    python -m rag_eval_bench           run the gate (exit 0 pass, 1 bars failed, 2 bad data or config)
    python -m rag_eval_bench freeze    record new checksums after an on-purpose data change
"""

import argparse
import os
import sys

from . import data
from .gate import ConfigError, format_report, load_config, run_gate
from .retrievers import default_retrievers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data")


def main(argv=None, retriever_factory=None):
    parser = argparse.ArgumentParser(prog="rag_eval_bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["run", "freeze"], default="run")
    parser.add_argument("--data-dir", default=DEFAULT_DATA, help="folder with corpus, questions, config")
    parser.add_argument("--config", default=None, help="config file (default: <data-dir>/config.json)")
    args = parser.parse_args(argv)

    try:
        if args.command == "freeze":
            for line in data.freeze(args.data_dir):
                print(line)
            print("Checksums written to %s" % os.path.join(args.data_dir, data.CHECKSUM_FILE))
            return 0

        data.verify_frozen(args.data_dir)
        docs = data.load_corpus(args.data_dir)
        questions = data.load_questions(args.data_dir, {d["id"] for d in docs})
        cfg = load_config(args.config or os.path.join(args.data_dir, "config.json"))
        factory = retriever_factory or default_retrievers
        gate = run_gate(docs, questions, cfg, factory(docs))
    except (data.DataError, ConfigError, OSError, ValueError, TypeError, AttributeError) as err:
        print("ERROR: %s" % err, file=sys.stderr)
        return 2

    print(format_report(gate))
    return 0 if gate.passed else 1
