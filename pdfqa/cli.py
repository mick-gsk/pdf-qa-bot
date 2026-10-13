"""Command line: index PDFs once, then ask questions.

    python -m pdfqa.cli index handbook.pdf policies.pdf
    python -m pdfqa.cli ask "How many vacation days do I get?"
    python -m pdfqa.cli eval examples/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evaluate
from .index import Index
from .ingest import chunk_pdf
from .llm import get_embedder, get_llm
from .qa import answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdfqa", description=__doc__.splitlines()[0])
    parser.add_argument("--index", default="data/index.json", help="index file (default: data/index.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_index = sub.add_parser("index", help="add PDFs to the index")
    p_index.add_argument("pdfs", nargs="+", type=Path)
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=4, help="number of excerpts to use")
    p_eval = sub.add_parser("eval", help="score retrieval and answers against a case file")
    p_eval.add_argument("cases", type=Path)
    p_eval.add_argument("-k", type=int, default=4, help="number of excerpts to use")
    p_eval.add_argument("-v", "--verbose", action="store_true", help="list failed questions")
    args = parser.parse_args(argv)

    if args.command == "eval":
        report = evaluate.run(args.cases, get_llm(), get_embedder(), k=args.k)
        print(json.dumps(report.summary(), indent=2))
        if args.verbose:
            print("\n".join(report.failures), file=sys.stderr)
        return 0

    index_path = Path(args.index)
    embed = get_embedder()
    index = Index.load(index_path, embed) if index_path.exists() else Index(embed=embed)

    if args.command == "index":
        for pdf in args.pdfs:
            chunks = chunk_pdf(pdf)
            index.add(chunks)
            print(f"{pdf.name}: {len(chunks)} chunks")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save(index_path)
        return 0

    if not len(index):
        print("Index is empty. Run: pdfqa index <file.pdf>", file=sys.stderr)
        return 1
    result = answer(args.question, index, get_llm(), k=args.k)
    print(result.text)
    for source in result.sources:
        print(f"  [{source.n}] {source.citation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
