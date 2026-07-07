from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_app.core.rag import KnowledgeBase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Campus RAG Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index one or more documents")
    ingest_parser.add_argument("paths", nargs="+", type=Path)

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=3)

    subparsers.add_parser("stats", help="Show index status")

    reset_parser = subparsers.add_parser("reset", help="Clear the local index")
    reset_parser.add_argument("--uploads", action="store_true", help="Also remove uploaded document copies")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    kb = KnowledgeBase()

    if args.command == "ingest":
        results = [kb.add_document(path) for path in args.paths]
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        result = kb.ask(args.question, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "stats":
        print(json.dumps(kb.health(), ensure_ascii=False, indent=2))
        return

    if args.command == "reset":
        print(json.dumps(kb.reset(clear_uploads=args.uploads), ensure_ascii=False, indent=2))
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
