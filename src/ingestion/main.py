"""Command-line entrypoint.

Usage:
    python -m ingestion.main init-db
    python -m ingestion.main ingest
    python -m ingestion.main ingest --limit 50 --categories "rag" "vector search"
    python -m ingestion.main search --query "current advances in RAG systems"
"""
from __future__ import annotations

import argparse
import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Scholar -> PostgreSQL/pgvector ingestion pipeline."
    )
    parser.add_argument("command", choices=["init-db", "ingest", "search"])
    parser.add_argument("--categories", nargs="*", help="override the category list")
    parser.add_argument("--limit", type=int, help="papers per category")
    parser.add_argument("--query", help="query string for the 'search' command")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    # Imported here so `--help` doesn't require DB/model dependencies.
    from .db import init_db, session_scope

    if args.command == "init-db":
        init_db()
        print("Initialized extension, tables, and indexes.")
        return

    if args.command == "ingest":
        from .pipeline import IngestionPipeline

        init_db()
        stats = IngestionPipeline().run(
            categories=args.categories, papers_per_category=args.limit
        )
        print(f"Done: {stats}")
        return

    if args.command == "search":
        if not args.query:
            parser.error("--query is required for the 'search' command")
        from .retriever import HybridRetriever

        with session_scope() as session:
            results = HybridRetriever(session).search(args.query, top_k=args.top_k)
            for i, hit in enumerate(results, 1):
                title = hit.metadata.get("title", "?")
                print(f"\n[{i}] score={hit.score:.4f}  {title}")
                print(f"    {hit.content[:400].strip()}...")


if __name__ == "__main__":
    main()
