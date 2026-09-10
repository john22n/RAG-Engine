import argparse
from lib.search_utils import load_movies
from lib.semantic_search import (
        SemanticSearch,
        verify_model,
        embed_text,
        verify_embeddings,
        embed_query_text,
    )


def search_query(query, limit):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_res = ss.search(query, limit)

    if search_res is not None:
        for i, res in enumerate(search_res):
            print(f"{i + 1}. {res["title"]} (score: {res["score"]:.4f})  {res["description"].strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model")

    embed_parser = subparsers.add_parser("embed_text", help="Verify model")
    embed_parser.add_argument("text", type=str, help="text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify model")

    embed_query_parser = subparsers.add_parser("embed_query", help="embed the query")
    embed_query_parser.add_argument("query", type=str, help="query to be embeded")

    search_parser = subparsers.add_parser("search", help="search embedded movies")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--limit", type=int ,default=5, help="optional limit")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_query(args.query, args.limit)

        case "embed_query":
            embed_query_text(args.query)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "verify":
            verify_model()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
