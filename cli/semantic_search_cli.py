import argparse
from lib.search_utils import load_movies
from lib.semantic_search import (
        SemanticSearch,
        verify_model,
        embed_text,
        verify_embeddings,
        embed_query_text,
        search_query,
        chunk_text,
        semantic_chunk_text,
        chunk,
        embed_chunks,
        search_chunks,
    )

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

    chunk_parser = subparsers.add_parser("chunk", help="chunk the text")
    chunk_parser.add_argument("text",type=str,  help="chunk the text")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="chunk the text")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="overlap text")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="sematic chunk text")
    semantic_chunk_parser.add_argument("text", type=str, help="text to sem chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="option to config size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="overlap config")

    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="embed document chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="embed document chunks")
    search_chunked_parser.add_argument("text",type=str, help="embed document chunks")
    search_chunked_parser.add_argument("--limit",type=int,default=5, help="embed document chunks")

    args = parser.parse_args()

    match args.command:
        case "search_chunked":
            search_chunks(args.text, args.limit)

        case "embed_chunks":
            embed_chunks()

        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)

        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)

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
