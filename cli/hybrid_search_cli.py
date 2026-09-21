import argparse

from lib.prompts.spell import spell_prompt
from lib.hybrid_search import normalize, search, rrf_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available coammands")

    normalize_parser = subparsers.add_parser("normalize", help="normalize data points")
    normalize_parser.add_argument("numbers", nargs='*', type=float, help="accept n amount of numbers for normalization")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="hybrid search using weights")
    weighted_search_parser.add_argument("query", type=str, help="search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="alpha weight")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="limit results")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="return searches using reciprocal rank fusion algo")
    rrf_search_parser.add_argument("query", type=str, help="search query to be searched")
    rrf_search_parser.add_argument("-k", type=int, help="optinal k param for rrf algo")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="limit results")
    rrf_search_parser.add_argument("--enhance", type=str, choices=['spell', 'rewrite', 'expand'], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=['individual', 'batch', 'cross_encoder'], help="reranking method to use")
    rrf_search_parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True, help="LLM evaluation of results")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize(args.numbers)
            for score in scores:
                print(f"* {score}")

        case "weighted-search":
            search(args.query, args.alpha, args.limit)

        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
