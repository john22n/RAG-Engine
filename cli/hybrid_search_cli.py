import argparse

from lib.hybrid_search import normalize, search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available coammands")

    normalize_parser = subparsers.add_parser("normalize", help="normalize data points")
    normalize_parser.add_argument("numbers", nargs='*', type=float, help="accept n amount of numbers for normalization")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="hybrid search using weights")
    weighted_search_parser.add_argument("query", type=str, help="search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="alpha weight")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="limit results")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize(args.numbers)
            for score in scores:
                print(f"* {score}")

        case "weighted-search":
            search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
