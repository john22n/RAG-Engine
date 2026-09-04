import argparse
import json
from lib.keyword_search import search_command, tokenize_text_helper, InvertedIndex


def build_command() -> None:
        Index = InvertedIndex()
        Index.build()
        Index.save()

def find_tf(doc_id: int, term: str) -> None:
    tokenized = tokenize_text_helper(term)
    idx = InvertedIndex()
    idx.load()
    count = idx.get_tf(doc_id, tokenized)
    print(f"term frequency: {count} for term: {term}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Term Frequency")

    tf_parser = subparsers.add_parser("tf", help="build movies cache")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term")

    args = parser.parse_args()

    match args.command:
        case "tf":
            find_tf(args.doc_id, args.term)

        case "search":
            #print search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            if results:
                for i, res in enumerate(results, 1):
                    print(f"{i}. ({res["id"]}) {res['title']} ")

        case "build":
            build_command()

        case _:
            parser.print_help()





if __name__ == "__main__":
    main()
