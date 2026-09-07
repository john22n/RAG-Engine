import argparse
import json
from lib.keyword_search import (
        search_command,
        tokenize_text_helper,
        InvertedIndex,
        build_command,
        tf_idf_command,
        tf_command,
        idf_command,
        bm25_idf_command,
        bm25_tf_command
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Term Frequency")

    tf_parser = subparsers.add_parser("tf", help="build movies cache")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term")

    idf_parser = subparsers.add_parser("idf", help="inverse document frequency")
    idf_parser.add_argument("term", type=str, help="term")

    tf_idf_parser = subparsers.add_parser("tfidf", help="get tf_idf value")
    tf_idf_parser.add_argument("doc_id", type=int, help="document id")
    tf_idf_parser.add_argument("term", type=str, help="term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get bm25 idf score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for given doc id and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="document id")
    bm25_tf_parser.add_argument("term", type=str, help="term to get bm25 tf score ")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", help="turnable bm25 k1 param")

    args = parser.parse_args()

    match args.command:
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")

        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF_IDF score of '{args.term}' in doc '{args.term}': {tf_idf:.2f}")

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of {args.term}' in docu '{args.doc_id}': {tf:.2f}")

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
