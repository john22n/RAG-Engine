import argparse
from lib.rag import rag_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="query for rag")

    summarize_parser = subparsers.add_parser("summarize", help="summairze with llm")
    summarize_parser.add_argument("query", type=str, help="query to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="set result limit")

    citation_parser = subparsers.add_parser("citations", help="cite llm responses")
    citation_parser.add_argument("query", type=str, help="query to cite ")
    citation_parser.add_argument("--limit", type=int, default=5, help="set res limit")

    question_parser = subparsers.add_parser("question", help="question for llm to answer")
    question_parser.add_argument("query", type=str, help="query to llm")
    question_parser.add_argument("--limit", type=int, default=5, help="limit answers")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rag_command(query)
            print(f"Search Results:")
            for res in results['search_results']:
                print(f"- {res}")

            print(f"RAG Response:")
            print(f"{results['rag_results']}")

        case "summarize":
            results = rag_command(args.query, args.limit)
            print(f"Search Results:")
            for res in results['search_results']:
                print(f"- {res}")

            print("LLM Summary:")
            print(f"{results['summary']}")

        case "citations":
            results = rag_command(args.query, args.limit, args.command)
            print(f"Search Results:")
            for res in results['search_results']:
                print(f"- {res}")

            print("LLM Answer")
            print(f"{results['citation']}")

        case "question":
            results = rag_command(args.query, args.limit, args.command)
            print(f"Search Results:")
            for res in results['search_results']:
                print(f"- {res}")

            print("Answer")
            print(f"{results['question']}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

## not implementing conflict resolution
