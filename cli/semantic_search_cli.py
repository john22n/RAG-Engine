import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model")

    embed_parser = subparsers.add_parser("embed_text", help="Verify model")
    embed_parser.add_argument("text", type=str, help="text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify model")
    args = parser.parse_args()

    match args.command:
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
