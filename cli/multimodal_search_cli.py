import argparse
from lib.multimodal_search import verify_image_embeddings, image_search

def main() -> None:
    parser = argparse.ArgumentParser(description="image verifier CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify_image_embedding", help="verify image embeddings")
    verify_parser.add_argument("image_path", type=str, help="image path to validate")

    search_images_parser = subparser.add_parser("image_search", help="find res based on image")
    search_images_parser.add_argument("image_path", type=str, help="image used in search")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embeddings(args.image_path)

        case "image_search":
            image_search(args.image_path)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
