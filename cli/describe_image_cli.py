import argparse
import mimetypes
from lib.llm_utils import LLM

def main() -> None:
    llm = LLM()
    parser = argparse.ArgumentParser(description="image cli search")

    parser.add_argument("--image", help="image file")
    parser.add_argument("--query", help="query on the image")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, 'rb') as file:
        data = file.read()

    res = llm.read_image(args.query, data, mime)

    print(f"Rewritten query: {res['content'].strip()}")
    print(f"Total tokens: {res['usage']}")


if __name__ == "__main__":
    main()
