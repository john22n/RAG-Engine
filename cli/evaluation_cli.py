import argparse
from lib.search_utils import load_golden_dataset
from lib.hybrid_search import rrf_search_eval

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
            "--limit",
            type=int,
            default=5,
            help="Number of results to evaluate (k for precision@k, recall@k"
            )
    args = parser.parse_args()
    limit = args.limit

    #evaluation logic here
    golden_dataset = load_golden_dataset()

    relevant: int = 0

    for test_case in golden_dataset['test_cases']:
        relevant_titles = []
        query = test_case['query']
        relavent_docs = test_case['relevant_docs']
        rrf_res = rrf_search_eval(query, 60, args.limit)
        titles = [item[1]['title'] for item in rrf_res]

        for doc in relavent_docs:
            if doc in titles:
                relevant = 1
                relevant_titles.append(doc)

        percision: float = relevant / args.limit
        printed_titles = ", ".join(titles)
        printed_relevant_titles = ", ".join(relevant_titles)

        print(f"k={args.limit}")
        print(f"- Query: {query}")
        print(f"  - Precision@{args.limit}: {percision:.4}")
        print(f"  - Retrieved: {printed_titles}")
        print(f"  - Relevant: {printed_relevant_titles}")

if __name__ == "__main__":
    main()
