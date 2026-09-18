import os
import sys

from app.pipeline import process_url


def run_chatbot(url: str = None):
    print("=" * 70)
    print("                      AI Web Chatbot                         ")
    print("=" * 70)

    # 1. Ask user for URL if not provided
    if not url:
        try:
            url = input("\nEnter your URL link: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return

    while not url:
        try:
            url = input("Please enter a valid website URL: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return

    # 2. Automatically run complete pipeline
    try:
        pipeline_data = process_url(url=url, verbose=True)
    except Exception as e:
        print(f"\n❌ Error processing website ({url}): {e}")
        return

    retriever = pipeline_data["retriever"]
    reranker = pipeline_data["reranker"]
    context_builder = pipeline_data["context_builder"]
    title = pipeline_data["title"]

    # 3. Start Chatbot Session
    print("=" * 70)
    print(f"💬 Chatbot Session Started — [{title}]")
    print("=" * 70)
    print("Ask questions about the website content.")
    print("Type 'exit' or 'quit' to stop.\n")

    try:
        while True:
            try:
                query = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if query.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break

            if not query:
                print("Please enter a question.")
                continue

            # Hybrid Retrieval (BM25 + Qdrant RRF)
            candidates = retriever.search(query, top_k=10, rerank=False)

            # Deterministic Reranker
            ranked_results = reranker.rerank(query, candidates, top_k=5)

            # Build Clean Context
            context = context_builder.build(ranked_results)

            if not context:
                print("\nBot: I couldn't find relevant information in the website.")
                continue

            print("\n" + "=" * 70)
            print("RETRIEVED WEBSITE CONTENT")
            print("=" * 70)
            print(context)
            print("=" * 70)

    finally:
        retriever.close()


def main():
    cli_url = sys.argv[1] if len(sys.argv) > 1 else None
    run_chatbot(url=cli_url)


if __name__ == "__main__":
    main()