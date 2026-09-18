import json
import os
import re
from urllib.parse import urlparse

from app.chunking.chunker import (
    create_chunks_from_section,
    parse_markdown_sections,
)
from app.cleaner.text_cleaner import clean_markdown
from app.crawler.static_crawler import (
    extract_ordered_content,
    fetch_page,
    parse_html,
)
from app.embeddings.e5_embedder import E5EmbeddingModel
from app.embeddings.embedder import EmbeddingModel
from app.markdown.converter import convert_to_markdown
from app.rag.context_builder import ContextBuilder
from app.retrieval.e5_hybrid_retriever import E5HybridRetriever
from app.retrieval.e5_retriever import E5Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.retrieval.reranker import DeterministicReranker
from app.vectorstore.e5_qdrant_store import E5QdrantStore
from app.vectorstore.qdrant_store import QdrantStore
from qdrant_client.models import PointStruct


def url_to_slug(url: str) -> str:
    """Generate a clean filesystem and collection slug from a URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        slug = parsed.netloc.replace(".", "_")
    else:
        slug = f"{path}"
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug).strip("_")
    return slug.lower() if slug else "website_document"


def process_url(
    url: str,
    chunk_size: int = 800,
    overlap: int = 100,
    markdown_dir: str = "data/markdown",
    chunks_dir: str = "data/chunks",
    qdrant_path: str = "qdrant_storage",
    embedding_model: str = "e5",
    verbose: bool = True,
):
    """
    Run the complete pipeline for any website URL:
    1. Fetch HTML & parse main content
    2. Convert to Markdown and clean text
    3. Structure-aware chunking
    4. Save Markdown and Chunks JSON
    5. Generate embeddings (multilingual-e5-base by default) and index into local Qdrant
    6. Return initialized Hybrid Retriever, Deterministic Reranker & Context Builder
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    slug = url_to_slug(url)
    os.makedirs(markdown_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    markdown_file = os.path.join(markdown_dir, f"{slug}.md")
    chunks_file = os.path.join(chunks_dir, f"{slug}.json")

    # Step 1: Crawl & Parse
    if verbose:
        print("\n[1/5] 🌐 Fetching and parsing webpage...")
    html = fetch_page(url)
    soup = parse_html(html)
    title = soup.title.get_text(strip=True) if soup.title else "Website Content"
    content = extract_ordered_content(soup)
    if verbose:
        print(f"      Title: {title}")
        print(f"      Extracted {len(content)} structured blocks.")

    # Step 2: Markdown Conversion & Cleaning
    if verbose:
        print("[2/5] 📝 Converting and cleaning Markdown...")
    raw_markdown = convert_to_markdown(content, title=title)
    cleaned_markdown = clean_markdown(raw_markdown)
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(cleaned_markdown)
    if verbose:
        print(f"      Saved Markdown to: {markdown_file}")

    # Step 3: Structure-Aware Chunking
    if verbose:
        print("[3/5] ✂️  Performing structure-aware chunking...")
    sections = parse_markdown_sections(cleaned_markdown)
    chunks = []
    for section in sections:
        sec_chunks = create_chunks_from_section(
            section, chunk_size=chunk_size, overlap=overlap
        )
        chunks.extend(sec_chunks)

    for idx, chunk in enumerate(chunks):
        chunk["id"] = idx

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"      Generated {len(chunks)} chunks.")
        print(f"      Saved Chunks to: {chunks_file}")

    if not chunks:
        raise ValueError(f"No structured content or chunks could be extracted from {url}")

    # Step 4: Embeddings & Qdrant Indexing
    texts = [chunk["text"] for chunk in chunks]
    use_e5 = embedding_model.lower() in {"e5", "multilingual-e5-base"}

    if use_e5:
        if verbose:
            print("[4/5] 🧠 Generating multilingual-e5-base embeddings & indexing into Qdrant...")
        embedder = E5EmbeddingModel()
        collection_name = "natural_environment_e5" if slug == "wiki_natural_environment" or slug == "natural_environment" else f"{slug}_e5"
        store = E5QdrantStore(path=qdrant_path, collection_name=collection_name)
        store.create_collection()
        vectors = embedder.encode_documents(texts, show_progress_bar=False)
    else:
        if verbose:
            print("[4/5] 🧠 Generating BGE-M3 embeddings & indexing into Qdrant...")
        embedder = EmbeddingModel()
        collection_name = "natural_environment" if slug == "wiki_natural_environment" or slug == "natural_environment" else slug
        store = QdrantStore(path=qdrant_path)
        store.COLLECTION_NAME = collection_name
        store.create_collection()
        vectors = embedder.encode(texts, show_progress_bar=False)

    points = []
    for chunk, vector in zip(chunks, vectors):
        point = PointStruct(
            id=chunk["id"],
            vector=vector.tolist(),
            payload={
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            },
        )
        points.append(point)

    store.client.upsert(
        collection_name=store.COLLECTION_NAME,
        points=points,
    )
    if verbose:
        print(f"      Indexed {len(points)} vector points into Qdrant collection '{store.COLLECTION_NAME}'.")

    # Step 5: Initialize Hybrid Retriever & Components
    if verbose:
        print("[5/5] ⚡ Initializing Hybrid Retrieval (BM25 + Qdrant RRF) & Reranker...")

    if use_e5:
        semantic_retriever = E5Retriever(
            top_k=20,
            embedder=embedder,
            store=store,
        )
        hybrid_retriever = E5HybridRetriever(
            chunks=chunks,
            top_k=5,
            candidate_k=20,
            rrf_k=60,
            e5_retriever=semantic_retriever,
        )
    else:
        semantic_retriever = QdrantRetriever(
            top_k=20,
            embedder=embedder,
            store=store,
        )
        hybrid_retriever = HybridRetriever(
            chunks=chunks,
            top_k=5,
            candidate_k=20,
            rrf_k=60,
            qdrant=semantic_retriever,
        )

    reranker = DeterministicReranker()
    context_builder = ContextBuilder(max_chunks=3, max_chars=8000)

    if verbose:
        print("      Pipeline setup completed successfully!\n")

    return {
        "url": url,
        "slug": slug,
        "title": title,
        "chunks": chunks,
        "markdown_file": markdown_file,
        "chunks_file": chunks_file,
        "retriever": hybrid_retriever,
        "reranker": reranker,
        "context_builder": context_builder,
        "store": store,
        "embedding_model": "multilingual-e5-base" if use_e5 else "BAAI/bge-m3",
    }
