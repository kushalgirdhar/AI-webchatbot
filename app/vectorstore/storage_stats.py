import json
import os
from pathlib import Path


def get_dir_size(path, physical=True):
    """Calculate size of directory in bytes (physical on-disk blocks or virtual st_size)."""
    total = 0
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    st = os.stat(fp)
                    if physical:
                        total += st.st_blocks * 512
                    else:
                        total += st.st_size
                except OSError:
                    pass
    return total


def format_bytes(size_bytes):
    """Format bytes into readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def inspect_storage(qdrant_path="qdrant_storage"):
    print("=" * 70)
    print("                TOTAL STORAGE BREAKDOWN & STATISTICS                ")
    print("=" * 70)

    # 1. Overall Summary
    qdrant_physical = get_dir_size(qdrant_path, physical=True)
    qdrant_virtual = get_dir_size(qdrant_path, physical=False)
    md_size = get_dir_size("data/markdown", physical=True)
    chunks_size = get_dir_size("data/chunks", physical=True)
    total_physical = qdrant_physical + md_size + chunks_size

    print("\n📦 Actual Physical Disk Usage (SSD Footprint):")
    print(f"  • Qdrant Vector Storage   : {format_bytes(qdrant_physical):<10} (virtual sparse pre-allocated: {format_bytes(qdrant_virtual)})")
    print(f"  • Raw Markdown Files      : {format_bytes(md_size):<10} (data/markdown)")
    print(f"  • Chunk JSON Files        : {format_bytes(chunks_size):<10} (data/chunks)")
    print(f"  • Total Actual Storage    : {format_bytes(total_physical)}")

    # 2. Per-Webpage Breakdown
    print("\n🌐 Processed Webpages on Disk:")
    chunks_dir = "data/chunks"
    md_dir = "data/markdown"

    if os.path.exists(chunks_dir):
        chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".json")])
        if chunk_files:
            print(f"  {'Webpage / Slug':<32} | {'Chunks':<8} | {'Markdown Size':<14} | {'Chunks JSON Size':<16}")
            print("  " + "-" * 78)
            for c_file in chunk_files:
                slug = c_file[:-5]
                c_path = os.path.join(chunks_dir, c_file)
                m_path = os.path.join(md_dir, f"{slug}.md")

                # Count chunks
                chunk_count = 0
                try:
                    with open(c_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chunk_count = len(data) if isinstance(data, list) else 0
                except Exception:
                    pass

                c_sz = os.path.getsize(c_path) if os.path.exists(c_path) else 0
                m_sz = os.path.getsize(m_path) if os.path.exists(m_path) else 0

                print(f"  {slug:<32} | {chunk_count:<8} | {format_bytes(m_sz):<14} | {format_bytes(c_sz):<16}")
        else:
            print("  No processed webpages found in data/chunks.")

    # 3. Qdrant Collections
    print("\n🔍 Qdrant Collections in Database:")
    meta_file = os.path.join(qdrant_path, "meta.json")
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            collections = meta.get("collections", {})
            for name, config in collections.items():
                vector_config = config.get("vectors", {})
                size = vector_config.get("size", "N/A")
                distance = vector_config.get("distance", "N/A")
                col_dir_1 = os.path.join(qdrant_path, "collections", name)
                col_dir_2 = os.path.join(qdrant_path, "collection", name)
                col_size = get_dir_size(col_dir_1, physical=True) + get_dir_size(col_dir_2, physical=True)
                print(f"  • Collection: '{name}'")
                print(f"    - Vector Dimension : {size}")
                print(f"    - Distance Metric  : {distance}")
                print(f"    - Collection Size  : {format_bytes(col_size)}")
        except Exception as e:
            print(f"  Could not read meta.json: {e}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    inspect_storage()
