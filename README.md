# AI Web Chatbot

AI Web Chatbot is a Python-based website knowledge and retrieval system being built from scratch.

The main idea is simple:

**Give the system a website URL → crawl the website → extract useful content → clean it → convert it to Markdown → split it into chunks → retrieve relevant information → answer user questions from the website content.**

The project is being designed as a general-purpose website retrieval system rather than a system tied to one particular website or dataset.

The current retrieval foundation uses **BM25 and deterministic reranking**. The next retrieval phase is **local embeddings + Qdrant + hybrid BM25/semantic retrieval**.

---

## What This Project Does

The project processes website content through several stages:

```text
Website URL
    ↓
Web Crawling
    ↓
HTML Parsing
    ↓
Main Content Extraction
    ↓
Text Cleaning
    ↓
Markdown Conversion
    ↓
Structure-Aware Chunking
    ↓
Retrieval (BM25)
    ↓
Deterministic Reranking
    ↓
Relevant Website Content
    ↓
Answer Layer
```

The system is being developed modularly so each stage operates independently and reliably.

---

## Libraries and What They Do

### Python

Python is the main programming language used for the complete backend and processing pipeline.

---

### Requests

**Purpose:** Static webpage fetching.

Used to send HTTP requests and download HTML pages.

```text
Website
   ↓
Requests
   ↓
HTML
```

---

### BeautifulSoup

**Purpose:** HTML parsing and content extraction.

It is used to parse downloaded HTML and work with elements such as:

- Headings
- Paragraphs
- Lists
- Links
- Main content containers

```text
HTML
 ↓
BeautifulSoup
 ↓
Structured Content
```

---

### Playwright

**Purpose:** Dynamic website crawling.

Some websites do not contain their useful content directly in the initial HTML and require JavaScript execution.

Playwright provides browser-based crawling for those cases.

```text
Dynamic Website
      ↓
   Playwright
      ↓
Rendered HTML
      ↓
Content Extraction
```

---

### Markdown

**Purpose:** Structured document representation.

Extracted website content is converted into Markdown so that headings, sections, paragraphs, and lists are represented in a clean and structured form.

Example:

```markdown
# Website Title

## About

Website information...

## Services

- Service 1
- Service 2
```

Markdown also makes the next chunking stage easier.

---

### Custom Text Cleaner

**Purpose:** Clean extracted website text.

The project has its own cleaning logic for problems such as:

- Extra spaces
- Extra blank lines
- Incorrect spaces around punctuation
- Broken hyphenated words
- Unwanted extraction noise

This keeps the text clean without changing its underlying meaning.

---

### Custom Structure-Aware Chunker

**Purpose:** Split large Markdown documents into smaller retrieval units.

The chunker considers:

- Headings
- Heading hierarchy
- Paragraphs
- Content blocks
- Chunk size
- Overlap
- Sentence boundaries

Chunk configuration:

```text
Chunk size: 800 characters
Overlap:    100 characters
```

Each chunk also keeps metadata such as:

- Title
- Section
- Heading path

---

### rank-bm25

**Purpose:** Keyword/lexical information retrieval.

BM25 is the first retrieval method implemented in the project.

It is useful when the query contains exact words, technical terms, names, or identifiers that appear in the website content.

```text
User Query
    ↓
Tokenization
    ↓
BM25
    ↓
Candidate Chunks (Top 20)
```

The project also includes document metadata in the searchable BM25 representation.

---

### Custom Tokenizer & Normalizer

**Purpose:** Prepare text for BM25 and deterministic scoring.

The tokenizer:

- Converts text to lowercase.
- Extracts normalized alphanumeric and hyphenated tokens.
- Filters standard English stop words (pronouns, prepositions, conjunctions, auxiliary verbs).
- Removes single-character token noise.
- Provides rule-based suffix normalization (singular/plural and verb inflections like `-ies`, `-es`, `-s`, `-ing`, `-ed`).

Example:

```text
"How do human activities affect the environment?"

→

["human", "activity", "affect", "environment"]
```

---

### Custom Deterministic Reranker

**Purpose:** Improve candidate ordering and eliminate lexical frequency bias without using AI models.

The reranker combines multiple deterministic signals:

1. **Query Intent Detection:** Automatically identifies intent (`DEFINITION`, `CAUSE_EFFECT`, `COMPARISON`, `LOCATION`, `QUANTITY`, `GENERAL`).
2. **Subject Entity Extraction:** Separates question meta-intent words (e.g. `define`, `describe`, `refer`, `meaning`) from the core subject entity.
3. **Subject-Definition Matching:** Detects exact subject copular definitions and composition patterns (`is`, `are`, `includes`, `consists of`, `encompasses`, `can be divided into`), while ignoring existential phrases (`there are`) and oblique mentions.
4. **Intent-Specific Semantic Matching:** 
   - `CAUSE_EFFECT`: Rewards action/influence terms and matching impact headers (`Human impact on water`).
   - `COMPARISON`: Rewards contrastive discourse connectors (`on the other hand`, `whereas`, `while`, `unlike`, `contrast`).
   - `LOCATION`: Rewards occurrence and geographic distribution indicators.
5. **Structural & Heading Alignment:** Prioritizes chunks whose immediate section heading matches the target entity.
6. **Lead Sentence Prominence:** Rewards occurrences in opening lead sentences.
7. **Relevance Thresholding:** Rejects completely out-of-scope queries (BM25 score $\le 0.0$) and multi-term queries with insufficient subject coverage (e.g. *"What is artificial intelligence?"* or empty/invalid inputs).

```text
BM25 Candidates (Top 20)
            ↓
  Deterministic Reranker
  ├── Query Intent Routing
  ├── Subject Entity Verification
  ├── Copula Definition & Action Match
  ├── Heading & Lead Prominence
  └── Relevance Thresholding
            ↓
  High-Precision Results (Top 5)
```

---

### Qdrant

**Purpose:** Vector database for semantic retrieval.

Qdrant is the planned vector-search component of the project.

The upcoming embedding pipeline will be:

```text
Website Chunk
     ↓
Embedding Model
     ↓
Vector
     ↓
Qdrant
```

At query time:

```text
User Query
     ↓
Embedding Model
     ↓
Query Vector
     ↓
Qdrant
     ↓
Semantic Results
```

Qdrant is being run locally rather than through a hosted vector database.

---

### BGE-M3

**Purpose:** Planned local embedding model.

BGE-M3 will convert website chunks and user queries into numerical vectors so the system can perform semantic retrieval.

This allows the system to find related content even when the wording of the query and document is different.

For example:

```text
Document:
"The lithosphere is the rigid outer layer of Earth."

Query:
"Which solid layer surrounds Earth's mantle?"
```

Semantic retrieval can identify the relationship even though the wording is different.

---

### FastAPI

**Purpose:** Planned backend API layer.

FastAPI will expose the retrieval/chatbot functionality through HTTP endpoints.

Planned flow:

```text
Frontend
   ↓
FastAPI
   ↓
Retrieval Pipeline
   ↓
Answer
   ↓
FastAPI
   ↓
Frontend
```

---

## Retrieval Architecture

### Current

```text
User Query
    ↓
Text Normalization & Intent Detection
    ↓
BM25 Candidate Retrieval (Top 20)
    ↓
Deterministic Reranker (Top 5)
    ↓
Relevant Chunks
```

### Planned

```text
                         User Query
                               │
                 ┌─────────────┴─────────────┐
                 ↓                           ↓
               BM25                    BGE-M3 Embedding
                 ↓                           ↓
         Keyword Results                   Vector
                                             ↓
                                          Qdrant
                                             ↓
                 └─────────────┬─────────────┘
                               ↓
                      Hybrid Retrieval
                               ↓
                     Deterministic Reranker
                               ↓
                        Relevant Chunks
```

The purpose of hybrid retrieval is to combine:

**BM25**
- Exact keyword matching
- Technical terms
- Names
- Numbers
- Identifiers

**Embeddings**
- Semantic similarity
- Synonyms
- Paraphrased questions
- Different wording
- Meaning-based retrieval

---

## Current Project Progress

### Completed

- Website crawling
- Static HTML fetching
- Main content extraction
- Ordered content extraction
- Text cleaning
- Markdown conversion
- Markdown storage
- Structure-aware chunking
- Chunk metadata
- JSON chunk storage
- Text tokenization & suffix normalization
- Stop-word filtering (with pronouns and auxiliary verbs)
- BM25 candidate retrieval ($K_{\text{candidates}} = 20$)
- Term coverage & exact phrase matching
- Intent detection & subject extraction
- Deterministic reranking with definition & action scoring
- Model-free relevance thresholding (out-of-scope & invalid query handling)

### In Progress / Next

- BGE-M3 integration
- Local embedding generation
- Qdrant vector storage
- Semantic retrieval
- BM25 + embedding hybrid retrieval
- Context building
- Extractive/rule-based answer layer
- FastAPI API
- Frontend
- Deployment

---

## Project Structure

```text
AI-webchatbot/
│
├── app/
│   ├── api/
│   ├── chunking/
│   │   ├── chunker.py
│   │   └── save_chunks.py
│   ├── cleaner/
│   │   └── text_cleaner.py
│   ├── crawler/
│   │   └── static_crawler.py
│   ├── embeddings/
│   ├── markdown/
│   │   ├── converter.py
│   │   └── save_markdown.py
│   ├── rag/
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── reranker.py
│   │   ├── term_coverage.py
│   │   └── text_processor.py
│   └── vectorstore/
│
├── data/
│   ├── chunks/
│   ├── markdown/
│   └── raw/
│
├── models/
│   └── embeddings/
│
├── qdrant/
├── qdrant_storage/
├── requirements.txt
└── README.md
```

---

# About the Developer

## Kushal Girdhar

Kushal Girdhar is a Python/backend and AI-focused developer currently working toward an **MCA in Artificial Intelligence and Data Science**.

### Education

- **BCA** — Maharaja Ganga Singh University, Bikaner
- **MCA in Artificial Intelligence and Data Science** — Vivekananda Global University, Jaipur

### Technical Interests

- Python
- Backend Development
- AI Engineering
- RAG Systems
- Information Retrieval
- FastAPI
- Flask
- Django
- SQL
- PostgreSQL
- MySQL
- Qdrant
- Docker
- Git/GitHub
- React
- Next.js

This project is part of Kushal's practical work in **Python backend development, AI engineering, information retrieval, vector databases, and RAG architecture**.

---

# Repository

**GitHub:** https://github.com/kushalgirdhar/AI-webchatbot

Repository owner: **Kushal Girdhar**

Repository name: **AI-webchatbot**

The repository contains the source code, application structure, data-processing pipeline, and project documentation.

---

## Project Vision

The final goal is to build a website chatbot that can:

```text
Enter Website URL
       ↓
Crawl Website
       ↓
Understand Website Structure
       ↓
Clean & Convert Content
       ↓
Create Retrieval Chunks
       ↓
Build BM25 + Vector Index
       ↓
Ask Questions
       ↓
Retrieve Relevant Website Content
       ↓
Build Context
       ↓
Generate/Extract an Answer
       ↓
Return the Answer with Relevant Sources
```

The system is being built from the fundamentals rather than relying on a ready-made RAG framework, so each major component can be understood, developed, and improved independently.
