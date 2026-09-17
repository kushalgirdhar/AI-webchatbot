# AI-WebChatbot — Retrieval Issues for Antigravity

## Purpose
Confirmed and identified retrieval issues from the tests. Investigate and fix them without changing the project's core constraints.

## Constraints
- General-purpose retrieval system
- No AI/embedding model
- No external AI API or API key
- No Ollama
- Python libraries only
- BM25 + deterministic/model-free retrieval
- Qdrant may be used for storage/retrieval, but do not introduce dense embeddings or AI models
- No Wikipedia-specific rules
- No query-specific hardcoded fixes

## Confirmed Issues

### 1. Definition detector creates false positives
**Query:** `How is the natural environment defined?`

**Expected:** chunk 0

**Actual:** 1) chunk 9, 2) chunk 49, 3) chunk 0

Chunk 9 contains `divisions are defined`; chunk 49 contains `Wilderness is generally defined...`. These receive definition boosts even though they do not define the requested subject.

**Required investigation:** Make definition detection verify that the requested subject is actually being defined. Do not use document-specific rules.

### 2. Definition score is too influential for non-definition questions
**Query:** `How do humans affect water?`

**Expected:** chunks 14, 15

**Actual:** 1) chunk 7, 2) chunk 14, 3) chunk 12, 4) chunk 13, 5) chunk 8

Chunk 7 says `Most water is found in various kinds of natural body of water.` and receives `Definition = 2.5`, although the query asks about human impact.

**Required investigation:** Determine how definition scoring should interact with query intent. Test definition, comparison, cause/effect, quantity, location, and process queries. Do not simply tune weights for this one query.

### 3. No-answer / relevance threshold is missing
Unknown queries such as:
- `What is quantum computing?`
- `What is blockchain?`
- `What is machine learning?`
- `Who invented the telephone?`

return chunks even when BM25 score is `0.0000`.

**Problem:** There is no mechanism for recognizing that the indexed content does not contain the requested information.

**Required investigation:** Add a general model-free relevance/no-answer mechanism that distinguishes relevant, weak/irrelevant, and no-match cases.

### 4. Lexical false positives for out-of-scope queries
**Query:** `What is artificial intelligence?`

Chunks 2 and 0 rank highly because the document contains `artificial`, although it does not explain artificial intelligence.

**Required investigation:** Prevent isolated/common lexical overlap from making unrelated chunks appear relevant. No embeddings or AI models.

### 5. Empty/invalid queries are not handled
Tested:
- `""`
- `" "`
- `"?"`
- `"."`
- `"the"`
- `"what is"`
- `"and or the"`

All return arbitrary chunks with score `0.0000`.

**Required investigation:** Validate queries before retrieval and safely handle empty, punctuation-only, and stop-word-only queries. Do not return arbitrary zero-score chunks.

## Evaluation / Generalization Concerns

### 6. Generalization Recall@1 is not perfect
15-query generalization results:

BM25:
- Recall@1 = 0.5333
- Recall@3 = 0.8000
- Recall@5 = 0.9333
- MRR = 0.6844

BM25 + reranker:
- Recall@1 = 0.8000
- Recall@3 = 1.0000
- Recall@5 = 1.0000
- MRR = 0.8889

Three queries did not place the expected chunk at rank 1.

**Required investigation:** Verify whether each is a genuine ranking failure or whether another returned chunk is also a valid answer source.

### 7. Climate/weather evaluation may be too restrictive
**Query:** `How is climate different from weather?`

Test expects `[32, 33]`, but chunk 34 ranks #1 and contains a direct climate/weather comparison.

**Required investigation:** Review the expected relevant chunk IDs. Do not change retrieval merely to force a particular chunk above another if the higher-ranked chunk is genuinely relevant.

## Tests Already Passed

### Unit tests
12/12 passed.

### Original 20-query evaluation
BM25:
- Recall@1 = 0.7500
- Recall@3 = 0.9500
- Recall@5 = 0.9500
- MRR = 0.8533

BM25 + reranker:
- Recall@1 = 1.0000
- Recall@3 = 1.0000
- Recall@5 = 1.0000
- MRR = 1.0000

### Candidate recall
Generalization dataset:
- Recall@5 = 0.9333
- Recall@10 = 1.0000
- Recall@20 = 1.0000
- Recall@30 = 1.0000
- Recall@50 = 1.0000
- Recall@61 = 1.0000

No candidate-pool bottleneck was found in this test.

## Instructions Before Changing Code

1. Reproduce every reported failure.
2. Identify root causes before implementing fixes.
3. Separate algorithm problems from evaluation-data problems.
4. Use ablation tests where useful.
5. Avoid random weight tuning.
6. Avoid hardcoded query-specific fixes.
7. Avoid Wikipedia-specific logic.
8. Preserve general-purpose behavior.
9. Do not introduce embeddings, AI models, external APIs, API keys, or Ollama.
10. Run all existing unit tests after changes.
11. Run both the original 20-query and 15-query generalization evaluations.
12. Report before/after metrics.
13. Only claim improvement when evaluation demonstrates it.
14. Verify that fixes do not regress other query types.

## Desired Outcome

A robust model-free retrieval pipeline supporting:
- definition questions
- comparison questions
- cause/effect questions
- quantity questions
- location questions
- process questions
- differently worded queries
- unknown/out-of-scope questions
- empty/invalid input
