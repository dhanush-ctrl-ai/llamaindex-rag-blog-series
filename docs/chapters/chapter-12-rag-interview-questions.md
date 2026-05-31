---
layout: default
---

# Chapter 12: RAG Interview Questions — How Top Companies Test You

> **Series:** Building a Production RAG System with LlamaIndex
> **Usecase:** You have built the full RAG system across the previous 11 chapters. Now you need to explain every decision in an interview. This chapter contains the 10 most frequently asked RAG questions across Google, Meta, Amazon, Microsoft, OpenAI, Cohere, Anthropic, and Databricks — with answers in the exact format interviewers reward.

---

## How to use this chapter

Every answer follows the same structure interviewers at FAANG and AI companies reward:

1. **Single user** — the simplest working solution for one person
2. **Scale to millions** — what breaks, and the exact architectural changes that fix it
3. **Pros and cons** — a clear tradeoff table for both approaches

This mirrors how staff engineers think. You start simple, justify every added complexity, and name the tradeoffs explicitly. Answers that jump straight to Kubernetes and Kafka without explaining the simple case fail interviews at every level.

---

## Q1 — Design a RAG system from scratch for a large-scale enterprise knowledge base

**Asked at:** Google · Meta · Amazon · Microsoft · Cohere · Databricks · Scale AI

**Frequency:** ★★★★★ — appears in virtually every ML/SWE loop at companies building RAG products

---

### Single user — the MVP

For one user querying a single corpus, the entire system is five lines of Python:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index  = VectorStoreIndex.from_documents(
    SimpleDirectoryReader("./docs").load_data()
)
response = index.as_query_engine().query("How do I roll back a deployment?")
print(response.response)
```

The pipeline behind those five lines has four stages:

**Ingestion** — `SimpleDirectoryReader` loads every file in `./docs` into `Document` objects. `VectorStoreIndex.from_documents` chunks each document with `SentenceSplitter` (default: 1024 tokens, 200 overlap), calls the embed model on each chunk, and stores the resulting vectors in a `SimpleVectorStore` (an in-memory Python dict).

**Index** — The `SimpleVectorStore` holds all embeddings as a NumPy matrix. It is rebuilt from scratch on every restart.

**Retrieval** — `query()` embeds the question, runs cosine similarity against all stored vectors, returns the top-2 `TextNode` objects with scores.

**Synthesis** — The `RetrieverQueryEngine` packs the two chunks into a prompt template, calls the LLM, and returns a `Response` object with `.response` (the answer) and `.source_nodes` (citations).

This works for a demo. It breaks the moment you have more than ~50k documents, need answers to survive a restart, or have more than one user.

---

### Scale to millions — the production system

At 10 million documents with 500 concurrent users, five things break simultaneously:

```
Single process      → parallel ingestion workers (num_workers=4+)
In-memory store     → Pinecone / Weaviate / pgvector
No caching          → Redis IngestionCache + DocstoreStrategy
O(N) vector scan    → ANN index (HNSW inside the vector DB)
Sync query          → async query engine + load balancer
```

The production architecture:

```python
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion.cache import RedisCache

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    embed_batch_size=64,
)

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50), Settings.embed_model],
    docstore=RedisDocumentStore.from_host_and_port("localhost", 6379),
    vector_store=PineconeVectorStore(pinecone_index=pc.Index("wiki")),
    cache=IngestionCache(cache=RedisCache.from_host_and_port("localhost", 6379)),
    docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
)

nodes = pipeline.run(documents=changed_docs, num_workers=4)
```

Query path: load-balanced HTTPS → async query engine → embed query → Pinecone ANN search (HNSW, sub-10ms) → top-k nodes → `SimilarityPostprocessor(cutoff=0.75)` → COMPACT synthesis → response with `source_nodes`.

---

### Pros and cons

| Concern | Single-user MVP | Production system |
|---|---|---|
| Setup time | 5 minutes | Days to weeks |
| Storage | RAM — lost on restart | Pinecone / Redis — persistent |
| Query latency | 10–100ms (O(N) scan) | < 10ms (HNSW ANN) |
| Ingestion cost | Re-embeds everything every run | Only changed documents |
| Horizontal scale | 1 process, 1 machine | N workers + load balancer |
| Observability | None | CallbackManager + structured logs |
| Operational complexity | Zero | Redis + Pinecone + workers to maintain |
| Best for | Prototypes, demos, < 50k docs | Live products, > 500k docs |

**The interviewer signal:** Start with the 5-line version and say "this breaks at scale for three specific reasons." Then name each reason and its fix. Engineers who jump straight to Pinecone without explaining why rarely get the offer.

---

## Q2 — How do you choose chunk size, and what happens when you get it wrong?

**Asked at:** Amazon · Cohere · Databricks · OpenAI · Hugging Face

**Frequency:** ★★★★★ — the single most probed technical detail in RAG interviews

---

### Single user — picking the right chunk size

Chunk size controls the granularity of what gets embedded and retrieved. The goal is: each chunk should contain exactly one complete thought — no more, no less.

```python
from llama_index.core.node_parser import SentenceSplitter

# The starting point for most prose documents
splitter = SentenceSplitter(
    chunk_size=512,       # tokens — roughly 350 words
    chunk_overlap=50,     # tokens bridging adjacent chunks
)

nodes = splitter.get_nodes_from_documents(documents)

# Inspect what you actually got
for node in nodes[:3]:
    print(f"tokens ≈ {len(node.text.split()) * 1.3:.0f}")
    print(f"text: {node.text[:120]}...")
    print()
```

The two failure modes to explain in every interview:

**Too small (< 100 tokens):** Each chunk is a sentence fragment. The embedding represents a narrow slice with no context. When retrieved, the LLM gets: "The window is 30 days." — no subject, no explanation of what it applies to. Answer quality collapses.

**Too large (> 1500 tokens):** The chunk contains three different topics. The embedding is an average of all of them — a blurry semantic blob. A specific question about topic A has low cosine similarity because the embedding is diluted by topics B and C. Precision collapses.

The sweet spot depends on document type:
- Prose / articles → 256–512 tokens
- Source code → `CodeSplitter`, 40–60 lines
- Dense technical manuals → `HierarchicalNodeParser`, 2048 parent / 512 child
- FAQ / structured data → 128–256 tokens

---

### Scale to millions — adaptive chunking strategies

At scale, a single global chunk size is wrong for every document type in a heterogeneous corpus. The production approach:

```python
from llama_index.core.node_parser import (
    SentenceSplitter, CodeSplitter, HierarchicalNodeParser
)

# Route documents to the right splitter based on metadata
def get_splitter(doc):
    mime = doc.metadata.get("mime_type", "")
    if "python" in mime or doc.metadata.get("file_name","").endswith(".py"):
        return CodeSplitter(language="python", chunk_lines=40)
    elif doc.metadata.get("page_count", 0) > 50:
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
    else:
        return SentenceSplitter(chunk_size=512, chunk_overlap=50)

# At ingestion time:
for doc in documents:
    splitter = get_splitter(doc)
    nodes = splitter.get_nodes_from_documents([doc])
    pipeline.run(nodes=nodes)
```

Also measure retrieval quality by chunk size on your actual data — run 20 representative queries and check whether the answer is in the top-3 retrieved chunks. This takes 30 minutes and avoids weeks of debugging wrong answers.

---

### Pros and cons

| Approach | Single global chunk size | Per-type adaptive chunking |
|---|---|---|
| Setup | 1 parameter | Router logic + per-type config |
| Works well for | Homogeneous corpus (all PDFs, all articles) | Mixed corpus (code + docs + FAQs) |
| Retrieval precision | Good if corpus is uniform | Better across all document types |
| Overlap handling | Fixed overlap everywhere | Tuned per type (code needs no overlap) |
| Maintenance | None | Update router when new document types added |
| Risk | Wrong size for minority doc types | Routing bugs create inconsistent chunking |

**The interviewer signal:** The answer they want is: chunk size is a precision vs. recall tradeoff on the embedding. Small = precise but fragmented. Large = contextual but diluted. Then mention `chunk_overlap` prevents information loss at boundaries. Interviewers at Cohere and Databricks specifically probe whether you have measured this on real data.

---

## Q3 — RAG vs fine-tuning — when do you use each, and can you use both?

**Asked at:** Google · Meta · Anthropic · Mistral · Hugging Face · Together AI

**Frequency:** ★★★★☆ — the classic ML design question, appears in system design and ML-depth rounds

---

### Single user — the mental model

Both approaches ground the model in domain knowledge. They do it in completely different places:

```
Fine-tuning:  knowledge baked INTO model weights
              └─ updated at training time
              └─ frozen after deployment

RAG:          knowledge injected INTO the context window
              └─ updated at query time
              └─ always fresh
```

When to choose RAG:
- Knowledge changes frequently (wiki updated daily, prices change hourly)
- You need to cite sources ("as per deployment-guide.txt, line 42")
- Data is private and must not be baked into a shared model
- You have < 1,000 labelled examples (not enough to fine-tune well)

When to choose fine-tuning:
- You need the model to adopt a specific tone, persona, or output format
- The domain has vocabulary the base model does not know (rare medical terms, internal jargon)
- Low-latency is critical and you cannot afford retrieval RTT
- Task is classification or extraction, not open-ended Q&A

When to use both (RAFT — Retrieval-Augmented Fine-Tuning):
- Fine-tune the model on (question, retrieved_context, answer) triples from your domain
- The model learns both the vocabulary AND how to reason over retrieved chunks
- Best accuracy, highest cost to build and maintain

```python
# RAFT training data format
training_example = {
    "question": "What is the staging rollback procedure?",
    "context": [
        "chunk_A: kubectl rollout undo deployment/app -n staging",
        "chunk_B: Run smoke tests after rollback completes",
        "distractor_C: Production uses Helm, not kubectl",  # RAFT trains model to ignore distractors
    ],
    "answer": "Run: kubectl rollout undo deployment/app -n staging, then verify with smoke tests."
}
```

---

### Scale to millions — which scales better

At 10M documents, fine-tuning does not scale at all — the model cannot hold 10M documents in its weights. RAG scales naturally because you just add more nodes to the vector store. The model stays the same; the knowledge base grows independently.

Fine-tuning is frozen the moment training ends. A document updated tomorrow does not affect a fine-tuned model. RAG reflects document updates after the next ingestion run.

---

### Pros and cons

| Dimension | RAG | Fine-tuning | RAG + Fine-tuning (RAFT) |
|---|---|---|---|
| Knowledge freshness | Always current | Frozen at training | Frozen tone, fresh knowledge |
| Citation / attribution | Built-in via source_nodes | None | Partial |
| Data privacy | Docs stay external | Baked into weights (risky) | Depends on deployment |
| Cost to update | Re-ingest changed docs | Full retrain (~$100–$10k+) | Retrain + re-ingest |
| Handles new vocabulary | Only if in retrieved chunk | Yes — learned in weights | Best of both |
| Scales to 10M docs | Yes — just a bigger vector DB | No — weights are fixed size | Yes (RAG component scales) |
| Best accuracy | Good | Good for narrow tasks | Best overall |
| Time to production | Days | Weeks | Months |

**The interviewer signal:** The answer interviewers want: "Use RAG for knowledge, fine-tuning for behaviour. If you need both, RAFT trains the model to reason over retrieved context rather than memorise facts." Anyone who says "fine-tuning is better than RAG" without qualification fails the tradeoff dimension.

---

## Q4 — Your RAG system is returning wrong answers confidently. How do you debug it?

**Asked at:** Amazon · Microsoft · Cohere · Scale AI · Glean · Perplexity

**Frequency:** ★★★★☆ — the production debugging question, especially at senior levels

---

### Single user — the three-layer debugging framework

Wrong answers in RAG always come from one of three layers. Inspect them in this order:

**Layer 1 — Retrieval failure.** Check `response.source_nodes`. Are the retrieved chunks about the right topic?

```python
response = engine.query("How do I roll back a deployment?")

print(f"Answer: {response.response}\n")
print("=== Retrieved chunks ===")
for i, node in enumerate(response.source_nodes):
    print(f"  [{i+1}] score={node.score:.4f}")
    print(f"       {node.node.text[:150]}...")
```

If scores are below 0.6 or chunks are about the wrong topic: the retriever is failing. Fix: re-examine chunk size, embed model, or add hybrid BM25 search.

**Layer 2 — Synthesis failure.** If the right chunks were retrieved but the answer is wrong, the LLM is hallucinating or ignoring context. Inspect the full prompt:

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
debug = LlamaDebugHandler(print_trace_on_end=False)
Settings.callback_manager = CallbackManager([debug])

response = engine.query("How do I roll back a deployment?")

llm_calls = debug.get_llm_inputs_outputs()
print("=== Full prompt sent to LLM ===")
print(llm_calls[0].inputs["messages"])
```

If the context is there but the LLM answered from training data: switch to `response_mode="refine"` or increase prompt instruction strength ("Answer ONLY from the provided context. Do not use prior knowledge.").

**Layer 3 — Data failure.** Right retrieval, right synthesis, still wrong: the source document is outdated, incomplete, or never ingested.

```python
# Check if the document exists in the docstore
doc_id = "deployment-guide.txt"
doc = docstore.get_document(doc_id)
print(f"Ingested: {doc is not None}")
print(f"Last modified in metadata: {doc.metadata.get('last_modified')}")
```

---

### Scale to millions — systematic observability

At scale, you cannot manually inspect every wrong answer. You need automated detection:

```python
# Log retrieval quality on every query
class RetrievalAuditHandler(BaseCallbackHandler):
    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        if event_type == CBEventType.RETRIEVE and payload:
            nodes = payload.get("nodes", [])
            metrics = {
                "query": payload.get("query_str",""),
                "num_nodes": len(nodes),
                "top_score": nodes[0].score if nodes else 0,
                "min_score": nodes[-1].score if nodes else 0,
                "low_confidence": nodes[0].score < 0.65 if nodes else True,
            }
            logger.info(json.dumps(metrics))  # → Datadog / CloudWatch
```

Alert when `top_score < 0.65` — this is the single metric most predictive of a wrong answer. Weekly: sample 50 queries manually and check source_nodes. Build an eval set from engineer thumbs-down feedback and run RAGAS faithfulness weekly.

---

### Pros and cons

| Debugging approach | Single-query manual inspection | Automated monitoring at scale |
|---|---|---|
| Setup | None — use LlamaDebugHandler | Requires structured logging infra |
| Coverage | One query at a time | Every query, 24/7 |
| Detail | Full prompt visible | Aggregated metrics |
| Time to find a bug | Minutes (for that query) | Hours (pattern emerges across queries) |
| Cost | Free | Infra + logging costs |
| Catches silent regressions | No | Yes — alerts on score drops |
| Best for | Development + initial debugging | Production monitoring |

**The interviewer signal:** The answer they want: "I start by checking source_nodes — if the scores are low, retrieval is the problem. If scores are high but the answer is wrong, it's a synthesis problem. If both are fine but the answer is still wrong, the document might not be in the corpus." This layered approach shows systematic thinking, not guessing.

---

## Q5 — How do you handle hybrid search — combining dense vector search with BM25?

**Asked at:** Elastic · OpenAI · Weaviate · Databricks · Glean · Perplexity

**Frequency:** ★★★☆☆ — common at companies building search infrastructure

---

### Single user — why you need both

Vector search misses exact terms. BM25 misses semantic intent. One query shows the problem immediately:

```
Query: "kubectl rollout undo v1.24.2"

Vector search result:  "To revert a deployment, use the rollback command..." (score: 0.71)
BM25 result:           "kubectl rollout undo v1.24.2 --record=true"       (score: 0.94)
```

The vector search found semantically related content but missed the exact version string. BM25 found the exact string. Neither alone is sufficient for a mixed corpus.

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever   = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

hybrid = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,             # disable query rewriting for simplest form
    mode="reciprocal_rerank",  # Reciprocal Rank Fusion (RRF)
)

nodes = hybrid.retrieve("kubectl rollout undo v1.24.2")
```

**Reciprocal Rank Fusion (RRF):** Each retriever returns a ranked list. A node's fused score is `sum(1 / (rank + 60))` across all lists. Nodes that appear near the top in multiple lists score highest. The constant 60 prevents top-ranked items from dominating too strongly.

---

### Scale to millions — production hybrid search

At scale, running two separate retrieval systems in serial adds latency. The production approach is async parallel:

```python
hybrid = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=4,        # LLM generates 4 query variants — improves recall
    mode="reciprocal_rerank",
    use_async=True,       # retrieves from both in parallel
)

# Both retrievers run concurrently; total latency = max(vector_latency, bm25_latency)
# instead of vector_latency + bm25_latency
nodes = await hybrid.aretrieve("kubectl rollout undo v1.24.2")
```

At 10M documents, keep BM25 in Elasticsearch (built for text search at scale) and dense vectors in Pinecone. Merge results at the application layer using RRF before sending to the synthesizer.

---

### Pros and cons

| Retriever | Vector search only | BM25 only | Hybrid (RRF) |
|---|---|---|---|
| Semantic queries | Excellent | Poor | Excellent |
| Exact term queries | Poor | Excellent | Excellent |
| Version numbers / codes | Poor | Excellent | Excellent |
| Implementation complexity | Low | Low | Medium |
| Latency | Low | Very low | Low (with async) |
| Infrastructure | Vector DB | Elasticsearch / Whoosh | Both |
| Recall on mixed queries | ~70% | ~70% | ~90%+ |

**The interviewer signal:** Name RRF specifically. Explain why 60 is used as the constant. Then explain that `num_queries=4` in `QueryFusionRetriever` generates query variants via the LLM to increase recall further — this shows depth beyond the basic answer.

---

## Q6 — How do you evaluate a RAG pipeline? What metrics do you measure?

**Asked at:** Google · Meta · Cohere · Hugging Face · Patronus AI · Arize

**Frequency:** ★★★☆☆ — appears in ML-depth rounds, especially at companies with eval infrastructure

---

### Single user — the four RAGAS metrics

RAGAS (Retrieval-Augmented Generation Assessment) separates retrieval quality from generation quality. Evaluate each independently:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Is the answer grounded in the retrieved context?
    answer_relevancy,    # Is the answer relevant to the question?
    context_precision,   # Are the retrieved chunks actually relevant?
    context_recall,      # Were all relevant chunks retrieved?
)
from datasets import Dataset

eval_dataset = Dataset.from_list([
    {
        "question":  "How do I roll back a deployment?",
        "answer":    response.response,
        "contexts":  [n.node.text for n in response.source_nodes],
        "ground_truth": "Run kubectl rollout undo deployment/app -n staging"
    },
    # ... more examples
])

result = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy,
                                          context_precision, context_recall])
print(result)
# {'faithfulness': 0.89, 'answer_relevancy': 0.91,
#  'context_precision': 0.76, 'context_recall': 0.83}
```

What each metric catches:

| Metric | What it measures | Low score means |
|---|---|---|
| Faithfulness | Answer claims are in context | LLM is hallucinating |
| Answer relevancy | Answer addresses the question | Synthesis is off-topic |
| Context precision | Retrieved chunks are relevant | Retriever has too much noise |
| Context recall | Right chunks were retrieved | Retriever is missing relevant docs |

---

### Scale to millions — continuous evaluation

For a live system with 500 queries/day, you cannot manually evaluate everything. The production approach:

```python
# Sample 5% of queries for automated evaluation
import random

if random.random() < 0.05:
    # Async evaluation — doesn't block the user response
    asyncio.create_task(evaluate_query(
        question=query_str,
        answer=response.response,
        contexts=[n.node.text for n in response.source_nodes],
    ))

# Weekly: run full RAGAS eval on 200-question labelled set
# Alert if faithfulness drops below 0.80 (indicates LLM starting to hallucinate)
# Alert if context_recall drops below 0.75 (indicates retrieval degradation)
```

Build the labelled question set from engineer feedback (thumbs-down → label as bad example, thumbs-up → label as good). After 2 weeks, you have 50–100 labelled examples that reflect real usage.

---

### Pros and cons

| Evaluation approach | Manual spot-check | RAGAS automated eval |
|---|---|---|
| Setup | None | Labelled eval set + RAGAS install |
| Coverage | Sparse — what you remember to check | Systematic — same 200 questions every week |
| Measures retrieval separately | Only if you check source_nodes | Yes — context_precision and context_recall |
| Detects regressions | No — no baseline | Yes — compare to last week's scores |
| Cost | Engineer time | LLM-as-judge costs (~$0.01/query) |
| Accuracy | Subjective | Measurable, reproducible |
| Best for | Initial development | Production monitoring + regression detection |

**The interviewer signal:** Name all four RAGAS metrics and what failure each catches. The insight interviewers look for: "faithfulness low means the LLM is ignoring the context; context_recall low means the retriever missed relevant chunks — these require completely different fixes." Conflating them is the most common wrong answer.

---

## Q7 — How do you keep your RAG knowledge base fresh when documents update daily?

**Asked at:** Amazon · LinkedIn · Stripe · Notion · Atlassian · Salesforce

**Frequency:** ★★★☆☆ — appears in system design rounds at companies with large document corpora

---

### Single user — content hashing for deduplication

The naive solution — delete everything and re-ingest daily — costs the same whether 1 doc changed or 10,000 did. The correct solution uses content hashing:

```python
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(), embed_model],
    docstore=SimpleDocumentStore(),
    docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
)

# Daily run — only processes documents whose content hash changed
changed_docs = fetch_documents_modified_since_yesterday()
nodes = pipeline.run(documents=changed_docs)
print(f"Re-embedded {len(nodes)} nodes from {len(changed_docs)} changed documents")
# On a stable day: "Re-embedded 0 nodes from 0 changed documents"
```

`UPSERTS_AND_DELETE` does three things: if a document is new → ingest. If changed → re-ingest and delete old nodes from the vector store. If unchanged → skip. Without the delete step, your vector store accumulates stale embeddings from old document versions indefinitely.

---

### Scale to millions — incremental ingestion with change detection

At 10M documents, even fetching all document hashes to check for changes is expensive. The production approach layers multiple caches:

```python
# 1. Source-level change detection — cheapest possible check
def fetch_changed_documents():
    last_run = redis.get("last_ingestion_timestamp")
    # Only pull docs from S3/Confluence/Notion modified after last run
    return source_api.list_modified_since(last_run)

# 2. Hash check — skip unchanged docs before any embedding
pipeline = IngestionPipeline(
    docstore=RedisDocumentStore.from_host_and_port("localhost", 6379),
    docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    cache=IngestionCache(cache=RedisCache(...)),  # skip re-embedding unchanged chunks
)

# 3. Run daily via cron — only touches changed documents
changed = fetch_changed_documents()  # e.g., 500 out of 10M
nodes = pipeline.run(documents=changed, num_workers=4)
redis.set("last_ingestion_timestamp", datetime.now().isoformat())
```

A 10M document corpus updated daily with 0.1% change rate (10,000 docs/day) costs ~$2/day in embed API calls. Without deduplication, the same corpus costs ~$2,000/day.

---

### Pros and cons

| Approach | Full re-ingest daily | Incremental with hash deduplication |
|---|---|---|
| Implementation | Simple — delete all, re-ingest | DocStore + change detection logic |
| API cost | Re-embeds everything every day | Only changed chunks |
| Correctness | Always fresh | Depends on change detection reliability |
| Stale node cleanup | Automatic (delete all) | Requires UPSERTS_AND_DELETE strategy |
| Failure recovery | Re-run from scratch | Idempotent — safe to retry |
| Scales to 10M docs | 27-hour run, unusable | 30-second run |
| Risk | Cost and time explosion | Change detection bug = missed updates |

**The interviewer signal:** Interviewers want `UPSERTS_AND_DELETE` named specifically, with the reason: without delete, old node versions accumulate and pollute retrieval. The cost example ($2 vs $2,000/day) lands well in system design rounds.

---

## Q8 — How does re-ranking improve retrieval quality, and when is the added latency worth it?

**Asked at:** Cohere · Microsoft · Perplexity · Glean · Vectara · Jina AI

**Frequency:** ★★★☆☆ — common at companies whose product is search or RAG-as-a-service

---

### Single user — bi-encoder vs cross-encoder

The retriever uses a bi-encoder: it embeds the query and each chunk independently, then computes cosine similarity. This is fast (one embed per query, one per chunk at index time) but imprecise — the query and chunk never "see" each other during scoring.

A cross-encoder re-ranks by processing (query, chunk) jointly. It sees the full interaction between query tokens and chunk tokens, giving much more accurate relevance scores:

```python
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank

# Option A: Cross-encoder reranker (fast, no LLM needed)
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,   # rerank 10, keep best 3
)

# Option B: LLM reranker (most accurate, slowest)
reranker = LLMRerank(top_n=3, choice_batch_size=5)

engine = index.as_query_engine(
    similarity_top_k=10,         # cast a wide net
    node_postprocessors=[reranker],  # then rerank to top 3
)
```

The two-stage pattern is: retrieve broadly with fast bi-encoder (top-20), rerank precisely with cross-encoder (top-3). This gives near-cross-encoder accuracy at near-bi-encoder speed.

---

### Scale to millions — when the latency is worth it

Reranking adds 100–400ms latency depending on the model. This is worth it when:

- The query is ambiguous (short queries like "rollback" return noisy results)
- The corpus is large (> 1M chunks, signal-to-noise in retrieval drops)
- The stakes are high (medical, legal, financial — wrong answer has real consequences)

It is not worth it when:
- Queries are specific and retrieval is already high-precision
- Latency budget is tight (< 200ms SLA)
- You can improve precision by tuning chunk size and embed model instead

```python
# Cost-based routing: only rerank when retrieval confidence is low
def get_engine(query_complexity: str):
    if query_complexity == "simple":
        return fast_engine   # no reranker, similarity_top_k=5
    else:
        return quality_engine  # LLMRerank top_n=3, similarity_top_k=15
```

---

### Pros and cons

| Approach | Bi-encoder only | Bi-encoder + cross-encoder rerank | Bi-encoder + LLM rerank |
|---|---|---|---|
| Latency | < 10ms | +100–200ms | +300–600ms |
| Retrieval accuracy | Baseline | +10–15% | +20–25% |
| Cost | Embed model only | Reranker model inference | LLM API cost per query |
| Infrastructure | Vector DB | Vector DB + reranker model | Vector DB + LLM call |
| Best for | Speed-sensitive, precise queries | Most production systems | High-stakes, complex queries |
| Scales to 10M | Yes | Yes | Yes (but expensive at volume) |

**The interviewer signal:** The key insight: "I retrieve top-20 with the bi-encoder because I optimise for recall at that stage — I want the right chunk to be somewhere in the candidate set. Then I rerank to top-3 for precision. These are two different objectives and they need two different models." This bi-objective framing is the answer that gets written down by interviewers.

---

## Q9 — How do you scale a RAG ingestion pipeline to 10 million documents?

**Asked at:** Google · Amazon · Databricks · Snowflake · Palantir · Anyscale

**Frequency:** ★★☆☆☆ — appears in senior/staff design rounds at data-heavy companies

---

### Single user — single process baseline

The baseline pipeline for one developer:

```python
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)
nodes = pipeline.run(documents=documents)  # single process, sequential
```

For 1,000 documents: works fine in minutes. For 10 million: fails in five specific ways.

---

### Scale to millions — the five bottlenecks and their fixes

Each bottleneck has one targeted fix. Apply them in order — the first three give the biggest wins:

```
Bottleneck 1: single process
Fix: num_workers=4 — splits document list across CPU processes via multiprocessing.Pool

Bottleneck 2: re-embedding unchanged documents
Fix: IngestionCache + UPSERTS_AND_DELETE — skip unchanged chunks entirely

Bottleneck 3: everything in RAM
Fix: RedisDocumentStore + PineconeVectorStore — persist across restarts

Bottleneck 4: embed API rate limits
Fix: local BAAI/bge model, embed_batch_size=64 — no rate limit, no per-token cost

Bottleneck 5: one machine
Fix: distributed worker pool (Celery/SQS) — horizontal scale
```

The fully production pipeline:

```python
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    embed_batch_size=64,
)
pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512), Settings.embed_model],
    docstore=RedisDocumentStore.from_host_and_port("localhost", 6379),
    vector_store=PineconeVectorStore(pinecone_index=pc.Index("wiki")),
    cache=IngestionCache(cache=RedisCache.from_host_and_port("localhost", 6379)),
    docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
)
nodes = pipeline.run(documents=changed_docs, num_workers=4)
```

At 10M documents with 0.1% daily change rate:
- Without fixes: 27-hour daily run, ~$2,000/day in embed costs
- With all fixes: 30-second daily run, ~$2/day

---

### Pros and cons

| Concern | Single-process baseline | Full production pipeline |
|---|---|---|
| Setup complexity | 5 lines | 30+ lines + Redis + Pinecone |
| Daily ingestion time (10M docs, 1% change) | 27 hours | ~5 minutes |
| Cost per daily run | $2,000 | $2 |
| Survives restart | No — RAM only | Yes — Redis + Pinecone |
| Horizontal scale | 1 machine max | N workers via SQS/Celery |
| Failure recovery | Restart from scratch | Idempotent — safe to re-run |
| Operational overhead | None | Redis + Pinecone + workers to maintain |

**The interviewer signal:** Name all five bottlenecks and state the fix for each in one sentence. Interviewers at Databricks and Snowflake specifically ask for the cost comparison — having a rough number ($2 vs $2,000/day) demonstrates that you think about production systems in economic terms, not just technical ones.

---

## Q10 — When would you use an agent instead of a standard RAG query engine?

**Asked at:** OpenAI · Anthropic · Google DeepMind · Adept · Cognition · Cohere

**Frequency:** ★★☆☆☆ — increasingly common as agentic AI becomes standard, especially at frontier AI companies

---

### Single user — the decision boundary

A query engine handles one thing: retrieve context → synthesize answer. It is stateless, single-step, and cheap. An agent runs a reasoning loop — it can call tools multiple times, make conditional decisions, and accumulate state across steps.

The boundary is the question itself:

```
"What is the staging rollback procedure?"
→ Single retrieval, single synthesis. Use query engine.

"Compare the staging and production rollback procedures and tell me if they would
 behave differently under a network partition."
→ Step 1: retrieve staging procedure
→ Step 2: retrieve production procedure
→ Step 3: retrieve network partition failure modes
→ Step 4: reason across all three
→ Multi-step, conditional. Use agent.
```

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

agent = ReActAgent.from_tools(
    tools=[
        QueryEngineTool.from_defaults(
            query_engine=staging_engine,
            name="staging_docs",
            description="Search staging environment documentation."
        ),
        QueryEngineTool.from_defaults(
            query_engine=production_engine,
            name="production_docs",
            description="Search production environment documentation."
        ),
    ],
    llm=Settings.llm,
    max_iterations=8,
)

response = agent.chat(
    "Compare staging vs production rollback. Any differences that could cause "
    "staging tests to pass but production rollback to fail?"
)
```

---

### Scale to millions — gating agent use for cost control

An agent averages 3–5 LLM calls per query. At 500 queries/day, uncontrolled agent use costs 3–5× more than a query engine. The production approach: route by query complexity.

```python
import re

AGENT_SIGNALS = ["compare", "difference between", "find all", "given x then y",
                 "what if", "across", "both", "versus", "vs"]

def route_query(query: str) -> str:
    q = query.lower()
    if any(signal in q for signal in AGENT_SIGNALS):
        return "agent"
    if len(query.split()) > 20:   # long queries often need multi-step reasoning
        return "agent"
    return "query_engine"

# Fast path: 80% of queries
if route_query(user_query) == "query_engine":
    response = await engine.aquery(user_query)
# Quality path: 20% of queries
else:
    response = await agent.achat(user_query)
```

This keeps the median latency at ~1s (query engine) while handling complex questions at ~5s (agent), without applying agent overhead universally.

---

### Pros and cons

| Dimension | Query engine | ReAct agent |
|---|---|---|
| LLM calls per query | 1–2 | 3–10 |
| Latency | 1–2 seconds | 5–30 seconds |
| Cost per query | ~$0.001 | ~$0.005–$0.02 |
| Handles multi-step questions | No | Yes |
| Handles external tool calls | No | Yes (FunctionTool) |
| Handles multi-turn conversation | No | Yes (memory buffer) |
| Predictability | High — deterministic flow | Lower — LLM chooses steps |
| Debugging | Check source_nodes | Check each step's reasoning |
| Best for | Single-hop Q&A, definitions, summaries | Comparisons, analysis, conditional reasoning |

**The interviewer signal:** The answer interviewers want: "I gate on query complexity — simple queries go to the query engine, complex ones to the agent. The signal is words like 'compare', 'find all', 'given X then Y'. Unconstrained agent use costs 5× more with no benefit on simple queries." This shows you think about production economics, not just capability.

---

## Quick reference

| # | Question | Companies | Key answer terms |
|---|---|---|---|
| 1 | Design a RAG system end-to-end | Google, Meta, Amazon, Microsoft | Loader → chunker → embedder → vector store → retriever → synthesizer |
| 2 | How do you choose chunk size? | Amazon, Cohere, Databricks | Precision vs recall tradeoff, chunk_overlap, measure on real data |
| 3 | RAG vs fine-tuning | Google, Meta, Anthropic | Knowledge vs behaviour, RAFT for both |
| 4 | Debug wrong answers | Amazon, Microsoft, Cohere | Layer 1: retrieval. Layer 2: synthesis. Layer 3: data |
| 5 | Hybrid search | Elastic, OpenAI, Weaviate | BM25 + vector, Reciprocal Rank Fusion, async parallel |
| 6 | Evaluate a RAG pipeline | Google, Cohere, Hugging Face | RAGAS: faithfulness, context precision, context recall, answer relevancy |
| 7 | Keep knowledge base fresh | Amazon, LinkedIn, Stripe | Content hashing, UPSERTS_AND_DELETE, incremental ingestion |
| 8 | Re-ranking | Cohere, Microsoft, Perplexity | Bi-encoder for recall, cross-encoder for precision, top-20 → top-3 |
| 9 | Scale ingestion to 10M docs | Google, Amazon, Databricks | 5 bottlenecks: parallelism, caching, storage, rate limits, distribution |
| 10 | Agent vs query engine | OpenAI, Anthropic, DeepMind | Multi-step = agent, gate by complexity, 3–5× cost difference |
