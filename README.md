# Building a Production RAG System with LlamaIndex

A deep-dive blog series for software engineers — from a single document on your laptop to 10 million documents in production.

🌐 **Live site:** https://dhanush-ctrl-ai.github.io/llamaindex-rag-blog-series

## The Running Usecase

Your engineering team has a **50,000-page internal wiki**, updated daily, queried by 500 engineers. Every chapter solves one real problem in building that system.

## Chapters

| # | Title | Status |
|---|---|---|
| 01 | The Problem with LLMs and Private Data | ✅ Published |
| 02 | How LlamaIndex Reads Your Documents | ✅ Published |
| 03 | Chunking — Turning Documents into Searchable Pieces | ✅ Published |
| 04 | Embeddings — Giving Text a Location in Meaning-Space | ✅ Published |
| 05 | The Ingestion Pipeline: The Assembly Line | 🔜 Coming soon |
| 06 | How Search Actually Works: Cosine Similarity from Scratch | 🔜 Coming soon |
| 07 | The Query Engine: From Question to Answer | 🔜 Coming soon |
| 08 | Choosing the Right Retriever | 🔜 Coming soon |
| 09 | Scaling Ingestion to 10 Million Documents | 🔜 Coming soon |
| 10 | Observability: Seeing Inside the Black Box | 🔜 Coming soon |
| 11 | Agents: When One Retrieval Isn't Enough | 🔜 Coming soon |

## Structure

```
docs/
├── index.html          ← Landing page (GitHub Pages root)
├── _config.yml         ← Jekyll config
└── chapters/           ← Individual blog posts (Markdown)
    ├── chapter-01-*.md
    ├── chapter-02-*.md
    ├── chapter-03-*.md
    └── chapter-04-*.md
```

## Tech Stack Covered

- 🦙 LlamaIndex Core
- 🤗 HuggingFace Embeddings (BAAI/bge)
- 🔍 ChromaDB / Pinecone
- ⚡ Redis (cache + docstore)
- 🧠 OpenAI GPT-4 / local Ollama
