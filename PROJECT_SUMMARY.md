# Cascading Context — Project Summary

## Overview

Cascading Context is a Python system that gives AI agents persistent memory across conversation sessions. It extracts structured facts from conversations, stores them with vector embeddings in a local DuckDB database, and retrieves the most relevant facts to inject into future session prompts. The result: an AI agent that remembers what it learned in prior conversations and applies that knowledge contextually.

## Problem Statement

Large language models are stateless — each conversation starts from scratch. When a user discusses a project over multiple sessions, the AI has no recollection of prior context. This leads to repetitive explanations, lost nuance, and inability to build on prior work.

Cascading Context solves this by creating an automated knowledge pipeline: extract facts from conversations, embed them semantically, and inject the most relevant ones into new sessions.

## Architecture

The system is a five-stage pipeline:

```
Session → Summarizer → Fact Store → Retriever → Context Injector → Agent Prompt
```

| Stage | Module | Role |
|-------|--------|------|
| **Session Manager** | `ctx/session.py` | Tracks conversations with messages and metadata |
| **Summarizer** | `ctx/summarizer.py` | Sends transcripts to Claude, extracts structured facts |
| **Fact Store** | `ctx/facts.py` + `ctx/db.py` | Persists facts with embeddings in DuckDB |
| **Retriever** | `ctx/retriever.py` | Cosine similarity search with temporal decay |
| **Context Injector** | `ctx/injector.py` | Formats and injects facts into system prompts |

### Data Flow

1. A conversation happens within a **session** (messages stored in DuckDB)
2. The **summarizer** sends the transcript to Claude and receives structured facts (claim, confidence, category, entities)
3. Each fact is **embedded** as a dense vector and stored in the **fact store**
4. When a new session begins, the **retriever** embeds the user's first message and finds the most relevant facts via cosine similarity, with optional temporal decay to favor recent knowledge
5. The **context injector** formats the top facts into a "What I already know" section and prepends it to the system prompt, with a 2k token budget guard

### Schema

```sql
-- Conversation tracking
sessions (id VARCHAR PK, title, created_at, domain)
messages (id INTEGER, session_id, role, content, created_at)

-- Knowledge storage
facts (id INTEGER PK, session_id, claim TEXT, category, confidence FLOAT,
       entities VARCHAR[], embedding FLOAT[], created_at)
```

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.12 | Ecosystem compatibility with AI/ML tooling |
| Package manager | uv | Fast, modern Python packaging |
| Database | DuckDB | Serverless, fast local analytics, native array types for embeddings |
| AI API | Anthropic (Claude) | Fact extraction via structured prompting |
| Vectors | NumPy | Cosine similarity, vector normalization |
| Testing | pytest | Standard, with tmp_path fixtures for DB isolation |
| CLI | argparse | Zero-dependency, built into Python |

## CLI Interface

```
ctx sessions list                          # List all sessions
ctx sessions create <domain> [--title T]   # Create a new session
ctx summarize <session_id>                 # Extract facts from a session via Claude
ctx facts list                             # List all stored facts
ctx retrieve <query> [--top-k N]           # Semantic search over facts
                     [--min-confidence F]   # Filter by confidence threshold
                     [--decay L]            # Temporal decay lambda
```

## Key Design Decisions

- **Pluggable embedder protocol**: The `Embedder` interface allows swapping the test embedder for a production model (e.g., OpenAI `text-embedding-3-small`, Anthropic, or a local model) without changing any other code
- **Exponential temporal decay**: `score = similarity * exp(-lambda * days)` makes older facts gradually less relevant without hard cutoffs. Default lambda=0 disables decay
- **Token budget guard**: The injector truncates facts if they exceed ~2000 tokens, preventing context window overflow
- **Append-only facts**: Facts are never updated or deleted — the system is an accumulating knowledge log
- **Session isolation**: Facts are tied to sessions, enabling per-domain or per-project knowledge management

## Evaluation Results

A 10-session chain test validates end-to-end retention:

- **Sessions 1-5**: Introduce 10 ground-truth facts about "Project Phoenix" (team, architecture, timeline, performance, tech stack)
- **Sessions 6-10**: Query for those facts via the context injector

**Result: 10/10 facts retained across all 5 query sessions.** Acceptance threshold was 7/10.

| Session | Query | Score |
|---------|-------|-------|
| 6 | "What do you know about Project Phoenix?" | 10/10 |
| 7 | "Tell me about Phoenix's architecture and tech stack" | 10/10 |
| 8 | "Who works on Phoenix and what's the timeline?" | 10/10 |
| 9 | "What performance characteristics does Phoenix have?" | 10/10 |
| 10 | "Give me a full overview of the Phoenix project" | 10/10 |

## Test Suite

22 tests covering all modules:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_session.py | 3 | Session CRUD, message ordering, session isolation |
| test_facts.py | 4 | Insert/count, embeddings, batch insert, determinism |
| test_retriever.py | 3 | Top-k ranking, confidence filtering, 50-fact search |
| test_decay.py | 3 | Decay ranking, no-decay equality, high-decay penalty |
| test_injector.py | 4 | Prompt augmentation, confidence labels, empty store, truncation |
| test_summarizer.py | 4 | Fact extraction, embeddings, transcript format, batch extraction |
| test_eval.py | 1 | End-to-end retention threshold |

## Potential Use Cases

- **Developer tools**: AI coding assistants that remember project context, architecture decisions, and debugging history across sessions
- **Customer support**: Agents that accumulate knowledge about a customer's setup, past issues, and preferences
- **Research assistants**: Persistent memory of papers read, hypotheses explored, and findings across research sessions
- **Personal AI**: An assistant that learns user preferences, projects, and relationships over time
- **Team knowledge bases**: Shared fact stores where multiple agents contribute and retrieve organizational knowledge

## Potential Extensions

- **Real embedding model**: Replace `NumpyRandomEmbedder` with OpenAI `text-embedding-3-small` or a local model (e.g., sentence-transformers) for true semantic search
- **Fact conflict resolution**: Detect when new facts contradict old ones (e.g., "Phoenix launches Q3" vs "Phoenix delayed to Q4") and update confidence or mark as superseded
- **Multi-user / multi-agent**: Scope facts by user or agent identity, with access controls
- **Fact provenance UI**: A dashboard showing which facts were injected into each session and how they influenced responses
- **Streaming summarization**: Extract facts incrementally during a conversation rather than at session end
- **Hierarchical memory**: Separate short-term (session), medium-term (recent facts), and long-term (high-confidence, frequently retrieved) memory tiers
- **Graph-based retrieval**: Use entity relationships to traverse related facts beyond vector similarity
- **Forgetting / privacy**: TTL on facts, per-domain purging, or user-initiated fact deletion for compliance
- **Evaluation with real LLM**: Run the chain test with Claude generating responses, scoring whether answers actually incorporate injected facts
- **MCP server**: Expose the fact store as a Model Context Protocol server so any MCP-compatible agent can use it

## Applicability

The cascading context pattern applies wherever AI agents need to maintain state across interactions:

- **Any multi-session AI application** where context continuity matters
- **RAG systems** that need a lightweight, local alternative to vector databases for personal or small-team use
- **Agent frameworks** (LangChain, CrewAI, AutoGen) as a memory backend
- **Claude Code / Cursor / Copilot extensions** that persist project knowledge
- **Embedded AI** in applications where users interact with an assistant over days or weeks

The architecture is intentionally minimal — ~500 lines of core code, no external services beyond DuckDB and the Anthropic API — making it easy to embed, extend, or adapt to specific use cases.

## Project Stats

| Metric | Value |
|--------|-------|
| Core code | 481 lines |
| Test code | 525 lines |
| Eval code | 227 lines |
| Dependencies | 3 (anthropic, duckdb, numpy) |
| Dev dependencies | 1 (pytest) |
| Python version | 3.12+ |
| Test count | 22 |
| Retention score | 10/10 |
