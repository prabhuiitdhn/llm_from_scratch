# LLM Chatbot — System Design (Short Overview)

This note captures the high-level, step-by-step system design for an LLM-based chatbot application. Deeper dives into each stage (memory strategies, guardrails, streaming architecture, RAG integration) will be added later.

---

## Pipeline at a glance

```
User → Input Handling → Context Assembly → LLM Inference → Post-processing → Response
```

---

## 1. Input Handling
- Receive user message (text, sometimes voice/image).
- Basic safety checks: input length limits, PII/profanity filtering, prompt-injection sanitation.

## 2. Session & Memory Management
- Maintain conversation history (short-term memory) — usually last N turns or token-budgeted.
- Optionally fetch long-term memory/user profile from a database (preferences, past facts).

### Deep Dive: Memory Stage

Memory is what makes a chatbot feel like it's having a *conversation* instead of answering isolated one-off prompts. Since LLMs are stateless (no memory between API calls), the application layer must simulate memory.

**Types of memory:**

| Type | Scope | Lifetime | Example |
|---|---|---|---|
| Short-term (session) memory | Current conversation | Cleared when session ends | Last 10 turns of this chat |
| Long-term (persistent) memory | Across sessions | Survives forever until deleted | User's name, preferences, past issues raised |
| Working memory | Current task/turn | Single request | Scratchpad for reasoning, tool outputs |

**1. Short-term (conversation) memory**

Simplest approach: store raw turn history and replay it into the prompt each time.

```
Turn 1: User: "My name is Raj"             → stored
Turn 2: Assistant: "Nice to meet you, Raj" → stored
Turn 3: User: "What's my name?"            → new request includes turns 1-2 as context
```

Problem: the context window is finite — as conversation grows, you can't keep sending everything. Common strategies to manage this:
- **Sliding window**: keep only the last N turns.
- **Token-budgeted truncation**: keep as many recent turns as fit in X tokens, drop oldest first.
- **Summarization**: periodically compress older turns into a short summary ("User asked about billing issues, resolved on 8/20") and keep only recent turns verbatim.
- **Hybrid**: rolling summary + last few raw turns (most common in production).

**2. Long-term memory**

Used when the chatbot needs to remember facts *across sessions* (different day, different device).
- Stored outside the prompt — typically a database or vector store.
- Retrieved selectively based on relevance (not dumped in wholesale), often via:
  - **Key-value facts**: structured profile store (name, preferences, account tier).
  - **Vector-based memory**: embed past conversation snippets, retrieve semantically relevant ones at query time (same retrieval pattern as RAG, just over "memory documents" instead of a knowledge base).
- Written to asynchronously after a conversation (extract salient facts → store), so it doesn't add latency to the live response.

**3. Working memory**
- Transient scratch space within a single turn — e.g., intermediate tool-call results, reasoning steps, retrieved chunks for this specific question.
- Discarded after the response is generated; not persisted.

**Why this matters for system design:**
- **Cost/latency**: more history = more tokens = higher cost and slower inference. Memory strategy is really a *token budget management* problem.
- **Relevance over completeness**: dumping all history isn't always better — irrelevant old context can dilute attention and degrade answer quality.
- **Consistency risk**: if summarization drops details, the bot may "forget" things it should remember — needs periodic evaluation.
- **Privacy/compliance**: long-term memory storing PII needs retention policies, user consent, and deletion mechanisms (right to be forgotten).

One-line summary: Memory in a chatbot system is split into short-term (recent turns, managed via windowing/summarization to fit the context budget) and long-term (persisted facts/embeddings retrieved selectively across sessions), with the core engineering challenge being *what to keep, what to compress, and what to retrieve* without blowing the token budget or losing important context.

## 3. Context Assembly (Prompt Construction)
- Combine: system prompt (persona/instructions) + retrieved context (if RAG-enabled) + conversation history + current user message.
- Fit everything within the model's context window (truncate oldest turns first if needed).

## 4. (Optional) Retrieval Step
- If chatbot needs external/private knowledge → embed query → vector search → fetch top-k chunks → inject into prompt (this is where RAG plugs in).

## 5. LLM Inference
- Send assembled prompt to the model (hosted API or self-hosted).
- Apply decoding params: temperature, top-p, max_tokens, stop sequences.
- Streaming response tokens back to the client for perceived low latency.

## 6. Post-processing / Guardrails
- Output moderation (toxicity, hallucination checks, policy compliance).
- Optional structured formatting (JSON, markdown) if downstream systems consume it.
- Tool/function-call parsing if the chatbot supports actions (e.g., "book a meeting").

## 7. Response Delivery & Logging
- Return response to user (chat UI/API).
- Log prompt, response, latency, token usage for observability and future fine-tuning/eval.

## 8. Feedback Loop
- Capture thumbs up/down or implicit signals.
- Use for periodic evaluation, prompt tuning, or fine-tuning data collection.

---

## One-line summary

A chatbot pipeline is *input → memory/context assembly → (optional retrieval) → LLM inference with decoding controls → safety/output processing → response + logging*, with a feedback loop closing the system for continuous improvement.
