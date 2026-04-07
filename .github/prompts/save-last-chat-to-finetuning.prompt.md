---
name: "Save Last Chat To Finetuning"
description: "prompt for appending the latest chat ask into finetuning.md as Q&A with beginner-to-senior conceptual explanation."
argument-hint: "Optional topic override (leave blank to use latest chat ask)"
agent: "Save Last Chat To Finetuning"
---
Use this workflow to save the latest user question/topic into [finetuning.md](../../finetuning.md) in interview-prep format.

Requirements:
- Add one new Q&A section only.
- Keep progression: beginner -> mid-level -> senior.
- Include: simple definition, intuition, technical understanding, practical example, and interview one-liner.
- Place the entry in the existing Q-section sequence using the next Q number.

Input handling:
- If user provides a topic argument, use it.
- Otherwise, use the most recent user ask from chat context.
