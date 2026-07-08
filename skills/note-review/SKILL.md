---
name: note-review
description: Use this skill when editing the user's personal AI/software-engineering notes. Check for factual errors or overconfident claims, then lightly rewrite the note into readable Markdown with clear bullets while preserving the user's original meaning and learning trace.
---

# Note Review

Use this skill for the user's personal learning notes, especially AI, software engineering, paper reading, and daily technical reflections.

## Goals

1. Check whether the note has factual errors, overconfident claims, or unclear technical wording.
2. Rewrite into readable Markdown with headings, bullets, and short paragraphs.
3. Preserve the user's original content and learning trace. Do not turn rough notes into a polished article unless asked.

## Workflow

- Read the full note first.
- Separate facts from guesses.
- Mark uncertain model-specific claims as `待核实` instead of deleting them.
- Keep the user's core wording when it is already clear.
- Add only small clarifications that improve correctness or readability.
- Prefer concise headings and bullet lists.
- Avoid large rewrites, extra theory dumps, or unrelated background.

## Review Checklist

- Are model names, paper names, and architecture claims accurate?
- Is a claim actually confirmed, or should it be marked `待核实`?
- Are concepts like attention, KV cache, MoE, MTP, RAG, RLHF, SFT, and inference optimization described precisely enough?
- Is the note easy to scan later?
- Does each section have a clear topic?

## Output Style

When editing a file, produce clean Markdown directly.

When only reviewing, respond with:

- `事实性问题`
- `不确定/待核实`
- `格式建议`
- `可选改写`
