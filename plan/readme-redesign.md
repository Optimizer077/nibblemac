# README redesign (owner request 2026-10-22 16:55: "major visual and structure update", after testing)

Starts when the open tests are final: 4B prefill speed (merge), overclock A/B. Numbers come from
result files only, as now.

## Problems with the current README (516 lines, 17 figures)
- The first screen is text: a paragraph, five bullets, a table. The speed story needs a scroll.
- Sections 1, 2 and 4 are a long technical essay (hardware, kernels, speculation, what failed) in the
  middle of the page; a visitor looking for "what is it, how fast, how do I run it" scrolls past it.
- No picture of the system as a whole, no "what it feels like" demo, no table of supported models
  (0.8B, 2B, Cerno 4B, GGUF import).
- Usage snippets for C, Python and the server are long and always expanded.

## Target structure (README about 200 lines; nothing is deleted, the essay moves to docs)
1. **Hero:** logo, name, tagline, badges (as now), plus a **real-time demo GIF**: the same chat prompt on
   the same Pi, pinfer (speculative) next to llama.cpp, replayed from recorded per-token timestamps
   (honest real time, labelled as such).
2. **At a glance:** one scorecard figure (desktop / phone × light / dark): 20.2 tok/s speculative, 7.9 plain,
   43 prefill, 6.9 at 4096 context, Cerno 4B decisions 4.5× llama.cpp; under it the comparison table.
3. **Quickstart:** three numbered steps (install the wheel, convert or import a model, chat / serve / Python).
4. **Models:** table: Qwen3.5-0.8B, Qwen3.5-2B, Cerno 4B (via cerno-pi), any Qwen3.5 Q4_0 / Q8_0 GGUF;
   speed, memory, exactness for each.
5. **How it works in four cards:** memory wall (decode animation), weight format and kernels,
   exact speculative decoding (animation), context and state reuse. Each 3–4 sentences with a link to
   **docs/HOW_IT_WORKS.md** (today's sections 1, 2, 4 with all their figures, unchanged in substance).
6. **System overview figure:** `.pq` model → engine (kernels, speculation, KV / state) → interfaces
   (C API, Python, CLI, OpenAI server) → uses (chat, cerno-pi).
7. **Using it:** Python, C, server in `<details>` blocks.
8. **Correctness and records:** short, with links (RESULTS.md, journal, CONTRIBUTING).
9. **Roadmap, related projects, credits, license, trademark notice.**

## New visuals (same figstyle palette, four variants each, alt text, phone-checked)
- scorecard figure; system overview figure; real-time demo GIF (needs one recording run on the Pi
  under the lock, final binary, both engines).

## Checks before merge
- Every number traced to a result file (script that lists README numbers vs RESULTS.md).
- Live page: all images load, phone width (360 / 412 px) and dark mode viewed.
- No broken anchors or links (link checker over README and docs).
