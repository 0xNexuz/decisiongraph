# DecisionGraph

**Ask what your team decided. Get the decision that still stands.**

Teams write decisions down, then change their minds and write new ones. Search finds the old record as easily as the new one, so AI agents and new joiners act on decisions that were replaced years ago.

DecisionGraph stores decisions in [HydraDB](https://github.com/hydra-db/hydradb) as a graph. Each decision is linked to the decisions that replaced it. To answer a question, DecisionGraph walks those links to the decision that is still current. Every correction makes the graph sharper: it confirms or rejects links, learns which link rules to trust, adds missing links, and points the question's words at the right decision.

Built for the HydraDB Hackathon (Nov 14–16, 2026). It covers both tracks: **a graph that improves itself**, and **the context layer for agents, traversing structure instead of burning tokens**.

- Live app: _add the Render URL here_
- Walkthrough video: _add the Loom link here_

## Results

The test set has 50 hand-written questions about 128 real architecture decision records (ADRs). Every system answers each question **before** it gets the correction for that question (test-then-train), in five rounds of ten. We repeat this with five different question orders and report the mean.

| System | Right decision | Outdated or rejected answer | Context tokens | Right when a topic returns in new words |
|---|---|---|---|---|
| Vector search (MiniLM, top 3 passages) | 52% | 28% | 588 | 33% |
| Vector search + memory of past corrections | 57% | 22% | 600 | 47% |
| DecisionGraph, learning off | 74% | 18% | 135 | 61% |
| **DecisionGraph, learning on** | **82%** (66% in round 1, **90%** in round 5) | **10%** (4% in round 5) | **131** | **87%** |

What these show:

- **The graph does real work.** With the same embeddings and the same seeds, adding the decision graph takes accuracy from 52% to 74%. The context shrinks from ~590 tokens to ~135, because the agent gets the current decision and a one-line history instead of three loose passages.
- **Learning is real and generalises across wordings.** With learning on, accuracy climbs from 66% in round 1 to 90% in round 5. When a topic comes back in different words, the learning graph is right 87% of the time. Vector search with a memory of past corrections manages 47%.
- **The baseline was tuned for its best result.** The memory baseline's similarity threshold was set to its best value (0.65) after testing 0.50–0.80.

Full numbers are in `data/results.json`, with per-round scores, ranges across the five orders, a trace of every answer and what the graph learned from it, and the final rule trust. Reproduce them with `python -m app.evaluate`.

## How it works

```
question ──► seeds: vector similarity + (:Term)-[:CORE_OF]->(:Decision) + learned (:Term)-[:HINTS]->(:Decision)
         ──► walk: (:Decision)-[:REPLACED_BY]->(:Decision) while the link is trusted
         ──► answer: the current decision + a one-line history (≈130 tokens)
feedback ──► confirm/reject walked links · update rule trust (:Rule {a,b}) · add learned links · add hint edges
```

**Graph model (all in HydraDB):**

| Node / edge | Meaning |
|---|---|
| `(:Decision {key,title,status,date,project})` | one ADR |
| `(:Project)-[:HAS]->(:Decision)` | which platform it belongs to |
| `(:Term)-[:CORE_OF]->(:Decision)` | terms from the ADR's title and decision |
| `(:Decision)-[:REPLACED_BY {rule,state,w}]->(:Decision)` | candidate "this was replaced by that" link, proposed by a rule; `state` is `candidate`, `confirmed` or `rejected` |
| `(:Term)-[:HINTS {w}]->(:Decision)` | learned: these words point at this decision |
| `(:Rule {name,a,b})` | how far each link rule is trusted (Beta counts, learned) |

**Link rules.** Four simple rules propose `REPLACED_BY` links. Each starts with a prior trust:

| Rule | Example | Prior |
|---|---|---|
| status-link | "Superseded by [036]" in the status section | 0.90 |
| body-ref-replace | a newer ADR cites an older one near "replace", "migrate", "supersede" | 0.45 |
| replace-term | a newer ADR says it replaces or switches from the tool an older ADR chose | 0.35 |
| body-ref | a newer ADR cites an older one by number | 0.20 |

The walker follows a link when its trust is at least 0.5. A confirmed link counts as 1 and a rejected one as 0; otherwise the link gets its rule's trust. Corrections move the rule trust, so a rule that keeps proving right starts being followed for topics nobody has corrected yet.

**Guards against learning the wrong thing:** a new link is only learned between decisions in the same project, in date order, and about the same subject (embedding similarity or a shared distinctive term). A path is only confirmed if every hop is either between related decisions or backed by strong evidence.

**Why not a vector store?** A vector store can't say that one document replaced another, so it can't walk from an old decision to the current one. Remembering past answers only helps when a question is worded almost the same. The graph's lessons are stored as structure, so they carry over to new wordings and to neighbouring questions.

## Run it locally

You need HydraDB's `graph-node` (build it from [hydra-db/hydradb](https://github.com/hydra-db/hydradb); see its README) and Python 3.11+.

```bash
pip install -r requirements.txt
# start graph-node on 127.0.0.1:8443 (see start.sh for the environment it needs)
python -m app.evaluate                     # reproduce the results
uvicorn app.server:app --port 8000         # the web app on http://localhost:8000
```

Or build the single container that does both: `docker build -t decisiongraph . && docker run -p 10000:10000 decisiongraph`.

## Deploy (free)

1. GitHub Actions (`.github/workflows/image.yml`) builds HydraDB from source together with the app, and publishes `ghcr.io/0xnexuz/decisiongraph:latest`.
2. Render's free tier runs that image as a web service. HydraDB runs inside the same container with local storage and small caches, using about 240 MB of RAM in total.
3. `.github/workflows/keep-awake.yml` pings the app every 10 minutes so it doesn't fall asleep (set the `APP_URL` repository variable).

On startup the app loads the graph and teaches it the 50 test corrections, so the live demo shows a graph that has already learned. The example questions on the page are worded differently from the test set. **Replay** resets the graph and runs the whole test live against HydraDB.

## Honest limits

- 50 questions over 128 ADRs is a small test. The questions and their right answers were written by hand for this project (see `data/questions.json`).
- The graph seeds from the same MiniLM embeddings as the baseline (int8, quantized from `Xenova/all-MiniLM-L6-v2`). A stronger embedding model would lift every system.
- Corrections in the live demo are shared by everyone using it, and Replay resets them.

## Data and licences

- Decision records: [ministryofjustice/cloud-platform](https://github.com/ministryofjustice/cloud-platform), [ministryofjustice/modernisation-platform](https://github.com/ministryofjustice/modernisation-platform), [alphagov/govuk-aws](https://github.com/alphagov/govuk-aws), all MIT. See `data/SOURCES.md`.
- HydraDB: [hydra-db/hydradb](https://github.com/hydra-db/hydradb), Elastic License 2.0, built from source at a pinned commit.
- Embedding model: all-MiniLM-L6-v2, Apache 2.0 (`models/minilm/LICENSE`).
- Font: Archivo, SIL Open Font License (`web/fonts/OFL.txt`).
- Code in this repository: MIT (see `LICENSE`).
