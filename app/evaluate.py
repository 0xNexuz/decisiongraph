"""Continual-learning evaluation (prequential / test-then-train).

Questions arrive as a stream in rounds. Every system answers each question
BEFORE it gets feedback on it, so every score is on a question the system has
not been told the answer to. After answering, the system receives the correct
decision (the same signal a user's thumbs-up/down plus correction gives) and may
learn from it.

Metrics (per round):
  accuracy        the decision presented as current is the right one
  stale_rate      the decision presented as current is outdated/rejected for
                  this question (the failure that makes agents act on old info)
  tokens          context tokens handed to the agent (WordPiece tokenizer)
  first_seen_acc  accuracy on questions whose topic has never appeared before
                  (pure generalisation: no memory of this topic to lean on)
"""
import json
import os
import random
import sys
import time

import numpy as np

from .corpus import DATA, build, chunks as make_chunks
from .embed import embed, embed_cached, _load
from .plainstore import PlainStore
from .vector import VectorRAG, VectorRAGMemory

ROUNDS = 5
NAMES = ["vector", "vector+memory", "plain-store", "seeds-only", "decisiongraph-frozen", "decisiongraph"]


def confidence(per_q, n_boot=4000, seed=0):
    """95% intervals by resampling the 50 questions (each question's score is its mean over the
    five question orders). Differences are paired: the same resampled questions for both systems."""
    rng = np.random.default_rng(seed)
    qids = sorted(per_q["decisiongraph"])
    acc = {k: np.array([np.mean([o[0] for o in v[q]]) for q in qids]) for k, v in per_q.items()}
    stale = {k: np.array([np.mean([o[1] for o in v[q]]) for q in qids]) for k, v in per_q.items()}
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    out = {"systems": {}, "diff_vs_decisiongraph": {}}
    for k in acc:
        out["systems"][k] = dict(accuracy=round(float(acc[k].mean()), 3), accuracy_ci=ci(acc[k][idx].mean(1)),
                                 stale_rate=round(float(stale[k].mean()), 3), stale_ci=ci(stale[k][idx].mean(1)))
        if k != "decisiongraph":
            d = acc["decisiongraph"] - acc[k]
            out["diff_vs_decisiongraph"][k] = dict(points=round(float(d.mean()), 3), ci=ci(d[idx].mean(1)))
    # learning on vs off on subsets of the stream, paired by (order, question), resampling whole questions
    L, F = per_q["decisiongraph"], per_q["decisiongraph-frozen"]

    def subset(keep):
        num = np.zeros(len(qids)); cnt = np.zeros(len(qids))
        for i, q in enumerate(qids):
            for a, b in zip(L[q], F[q]):
                if keep(a):
                    num[i] += a[0] - b[0]; cnt[i] += 1
        boot = num[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
        return dict(points=round(float(num.sum() / cnt.sum()), 3), ci=ci(boot), n=int(cnt.sum()))
    out["learning_vs_frozen"] = dict(all=subset(lambda a: True), rounds_4_5=subset(lambda a: a[2] >= 3),
                                     repeat_topic=subset(lambda a: not a[3]), first_seen=subset(lambda a: a[3]))
    return out


def count_tokens(text):
    _, tok = _load()
    tok.no_truncation()
    n = len(tok.encode(text).ids)
    tok.enable_truncation(256)
    return n


def load_questions():
    return json.load(open(os.path.join(DATA, "questions.json")))["questions"]


def order(questions, seed):
    qs = list(questions)
    random.Random(seed).shuffle(qs)
    return qs


def run(systems, questions, qvecs, seed, verbose=False):
    stream = order(questions, seed)
    per_round = len(stream) // ROUNDS
    res = {s.name: [] for s in systems}
    seen_topics = set()
    log = []
    for i, item in enumerate(stream):
        rnd = min(i // per_round, ROUNDS - 1)
        first = item["topic"] not in seen_topics
        qv = qvecs[item["id"]]
        for s in systems:
            t0 = time.time()
            out = s.answer(item["q"], qv)
            ms = (time.time() - t0) * 1000
            ok = out["answer"] == item["answer"]
            stale = out["answer"] in item["stale"]
            changes = s.feedback(item["q"], qv, out, item["answer"])
            res[s.name].append(dict(round=rnd, ok=ok, stale=stale, first=first, tokens=count_tokens(out["context"]),
                                    ms=ms, qid=item["id"], got=out["answer"], want=item["answer"]))
            if verbose and s.name == "decisiongraph":
                log.append(dict(qid=item["id"], q=item["q"], round=rnd + 1, ok=ok, got=out["answer"],
                                want=item["answer"], start=out.get("start"), path=out.get("path"), learned=changes))
        seen_topics.add(item["topic"])
    return res, log


def summarise(rows):
    out = []
    for r in range(ROUNDS):
        xs = [x for x in rows if x["round"] == r]
        fs = [x for x in xs if x["first"]]
        rp = [x for x in xs if not x["first"]]
        out.append(dict(round=r + 1, n=len(xs),
                        repeat_n=len(rp),
                        repeat_acc=round(float(np.mean([x["ok"] for x in rp])), 3) if rp else None,
                        accuracy=round(float(np.mean([x["ok"] for x in xs])), 3),
                        stale_rate=round(float(np.mean([x["stale"] for x in xs])), 3),
                        tokens=round(float(np.mean([x["tokens"] for x in xs])), 1),
                        first_seen_n=len(fs),
                        first_seen_acc=round(float(np.mean([x["ok"] for x in fs])), 3) if fs else None))
    rp = [x for x in rows if not x["first"]]
    fs = [x for x in rows if x["first"]]
    overall = dict(accuracy=round(float(np.mean([x["ok"] for x in rows])), 3),
                   repeat_acc=round(float(np.mean([x["ok"] for x in rp])), 3),
                   first_seen_acc=round(float(np.mean([x["ok"] for x in fs])), 3),
                   stale_rate=round(float(np.mean([x["stale"] for x in rows])), 3),
                   tokens=round(float(np.mean([x["tokens"] for x in rows])), 1),
                   ms=round(float(np.mean([x["ms"] for x in rows])), 1))
    return dict(rounds=out, overall=overall)


class NoWalk:
    """DecisionGraph's seeding alone, no walk along 'replaced by' links (ablation)."""
    name = "seeds-only"

    def __init__(self, g):
        self.g = g

    def answer(self, q, qv):
        ranked, terms = self.g.seeds(q, qv)
        rid = ranked[0][0]
        return dict(answer=rid, context=self.g.context(rid, []), path=[])

    def feedback(self, *a):
        return []


def score_fixed(systems, items, vecs):
    """Answer a fixed list with no feedback (for held-out and out-of-scope sets)."""
    out = {}
    for s in systems:
        rows = []
        for it in items:
            r = s.answer(it["q"], vecs[it["id"]])
            rows.append(dict(id=it["id"], kind=it.get("kind", "oos"), ok=r["answer"] == it["answer"], got=r["answer"], want=it["answer"]))
        out[s.name] = rows
    return out


class Frozen:
    """DecisionGraph with learning switched off (ablation)."""
    name = "decisiongraph-frozen"

    def __init__(self, g):
        self.g = g

    def answer(self, q, qv):
        return self.g.answer(q, qv)

    def feedback(self, *a):
        return []


def main(seeds=(7, 11, 23, 42, 99)):
    from .graph import DecisionGraph
    recs, per, idf, links = build()
    ch = make_chunks(recs)
    cvecs = embed_cached([c["text"] for c in ch], "chunks")
    questions = load_questions()
    qv = embed_cached([q["q"] for q in questions], "questions")
    qvecs = {q["id"]: qv[i] for i, q in enumerate(questions)}
    vec = VectorRAG(recs, ch, cvecs)
    mem = VectorRAGMemory(recs, ch, cvecs)
    plain = PlainStore(recs, per, idf, links, ch, cvecs)
    graph = DecisionGraph(recs, per, idf, links, vec)
    H = json.load(open(os.path.join(DATA, "heldout.json")))
    heldout, oos = H["heldout"], H["out_of_scope"]
    hv = embed_cached([x["q"] for x in heldout + oos], "heldout")
    hvecs = {x["id"]: hv[i] for i, x in enumerate(heldout + oos)}
    report = dict(seeds=list(seeds), runs=[], questions=len(questions), decisions=len(recs),
                  candidate_links=len(links))
    per_q = {}
    for sd in seeds:
        mem.reset()
        plain.reset()
        graph.load()
        res, log = run([vec, mem, plain, graph], questions, qvecs, sd, verbose=(sd == seeds[0]))
        if sd == seeds[0]:
            report["trace"] = log
            report["rules_after"] = graph.rules()
            report["graph_after"] = graph.stats()
        # held-out and out-of-scope questions, answered in the state the learning run left each system in
        trained = score_fixed([vec, mem, plain, graph], heldout + oos, hvecs)
        graph.load()
        fresh = score_fixed([Frozen(graph), NoWalk(graph)], heldout + oos, hvecs)
        report.setdefault("heldout_runs", []).append({**trained, **fresh})
        frozen, _ = run([Frozen(graph), NoWalk(graph)], questions, qvecs, sd)
        res.update(frozen)
        for name, rows in res.items():
            for x in rows:
                per_q.setdefault(name, {}).setdefault(x["qid"], []).append((x["ok"], x["stale"], x["round"], x["first"]))
        report["runs"].append(dict(seed=sd, systems={k: summarise(v) for k, v in res.items()}))
        print(f"seed {sd}: " + "  ".join(
            f"{k}: " + " ".join(str(r['accuracy']) for r in summarise(v)['rounds']) for k, v in res.items()), flush=True)
    # mean over seeds
    agg = {}
    for name in NAMES:
        rounds = []
        for r in range(ROUNDS):
            vals = [run_["systems"][name]["rounds"][r] for run_ in report["runs"]]
            fs = [v["first_seen_acc"] for v in vals if v["first_seen_acc"] is not None]
            rp = [v["repeat_acc"] for v in vals if v["repeat_acc"] is not None]
            rounds.append(dict(round=r + 1,
                               accuracy=round(float(np.mean([v["accuracy"] for v in vals])), 3),
                               accuracy_min=min(v["accuracy"] for v in vals),
                               accuracy_max=max(v["accuracy"] for v in vals),
                               stale_rate=round(float(np.mean([v["stale_rate"] for v in vals])), 3),
                               tokens=round(float(np.mean([v["tokens"] for v in vals])), 1),
                               first_seen_acc=round(float(np.mean(fs)), 3) if fs else None,
                               repeat_acc=round(float(np.mean(rp)), 3) if rp else None))
        ov = [run_["systems"][name]["overall"] for run_ in report["runs"]]
        agg[name] = dict(rounds=rounds, overall=dict(
            accuracy=round(float(np.mean([o["accuracy"] for o in ov])), 3),
            repeat_acc=round(float(np.mean([o["repeat_acc"] for o in ov])), 3),
            first_seen_acc=round(float(np.mean([o["first_seen_acc"] for o in ov])), 3),
            stale_rate=round(float(np.mean([o["stale_rate"] for o in ov])), 3),
            tokens=round(float(np.mean([o["tokens"] for o in ov])), 1),
            ms=round(float(np.mean([o["ms"] for o in ov])), 1)))
    report["mean"] = agg
    # held-out summary (mean over the five runs)
    ho = {}
    for name in NAMES:
        runs = [r[name] for r in report["heldout_runs"] if name in r]
        def acc(kind):
            vals = [np.mean([x["ok"] for x in rr if x["kind"] == kind]) for rr in runs]
            return round(float(np.mean(vals)), 3)
        ho[name] = dict(rep=acc("rep"), new=acc("new"), out_of_scope=acc("oos"),
                        all_heldout=round(float(np.mean([np.mean([x["ok"] for x in rr if x["kind"] != "oos"]) for rr in runs])), 3))
    report["heldout"] = ho
    report["ci"] = confidence(per_q)
    print("95% CI:", json.dumps(report["ci"]))
    print("held-out:", json.dumps(ho))
    out = os.path.join(DATA, os.environ.get("RESULTS_FILE", "results.json"))
    json.dump(report, open(out, "w"), indent=1, default=str)
    for name, a in agg.items():
        print(f"{name:14} " + " | ".join(f"R{r['round']} acc {r['accuracy']:.2f} stale {r['stale_rate']:.2f} tok {r['tokens']:.0f}" for r in a["rounds"]))
    return report


if __name__ == "__main__":
    main()
