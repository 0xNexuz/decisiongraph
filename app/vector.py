"""Vector-store baselines.

VectorRAG        : plain top-k chunk retrieval over ADR text (MiniLM embeddings).
VectorRAGMemory  : the same, plus a feedback memory: confirmed question -> answer
                   pairs are stored as vectors and reused when a new question is
                   very similar to an old one. This is the fair "vector store that
                   also learns" baseline.
"""
import numpy as np

from .embed import embed, embed_cached

TOP_K = 3
ABSTAIN_BELOW = 0.42  # same abstention rule as DecisionGraph: no chunk this close means no answer


class VectorRAG:
    name = "vector"

    def __init__(self, recs, chunks, chunk_vecs=None):
        self.recs = {r["id"]: r for r in recs}
        self.chunks = chunks
        self.vecs = chunk_vecs if chunk_vecs is not None else embed_cached([c["text"] for c in chunks], "chunks")

    def search(self, qvec, k=TOP_K):
        sims = self.vecs @ qvec
        order = np.argsort(-sims)
        out, seen = [], set()
        for i in order:
            c = self.chunks[i]
            out.append((c, float(sims[i])))
            seen.add(c["adr"])
            if len(out) >= k:
                break
        return out

    def answer(self, q, qvec):
        hits = self.search(qvec)
        if hits[0][1] < ABSTAIN_BELOW:
            return dict(answer=None, context="No decision on record answers this question.", retrieved=[], scores=[], path=[], abstained=True)
        top = hits[0][0]["adr"]
        context = "\n\n".join(h[0]["text"] for h in hits)
        return dict(answer=top, context=context, retrieved=[h[0]["adr"] for h in hits],
                    scores=[round(h[1], 3) for h in hits], path=[])

    def feedback(self, q, qvec, result, correct_id):
        pass

    def reset(self):
        pass


class VectorRAGMemory(VectorRAG):
    name = "vector+memory"
    THRESHOLD = 0.65  # best value for this baseline over 0.50-0.80 (tuned to give it its best shot)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mem_vecs, self.mem_ans, self.mem_q = [], [], []

    def reset(self):
        self.mem_vecs, self.mem_ans, self.mem_q = [], [], []

    def answer(self, q, qvec):
        res = super().answer(q, qvec)
        if self.mem_vecs and not res.get("abstained"):
            sims = np.array(self.mem_vecs) @ qvec
            j = int(np.argmax(sims))
            if sims[j] >= self.THRESHOLD:
                ans = self.mem_ans[j]
                r = self.recs[ans]
                res["answer"] = ans
                res["context"] = f"{r['title']}. {r['decision']}\n\n" + res["context"]
                res["memory_hit"] = self.mem_q[j]
        return res

    def feedback(self, q, qvec, result, correct_id):
        if correct_id is None:
            return
        self.mem_vecs.append(qvec)
        self.mem_ans.append(correct_id)
        self.mem_q.append(q)


class PlainStore(VectorRAGMemory):
    """The strongest 'plain store' we could build without a graph: vector search plus the
    metadata a document store keeps (status and date) plus the correction memory above.
    - drops records whose status says superseded or rejected,
    - among the closest records in the same project, prefers the newest one,
    - reuses a past correction when the question is close enough.
    This is the answer to "would this fall apart on a plain store?". POOL and NEAR were tuned
    over 3-10 and 0.04-0.15 to give this baseline its best score."""
    name = "plain-store"
    POOL = 3        # how many nearby records the recency rule looks at
    NEAR = 0.08     # only records within this similarity of the best one count as "about the same thing"

    def answer(self, q, qvec):
        sims = self.vecs @ qvec
        order = np.argsort(-sims)
        if sims[order[0]] < ABSTAIN_BELOW:
            return dict(answer=None, context="No decision on record answers this question.", retrieved=[], scores=[], path=[], abstained=True)
        best, cands = None, []
        for i in order:
            adr = self.chunks[i]["adr"]
            r = self.recs[adr]
            if r["status"] in ("superseded", "rejected", "partly-superseded"):
                continue
            if best is None:
                best = float(sims[i])
            if float(sims[i]) < best - self.NEAR or len(cands) >= self.POOL:
                break
            if adr not in [c[0] for c in cands]:
                cands.append((adr, float(sims[i])))
        if not cands:
            return super().answer(q, qvec)
        top_project = self.recs[cands[0][0]]["project"]
        same = [c for c in cands if self.recs[c[0]]["project"] == top_project]
        pick = max(same, key=lambda c: self.recs[c[0]]["date"])[0]
        r = self.recs[pick]
        res = dict(answer=pick, context=f"{r['title']}. {r['decision']}", retrieved=[c[0] for c in cands], scores=[], path=[])
        if self.mem_vecs:
            ms = np.array(self.mem_vecs) @ qvec
            j = int(np.argmax(ms))
            if ms[j] >= self.THRESHOLD:
                a = self.mem_ans[j]
                res["answer"] = a
                res["context"] = f"{self.recs[a]['title']}. {self.recs[a]['decision']}"
        return res
