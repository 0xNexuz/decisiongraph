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
