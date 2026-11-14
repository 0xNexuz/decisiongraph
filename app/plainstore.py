"""The strongest plain store we know how to build without a graph database.

Written from an outside reviewer's version. Python dicts only:
- the same hybrid seeding DecisionGraph starts from (vector similarity, IDF term overlap,
  project filter, same constants), minus anything learned,
- follows the "superseded by" pointer a record's status section names, as far as it goes
  (a document store can keep that as a metadata field),
- a vector memory of past corrections (threshold 0.65, the value that suits it best),
  with the pointer chase applied after.
It cannot infer a replacement nobody wrote down, and it cannot learn one from a correction.
"""
from collections import defaultdict

import numpy as np

from .corpus import tokens
from .graph import ABSTAIN_BELOW, GENERIC, detect_project


class PlainStore:
    name = "plain-store"
    THRESHOLD = 0.65

    def __init__(self, recs, per, idf, links, chunks, chunk_vecs):
        self.R = {r["id"]: r for r in recs}
        self.idf, self.ch, self.cv = idf, chunks, chunk_vecs
        self.vocab = {t for core, _ in per.values() for t in core if idf.get(t, 0) > 0.8}
        self.inv = defaultdict(set)
        for rid, (core, _) in per.items():
            for t in core:
                if t in self.vocab:
                    self.inv[t].add(rid)
        self.sup = defaultdict(list)
        for l in links:
            if l["rule"] == "status-link":
                self.sup[l["old"]].append(l["new"])
        self.reset()

    def reset(self):
        self.mv, self.ma = [], []

    def chase(self, rid, depth=5):
        best = [rid]

        def dfs(n, p):
            nonlocal best
            if len(p) > len(best):
                best = p
            if len(p) > depth:
                return
            for m in self.sup.get(n, []):
                if m not in p:
                    dfs(m, p + [m])
        dfs(rid, [rid])
        return best[-1]

    def seed(self, q, v):
        sims = self.cv @ v
        vec = {}
        for i in np.argsort(-sims)[:24]:
            a = self.ch[i]["adr"]
            vec[a] = max(vec.get(a, 0), float(sims[i]))
        terms = sorted([t for t in dict.fromkeys(tokens(q)) if t in self.vocab and t not in GENERIC],
                       key=lambda t: -self.idf.get(t, 0))[:8]
        ts = defaultdict(float)
        for t in terms:
            for rid in self.inv[t]:
                ts[rid] += self.idf[t]
        p = detect_project(q)
        sc = {}
        for rid in set(vec) | set(ts):
            s = vec.get(rid, 0) + 0.04 * min(ts.get(rid, 0), 8.0)
            if p and self.R[rid]["project"] != p:
                s -= 1.0
            sc[rid] = s
        return sorted(sc.items(), key=lambda kv: -kv[1])

    def answer(self, q, qvec):
        rk = self.seed(q, qvec)
        if rk[0][1] < ABSTAIN_BELOW:
            return dict(answer=None, context="No decision on record answers this question.", path=[], abstained=True)
        ans = self.chase(rk[0][0])
        if self.mv:
            s = np.array(self.mv) @ qvec
            j = int(np.argmax(s))
            if s[j] >= self.THRESHOLD:
                ans = self.chase(self.ma[j])
        r = self.R[ans]
        return dict(answer=ans, context=f"{r['title']}. {r['decision']}", path=[])

    def feedback(self, q, qvec, result, correct_id):
        if correct_id is None:
            return []
        self.mv.append(qvec)
        self.ma.append(correct_id)
        return []
