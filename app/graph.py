"""DecisionGraph: a decision graph in HydraDB that gets sharper with every answer.

Graph model (all stored in HydraDB):
  (:Project)-[:HAS]->(:Decision)
  (:Term)-[:CORE_OF]->(:Decision)               terms from an ADR's title and decision
  (:Decision)-[:REPLACED_BY {rule,state,w}]->(:Decision)
                                                  candidate "this decision was replaced by that one"
                                                  links, each proposed by a rule
  (:Term)-[:HINTS {w}]->(:Decision)               learned from feedback
  (:Rule {name,a,b})                              how far each link rule can be trusted,
                                                  learned from feedback (Beta counts)

Answering walks the graph: find the decision the question is about, then
follow trusted REPLACED_BY links forward until reaching the decision that is
still current. Feedback confirms or rejects links, adjusts rule trust, adds
learned links and hint edges, so later questions (including new ones) walk
better paths.
"""
import re
from collections import defaultdict

import numpy as np

from .corpus import PROJECT_LABEL, PROJECT_NAMES, RULE_PRIOR, tokens
from .hydra import Hydra

PROJECT_VID = {"cloud-platform": 10, "modernisation-platform": 20, "govuk-aws": 30}
RULE_VID = {name: 50 + i for i, name in enumerate(RULE_PRIOR)}
TERM_BASE = 100000
FOLLOW = 0.5          # follow a link when its trust is at least this
PRIOR_STRENGTH = 6.0  # how many observations a rule prior is worth
MAX_HOPS = 5


def chunked(xs, n=900):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


GENERIC = {"cloud", "platform", "modernisation", "gov.uk", "govuk", "moj", "aws", "container", "mp", "cp",
           "uk", "gov", "team", "service", "services", "still", "now"}


def detect_project(q):
    ql = " " + q.lower() + " "
    found = [p for p, names in PROJECT_NAMES.items() if any(n in ql for n in names)]
    return found[0] if len(found) == 1 else None


class DecisionGraph:
    name = "decisiongraph"

    def __init__(self, recs, per_terms, idf, links, vector, db=None):
        self.db = db or Hydra()
        self.recs = {r["id"]: r for r in recs}
        self.by_vid = {r["vid"]: r for r in recs}
        self.per_terms, self.idf, self.links, self.vector = per_terms, idf, links, vector
        vocab = sorted({t for core, _ in per_terms.values() for t in core if idf.get(t, 0) > 0.8})
        self.term_vid = {t: TERM_BASE + i for i, t in enumerate(vocab)}
        self._project_members = {}
        from .embed import embed_cached
        ids = list(self.recs)
        v = embed_cached([f"{self.recs[i]['title']}. {self.recs[i]['decision'][:600]}" for i in ids], "adrs")
        self.adr_vec = {i: v[k] for k, i in enumerate(ids)}

    def related(self, a, b):
        """Are two decisions about the same thing? (guards learned links)"""
        ta, tb = self.per_terms[a][0], self.per_terms[b][0]
        shared = [t for t in ta & tb if self.idf.get(t, 0) > 2.5 and t not in GENERIC]
        return float(self.adr_vec[a] @ self.adr_vec[b]) >= 0.5 or len(shared) >= 1

    # ---------- loading ----------
    def clear(self):
        ids = [r["vid"] for r in self.recs.values()] + list(self.term_vid.values()) + \
              list(PROJECT_VID.values()) + list(RULE_VID.values())
        for part in chunked([{"id": i} for i in ids]):
            self.db.q("UNWIND $rows AS row MATCH (n {id: row.id}) DETACH DELETE n", rows=part)

    def load(self):
        db = self.db
        self.clear()
        # nodes first (labels + properties), then edges
        dec_rows = [dict(id=r["vid"], key=r["id"], title=r["title"], status=r["status"], date=r["date"],
                         project=r["project"]) for r in self.recs.values()]
        for part in chunked(dec_rows):
            db.q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Decision, n.key = row.key, n.title = row.title, "
                 "n.status = row.status, n.date = row.date, n.project = row.project", rows=part)
        db.q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Project, n.name = row.name",
             rows=[dict(id=v, name=p) for p, v in PROJECT_VID.items()])
        term_rows = [dict(id=v, name=t) for t, v in self.term_vid.items()]
        for part in chunked(term_rows):
            db.q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Term, n.name = row.name", rows=part)
        db.q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Rule, n.name = row.name, n.a = row.a, n.b = row.b",
             rows=[dict(id=v, name=n, a=RULE_PRIOR[n] * PRIOR_STRENGTH, b=(1 - RULE_PRIOR[n]) * PRIOR_STRENGTH)
                   for n, v in RULE_VID.items()])
        has = [dict(src=PROJECT_VID[r["project"]], dst=r["vid"]) for r in self.recs.values()]
        db.q("UNWIND $rows AS row CREATE (a {id: row.src})-[:HAS]->(b {id: row.dst})", rows=has)
        core = []
        for rid, (terms, _) in self.per_terms.items():
            for t in terms:
                if t in self.term_vid:
                    core.append(dict(src=self.term_vid[t], dst=self.recs[rid]["vid"]))
        for part in chunked(core):
            db.q("UNWIND $rows AS row CREATE (a {id: row.src})-[:CORE_OF]->(b {id: row.dst})", rows=part)
        for l in self.links:
            old, new = self.recs[l["old"]], self.recs[l["new"]]
            db.q("MERGE (a {id: $o})-[:REPLACED_BY {rule: $rule, state: 'candidate', w: $w}]->(b {id: $n})",
                 o=old["vid"], n=new["vid"], rule=l["rule"], w=RULE_PRIOR[l["rule"]])
        self._project_members = {}
        return dict(decisions=len(dec_rows), terms=len(term_rows), core_edges=len(core), links=len(self.links))

    def stats(self):
        out = {}
        for label in ["Decision", "Term", "Rule", "Project"]:
            out[label] = self.db.q(f"MATCH (n:{label}) RETURN count(*) AS n")[0]["n"]
        out["CORE_OF"] = self.db.q("MATCH (a:Term)-[:CORE_OF]->(b) RETURN count(*) AS n")[0]["n"]
        out["REPLACED_BY"] = self.db.q("MATCH (a:Decision)-[:REPLACED_BY]->(b) RETURN count(*) AS n")[0]["n"]
        out["HINTS"] = self.db.q("MATCH (a:Term)-[:HINTS]->(b) RETURN count(*) AS n")[0]["n"]
        return out

    # ---------- reading ----------
    def rules(self):
        rows = self.db.q("MATCH (n:Rule) RETURN n.name AS name, n.a AS a, n.b AS b")
        return {r["name"]: dict(a=r["a"], b=r["b"], trust=r["a"] / (r["a"] + r["b"])) for r in rows}

    def project_members(self, p):
        if p not in self._project_members:
            rows = self.db.q("MATCH (p:Project {id: $id})-[:HAS]->(d) RETURN d.id AS id", id=PROJECT_VID[p])
            self._project_members[p] = {r["id"] for r in rows}
        return self._project_members[p]

    def out_links(self, vid):
        return self.db.q("MATCH (a:Decision {id: $id})-[r:REPLACED_BY]->(b) "
                         "RETURN b.id AS id, r.rule AS rule, r.state AS state, r.w AS w", id=vid)

    def q_terms(self, q):
        ts = [t for t in dict.fromkeys(tokens(q)) if t in self.term_vid and t not in GENERIC]
        return sorted(ts, key=lambda t: -self.idf.get(t, 0))[:8]

    def seeds(self, q, qvec):
        """Score decisions: vector similarity + graph term structure + learned hints."""
        sims = self.vector.vecs @ qvec
        vec = {}
        for i in np.argsort(-sims)[:24]:
            rid = self.vector.chunks[i]["adr"]
            vec[rid] = max(vec.get(rid, 0.0), float(sims[i]))
        terms = self.q_terms(q)
        term_score, hint_score = defaultdict(float), defaultdict(float)
        if terms:
            rows = self.db.q("UNWIND $rows AS row MATCH (a {id: row.id})-[:CORE_OF]->(b) RETURN row.id AS src, b.id AS dst",
                             rows=[{"id": self.term_vid[t]} for t in terms])
            name = {self.term_vid[t]: t for t in terms}
            for r in rows:
                rec = self.by_vid.get(r["dst"])
                if rec:
                    term_score[rec["id"]] += self.idf.get(name[r["src"]], 0)
            for t in terms:
                for r in self.db.q("MATCH (a:Term {id: $id})-[h:HINTS]->(d) RETURN d.id AS id, h.w AS w",
                                   id=self.term_vid[t]):
                    rec = self.by_vid.get(r["id"])
                    if rec:
                        hint_score[rec["id"]] += r["w"]
        proj = detect_project(q)
        members = self.project_members(proj) if proj else None
        scores = {}
        for rid in set(vec) | set(term_score) | set(hint_score):
            s = vec.get(rid, 0.0) + 0.04 * min(term_score.get(rid, 0.0), 8.0) + 0.15 * min(hint_score.get(rid, 0.0), 4.0)
            if members is not None and self.recs[rid]["vid"] not in members:
                s -= 1.0
            scores[rid] = s
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked, terms

    def walk(self, start, rules):
        path, cur, seen = [], start, {start}
        for _ in range(MAX_HOPS):
            best = None
            for l in self.out_links(self.recs[cur]["vid"]):
                nxt = self.by_vid.get(l["id"])
                if not nxt or nxt["id"] in seen:
                    continue
                eff = self.effective(l, rules)
                if eff >= FOLLOW and (best is None or eff > best[1]):
                    best = (nxt["id"], eff, l["rule"], l["state"])
            if not best:
                break
            path.append(dict(frm=cur, to=best[0], trust=round(best[1], 2), rule=best[2], state=best[3]))
            cur = best[0]
            seen.add(cur)
        return cur, path

    @staticmethod
    def effective(link, rules):
        if link["state"] == "confirmed":
            return 1.0
        if link["state"] == "rejected":
            return 0.0
        return rules[link["rule"]]["trust"]

    def answer(self, q, qvec):
        rules = self.rules()
        ranked, terms = self.seeds(q, qvec)
        start = ranked[0][0]
        final, path = self.walk(start, rules)
        return dict(answer=final, start=start, path=path, terms=terms,
                    seeds=[(rid, round(s, 3)) for rid, s in ranked[:5]],
                    context=self.context(final, path))

    def context(self, rid, path):
        r = self.recs[rid]
        lines = [f"CURRENT DECISION: {r['title']} ({PROJECT_LABEL[r['project']]}, ADR {r['num']}, {r['date'] or 'undated'}, status: {r['status']})",
                 f"Decision: {r['decision'][:700]}"]
        if path:
            hist = " -> ".join(f"{self.recs[p['frm']]['title']} ({self.recs[p['frm']]['date'] or 'undated'}, {self.recs[p['frm']]['status']})" for p in path)
            lines.append(f"Replaces: {hist}")
        return "\n".join(lines)

    # ---------- learning ----------
    def _bump_rule(self, rule, good):
        r = self.db.q("MATCH (n:Rule {id: $id}) RETURN n.a AS a, n.b AS b", id=RULE_VID[rule])[0]
        a, b = (r["a"] + 1, r["b"]) if good else (r["a"], r["b"] + 1)
        self.db.q("MATCH (n:Rule {id: $id}) SET n.a = $a, n.b = $b", id=RULE_VID[rule], a=a, b=b)

    def _set_link(self, old, new, state):
        self.db.q("MATCH (a:Decision {id: $o})-[r:REPLACED_BY]->(b {id: $n}) SET r.state = $s",
                  o=self.recs[old]["vid"], n=self.recs[new]["vid"], s=state)

    def _reachable(self, start, goal, rules, any_link=True):
        """Path start -> goal over REPLACED_BY links (any non-rejected link), BFS."""
        frontier, prev = [start], {start: None}
        for _ in range(MAX_HOPS):
            nxt = []
            for cur in frontier:
                for l in self.out_links(self.recs[cur]["vid"]):
                    rec = self.by_vid.get(l["id"])
                    if not rec or rec["id"] in prev or l["state"] == "rejected":
                        continue
                    prev[rec["id"]] = (cur, l)
                    if rec["id"] == goal:
                        out, x = [], goal
                        while prev[x]:
                            p, link = prev[x]
                            out.append((p, x, link))
                            x = p
                        return list(reversed(out))
                    nxt.append(rec["id"])
            frontier = nxt
        return None

    def feedback(self, q, qvec, result, correct):
        """Learn from one judged answer. Returns a list of what changed."""
        changes = []
        rules = self.rules()
        got, start, path = result["answer"], result["start"], result["path"]
        if got == correct:
            for p in path:
                if p["state"] != "confirmed":
                    self._set_link(p["frm"], p["to"], "confirmed")
                    self._bump_rule(p["rule"], True)
                    changes.append(f"confirmed link {p['frm']} -> {p['to']} ({p['rule']})")
        else:
            # links the walk took that led away from the right answer
            on_path_nodes = [start] + [p["to"] for p in path]
            if correct in on_path_nodes:
                # walked past the right answer: the hop out of it was wrong
                i = on_path_nodes.index(correct)
                bad = path[i]
                self._set_link(bad["frm"], bad["to"], "rejected")
                self._bump_rule(bad["rule"], False)
                changes.append(f"rejected link {bad['frm']} -> {bad['to']} ({bad['rule']})")
            else:
                for p in path:
                    if self._reachable(p["to"], correct, rules) is None and p["state"] != "confirmed":
                        self._set_link(p["frm"], p["to"], "rejected")
                        self._bump_rule(p["rule"], False)
                        changes.append(f"rejected link {p['frm']} -> {p['to']} ({p['rule']})")
                        break
                # is the right answer reachable from where we were, through links we did not trust?
                hops = None
                for node in on_path_nodes:
                    hops = self._reachable(node, correct, rules)
                    if hops:
                        break
                if hops:
                    for frm, to, link in hops:
                        if link["state"] != "confirmed":
                            self._set_link(frm, to, "confirmed")
                            self._bump_rule(link["rule"], True)
                            changes.append(f"confirmed link {frm} -> {to} ({link['rule']})")
                else:
                    s, c = self.recs[start], self.recs[correct]
                    if (s["project"] == c["project"] and s["date"] and c["date"] and s["date"] < c["date"]
                            and self.related(start, correct)):
                        self.db.q("MERGE (a {id: $o})-[:REPLACED_BY {rule: 'learned', state: 'confirmed', w: 1.0}]->(b {id: $n})",
                                  o=s["vid"], n=c["vid"])
                        changes.append(f"learned new link {start} -> {correct}")
        # hint edges: the question's terms point at the right decision
        for t in [t for t in result["terms"] if self.idf.get(t, 0) > 1.5][:5]:
            tv, dv = self.term_vid[t], self.recs[correct]["vid"]
            rows = self.db.q("MATCH (a:Term {id: $t})-[h:HINTS]->(b {id: $d}) RETURN h.w AS w", t=tv, d=dv)
            if rows:
                self.db.q("MATCH (a:Term {id: $t})-[h:HINTS]->(b {id: $d}) SET h.w = $w", t=tv, d=dv, w=rows[0]["w"] + 1.0)
            else:
                self.db.q("MERGE (a {id: $t})-[:HINTS {w: 1.0}]->(b {id: $d})", t=tv, d=dv)
        if result["terms"]:
            changes.append(f"hinted {', '.join(result['terms'][:5])} -> {correct}")
        return changes
