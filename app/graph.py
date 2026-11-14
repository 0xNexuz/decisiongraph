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
BLOCKED = 100.0       # path cost of a link the walker must not follow (trust below FOLLOW)
QUESTION_BASE = 500000


def cost_of(w):
    return 1.0 if w >= FOLLOW else BLOCKED


def unwrap(props):
    return {k: next(iter(v.values())) if isinstance(v, dict) and v else v for k, v in (props or {}).items()}


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
        self.q_next, self._qid = QUESTION_BASE, -1
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
        qs = [{"id": r["id"]} for r in self.db.q("MATCH (n:Question) RETURN n.id AS id")]
        for part in chunked(qs):
            self.db.q("UNWIND $rows AS row MATCH (n {id: row.id}) DETACH DELETE n", rows=part)
        self.q_next = QUESTION_BASE
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
            w = RULE_PRIOR[l["rule"]]
            db.q("MERGE (a {id: $o})-[:REPLACED_BY {rule: $rule, state: 'candidate', w: $w, cost: $c, reach: 1.0, by: -1}]->(b {id: $n})",
                 o=old["vid"], n=new["vid"], rule=l["rule"], w=w, c=cost_of(w))
        self._project_members = {}
        return dict(decisions=len(dec_rows), terms=len(term_rows), core_edges=len(core), links=len(self.links))

    def stats(self):
        out = {}
        for label in ["Decision", "Term", "Rule", "Project"]:
            out[label] = self.db.q(f"MATCH (n:{label}) RETURN count(*) AS n")[0]["n"]
        out["CORE_OF"] = self.db.q("MATCH (a:Term)-[:CORE_OF]->(b) RETURN count(*) AS n")[0]["n"]
        out["REPLACED_BY"] = self.db.q("MATCH (a:Decision)-[:REPLACED_BY]->(b) RETURN count(*) AS n")[0]["n"]
        out["HINTS"] = self.db.q("MATCH (a:Term)-[:HINTS]->(b) RETURN count(*) AS n")[0]["n"]
        out["Question"] = self.db.q("MATCH (n:Question) RETURN count(*) AS n")[0]["n"]
        out["trusted_links"] = self.db.q("MATCH (a:Decision)-[r:REPLACED_BY]->(b) WHERE r.cost < 2.0 RETURN count(*) AS n")[0]["n"]
        return out

    def replaced_count(self):
        """How many decisions the graph currently knows were replaced: one algo.MSpaths call per
        project, all pairs, over trusted links only (cost filter)."""
        known = set()
        for p in PROJECT_VID:
            keys = [r["id"] for r in self.recs.values() if r["project"] == p]
            for row in self.ms_paths(keys, keys, "outgoing", 1, limit=4000):
                known.add(row[0]["frm"])
        return len(known)

    # ---------- reading ----------
    def rules(self):
        rows = self.db.q("MATCH (n:Rule) RETURN n.name AS name, n.a AS a, n.b AS b")
        return {r["name"]: dict(a=r["a"], b=r["b"], trust=r["a"] / (r["a"] + r["b"])) for r in rows}

    def project_members(self, p):
        if p not in self._project_members:
            rows = self.db.q("MATCH (p:Project {id: $id})-[:HAS]->(d) RETURN d.id AS id", id=PROJECT_VID[p])
            self._project_members[p] = {r["id"] for r in rows}
        return self._project_members[p]

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
            where = " OR ".join(f"t.id = {self.term_vid[t]}" for t in terms)
            for r in self.db.q(f"MATCH (t:Term)-[h:HINTS]->(d) WHERE {where} RETURN d.id AS id, h.w AS w"):
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

    # ---------- traversal inside HydraDB ----------
    def ms_paths(self, sources, targets, direction, max_len, limit=400, cost="cost"):
        """algo.MSpaths over REPLACED_BY, following only trusted links (cost filter).
        Returns each path as a list of hops in walking order."""
        lit = lambda ks: ", ".join("'" + k.replace("'", "") + "'" for k in ks)
        rows = self.db.q(
            "CALL algo.MSpaths({sourceLabel: 'Decision', sourceProperty: 'key', "
            f"sourceValues: [{lit(sources)}], targetValues: [{lit(targets)}], relTypes: ['REPLACED_BY'], "
            f"relDirection: '{direction}', maxLen: {max_len}, pathCount: 1, pairwise: false, resultLimit: {limit}, "
            f"weightProp: '{cost}', costProp: '{cost}', maxCost: {max_len}}}) YIELD path RETURN path")
        out = []
        for r in rows:
            p = r.get("path") or {}
            order = [n["id"] for n in p.get("nodes", [])]
            rels = {(x["src"], x["dst"]): unwrap(x.get("properties")) for x in p.get("relationships", [])}
            hops = []
            for u, v in zip(order, order[1:]):
                old, new = (u, v) if (u, v) in rels else (v, u)
                props = rels.get((old, new), {})
                if old not in self.by_vid or new not in self.by_vid:
                    break
                hops.append(dict(frm=self.by_vid[u]["id"], to=self.by_vid[v]["id"], old=self.by_vid[old]["id"],
                                 new=self.by_vid[new]["id"], trust=round(float(props.get("w", 0)), 2),
                                 rule=props.get("rule", "?"), state=props.get("state", "?"), by=props.get("by", -1)))
            if hops:
                out.append(hops)
        return out

    def question_texts(self, ids):
        ids = [i for i in set(ids) if isinstance(i, int) and i >= QUESTION_BASE]
        if not ids:
            return {}
        where = " OR ".join(f"n.id = {i}" for i in ids)
        return {r["id"]: r["text"] for r in self.db.q(f"MATCH (n:Question) WHERE {where} RETURN n.id AS id, n.text AS text")}

    def same_project(self, rid):
        p = self.recs[rid]["project"]
        return [r["id"] for r in self.recs.values() if r["project"] == p and r["id"] != rid]

    @staticmethod
    def _best(paths):
        def key(h):
            last = h[-1]["to"]
            return (len(h), min(x["trust"] for x in h))
        return max(paths, key=key) if paths else []

    def walk(self, start, rules=None):
        """From the decision a question is about, walk trusted 'replaced by' links to the decision
        that still stands. One algo.MSpaths call evaluates every reachable decision at once."""
        paths = self.ms_paths([start], self.same_project(start), "outgoing", MAX_HOPS)
        best = self._best(paths)
        path = [dict(frm=h["old"], to=h["new"], trust=h["trust"], rule=h["rule"], state=h["state"], by=h["by"]) for h in best]
        return (path[-1]["to"] if path else start), path

    def history(self, rid, rules=None, limit=3):
        """Older decisions this one replaced, walking trusted links backwards inside HydraDB."""
        paths = self.ms_paths([rid], self.same_project(rid), "incoming", limit)
        best = self._best(paths)
        hops = [dict(frm=h["old"], to=h["new"], trust=h["trust"], rule=h["rule"], state=h["state"], by=h["by"]) for h in best]
        return list(reversed(hops))

    def answer(self, q, qvec):
        rules = self.rules()
        ranked, terms = self.seeds(q, qvec)
        start = ranked[0][0]
        final, path = self.walk(start)
        lineage = self.history(final)
        return dict(answer=final, start=start, path=path, lineage=lineage, terms=terms,
                    seeds=[(rid, round(s, 3)) for rid, s in ranked[:5]],
                    context=self.context(final, lineage))

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
        # the rule's new trust flows to every link it proposed that nobody has judged yet
        t = a / (a + b)
        self.db.q("MATCH (x:Decision)-[r:REPLACED_BY]->(y) WHERE r.rule = $rule AND r.state = 'candidate' "
                  "SET r.w = $w, r.cost = $c", rule=rule, w=t, c=cost_of(t))

    def _set_link(self, old, new, state):
        w = 1.0 if state == "confirmed" else 0.0
        self.db.q("MATCH (a:Decision {id: $o})-[r:REPLACED_BY]->(b {id: $n}) SET r.state = $s, r.w = $w, r.cost = $c, r.reach = $rc, r.by = $q",
                  o=self.recs[old]["vid"], n=self.recs[new]["vid"], s=state, w=w, c=cost_of(w),
                  rc=1.0 if state == "confirmed" else BLOCKED, q=self._qid)

    def _reachable(self, start, goal, rules=None):
        """Path start -> goal over any link not rejected (trusted or not), found by algo.MSpaths
        with the 'reach' cost. Returns [(old, new, link)] or None."""
        if start == goal:
            return None
        paths = self.ms_paths([start], [goal], "outgoing", MAX_HOPS, cost="reach")
        if not paths:
            return None
        return [(h["old"], h["new"], dict(rule=h["rule"], state=h["state"])) for h in paths[0]]

    def feedback(self, q, qvec, result, correct):
        """Learn from one judged answer. Returns a list of what changed.
        The question itself is stored as a (:Question)-[:RESOLVED_TO]->(:Decision) node, and every
        link this correction confirms or adds records it in r.by, so each lesson is traceable."""
        changes = []
        self._qid = self.q_next
        self.q_next += 1
        self.db.q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Question, n.text = row.text, n.was_right = row.ok",
                  rows=[dict(id=self._qid, text=q[:300], ok=result["answer"] == correct)])
        self.db.q("MERGE (a {id: $q})-[:RESOLVED_TO]->(b {id: $d})", q=self._qid, d=self.recs[correct]["vid"])
        rules = self.rules()
        got, start, path = result["answer"], result["start"], result["path"]
        if got == correct:
            for p in path:
                if p["state"] != "confirmed":
                    self._set_link(p["frm"], p["to"], "confirmed")
                    self._bump_rule(p["rule"], True)
                    changes.append(dict(kind="confirmed", frm=p["frm"], to=p["to"], rule=p["rule"]))
        else:
            # links the walk took that led away from the right answer
            on_path_nodes = [start] + [p["to"] for p in path]
            if correct in on_path_nodes:
                # walked past the right answer: the hop out of it was wrong
                i = on_path_nodes.index(correct)
                bad = path[i]
                self._set_link(bad["frm"], bad["to"], "rejected")
                self._bump_rule(bad["rule"], False)
                changes.append(dict(kind="rejected", frm=bad["frm"], to=bad["to"], rule=bad["rule"]))
            else:
                for p in path:
                    if self._reachable(p["to"], correct, rules) is None and p["state"] != "confirmed":
                        self._set_link(p["frm"], p["to"], "rejected")
                        self._bump_rule(p["rule"], False)
                        changes.append(dict(kind="rejected", frm=p["frm"], to=p["to"], rule=p["rule"]))
                        break
                # is the right answer reachable from where we were, through links we did not trust?
                hops = None
                for node in on_path_nodes:
                    hops = self._reachable(node, correct, rules)
                    if hops:
                        break
                if hops and not all(self.related(frm, to) or l["rule"] in ("status-link", "replace-term", "learned") for frm, to, l in hops):
                    hops = None  # a path through unrelated decisions is not evidence
                if hops:
                    for frm, to, link in hops:
                        if link["state"] != "confirmed":
                            self._set_link(frm, to, "confirmed")
                            self._bump_rule(link["rule"], True)
                            changes.append(dict(kind="confirmed", frm=frm, to=to, rule=link["rule"]))
                else:
                    s, c = self.recs[start], self.recs[correct]
                    if (s["project"] == c["project"] and s["date"] and c["date"] and s["date"] < c["date"]
                            and self.related(start, correct)):
                        self.db.q("MERGE (a {id: $o})-[:REPLACED_BY {rule: 'learned', state: 'confirmed', w: 1.0, cost: 1.0, reach: 1.0, by: $q}]->(b {id: $n})",
                                  o=s["vid"], n=c["vid"], q=self._qid)
                        changes.append(dict(kind="learned", frm=start, to=correct, rule="learned"))
        # hint edges: the question's terms point at the right decision
        hint_terms = [t for t in result["terms"] if self.idf.get(t, 0) > 1.5][:5]
        for t in hint_terms:
            tv, dv = self.term_vid[t], self.recs[correct]["vid"]
            rows = self.db.q("MATCH (a:Term {id: $t})-[h:HINTS]->(b {id: $d}) RETURN h.w AS w", t=tv, d=dv)
            if rows:
                self.db.q("MATCH (a:Term {id: $t})-[h:HINTS]->(b {id: $d}) SET h.w = $w", t=tv, d=dv, w=rows[0]["w"] + 1.0)
            else:
                self.db.q("MERGE (a {id: $t})-[:HINTS {w: 1.0}]->(b {id: $d})", t=tv, d=dv)
        if hint_terms:
            changes.append(dict(kind="hint", terms=hint_terms, to=correct))
        return changes
