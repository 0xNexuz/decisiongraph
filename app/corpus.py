"""Parse public Architecture Decision Records into clean records, chunks,
terms and candidate links between decisions.

Sources (all MIT licensed):
  ministryofjustice/cloud-platform, ministryofjustice/modernisation-platform,
  alphagov/govuk-aws
"""
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

PROJECT_NAMES = {
    "cloud-platform": ["cloud platform", "container platform", "cp "],
    "modernisation-platform": ["modernisation platform", "mp "],
    "govuk-aws": ["gov.uk", "govuk"],
}
PROJECT_LABEL = {
    "cloud-platform": "MoJ Cloud Platform",
    "modernisation-platform": "MoJ Modernisation Platform",
    "govuk-aws": "GOV.UK on AWS",
}
PROJECT_PREFIX = {"cloud-platform": 1000, "modernisation-platform": 2000, "govuk-aws": 3000}

STOP = set("""a an the and or of to in on for with by is are was were be been being it its this that these those
we our us you your they them their i as at from into over than then so such not no can will would should could may might
must do does did done have has had having which who whom what when where why how all any each other some more most
also only very just there here about after before because if while within without between up down out off use used using
new now still per via etc eg ie e g one two three both however would like need needs needed make made get got able across
since being way ways team teams time first well much many see""".split())

STATUS_WORDS = {
    "superseded": "superseded", "rejected": "rejected", "proposed": "proposed",
    "accepted": "accepted", "approved": "accepted", "pending": "proposed",
    "incomplete": "proposed", "partly superseded": "partly-superseded",
}
REPLACE_VERBS = re.compile(r"\b(replac\w*|switch\w*|migrat\w*|instead of|in favour of|in favor of|retire\w*|decommission\w*|move (?:on )?from|moving from|deprecat\w*|supersed\w*|no longer)\b", re.I)


def norm_status(raw):
    s = raw.lower()
    if "partly superseded" in s:
        return "partly-superseded"
    for k in ["superseded", "rejected", "proposed", "pending", "incomplete", "approved", "accepted"]:
        if k in s:
            return STATUS_WORDS[k]
    return "unknown"


def parse_date(s):
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def section(text, names):
    for n in names:
        m = re.search(r"^##+\s*" + n + r"\s*\n+(.+?)(?=\n##?\s|\Z)", text, re.S | re.M | re.I)
        if m:
            return m.group(1).strip()
    return ""


def clean(md):
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)
    md = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", md)
    md = re.sub(r"^\[[^\]]+\]:.*$", " ", md, flags=re.M)
    md = re.sub(r"[`*_>#|]", " ", md)
    md = re.sub(r"[✅❌🤔⌛️⌛]", " ", md)
    return re.sub(r"\s+", " ", md).strip()


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9][a-z0-9.+\-]*[a-z0-9]|[a-z]", text.lower())
            if t not in STOP and len(t) > 1 and not t.isdigit()]


def load_records():
    raw = json.load(open(os.path.join(DATA, "adrs_raw.json")))
    recs = []
    for a in raw:
        t = a["text"]
        if not a["status"].strip():
            continue  # a few files are empty stubs
        decision = section(t, ["Decision", "What’s proposed", "What's proposed", "Proposal", "Decision Outcome"])
        context = section(t, ["Context", "Reasoning", "The existing decision", "Introduction"])
        consequences = section(t, ["Consequences"])
        if not decision:
            decision = context[:800]
        r = dict(
            id=a["id"], vid=PROJECT_PREFIX[a["project"]] + a["num"], project=a["project"], num=a["num"],
            title=re.sub(r"^\d+\s+", "", a["title"]).strip(), date=parse_date(a["date"]),
            status=norm_status(a["status"]), status_raw=clean(a["status"])[:240], url=a["url"],
            decision=clean(decision), context=clean(context), consequences=clean(consequences),
            raw=t, status_section=a["status"],
        )
        r["full"] = clean(t)
        recs.append(r)
    # duplicate number 29/30 in cloud-platform share a title; keep both but disambiguate
    return recs


def chunks(recs, words=160, overlap=40):
    """Standard RAG chunks: title-prefixed windows over the cleaned ADR text."""
    out = []
    for r in recs:
        w = r["full"].split()
        step = words - overlap
        for i in range(0, max(len(w), 1), step):
            piece = " ".join(w[i:i + words])
            if not piece:
                break
            out.append(dict(adr=r["id"], text=f"{r['title']} ({PROJECT_LABEL[r['project']]}). {piece}"))
            if i + words >= len(w):
                break
    return out


def term_index(recs):
    """Terms per decision with idf. Title + decision terms are 'core' terms."""
    df = Counter()
    per = {}
    for r in recs:
        core = tokens(r["title"]) + tokens(r["decision"])
        rest = tokens(r["context"]) + tokens(r["consequences"])
        bag = Counter(core * 3 + rest)
        per[r["id"]] = (set(core), bag)
        df.update(set(bag))
    n = len(recs)
    idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
    return per, idf


ADR_REF = re.compile(r"(?:ADR[- #]*|Decision\s+|ADR\s+#)0*(\d{1,4})\b|\(\.?/?0*(\d{1,4})-[A-Za-z][^)]*\.md\)|\b0*(\d{2,4})-[A-Za-z][\w-]*\.md\b", re.I)


def candidate_links(recs, per_terms, idf):
    """Candidate 'newer decision replaces older decision' links, each tagged
    with the rule that proposed it. Rules start with different trust levels
    and the graph learns how much to trust each rule from feedback."""
    by_proj_num = {(r["project"], r["num"]): r for r in recs}
    links = {}

    def add(new, old, rule, evidence):
        if new["id"] == old["id"]:
            return
        key = (new["id"], old["id"])
        if key not in links or RULE_PRIOR[rule] > RULE_PRIOR[links[key]["rule"]]:
            links[key] = dict(new=new["id"], old=old["id"], rule=rule, evidence=evidence[:200])

    for r in recs:
        # rule 1: the status section of an ADR names its replacement
        st = r["status_section"]
        if re.search(r"supersed|rejected|amended by|see \[", st, re.I):
            for m in re.finditer(r"\]\((?:[^)\s]*/)?\.?/?0*(\d{1,4})-[^)/]*\.md\)", st):
                tgt = by_proj_num.get((r["project"], int(m.group(1))))
                if tgt:
                    add(tgt, r, "status-link", clean(st))
        # rule 1b: an ADR says it amends / supersedes an older one in its status
        if re.search(r"\bamends\b|\bsupersedes\b", st, re.I):
            for m in re.finditer(r"\]\((?:[^)\s]*/)?\.?/?0*(\d{1,4})-[^)/]*\.md\)", st):
                tgt = by_proj_num.get((r["project"], int(m.group(1))))
                if tgt:
                    add(r, tgt, "status-link", clean(st))
        # rule 2: a newer ADR's body cites an older ADR by number
        body = r["raw"].split("## Status", 1)[-1]
        for m in ADR_REF.finditer(body):
            num = int(next(g for g in m.groups() if g))
            tgt = by_proj_num.get((r["project"], num))
            if tgt and tgt["id"] != r["id"] and (tgt["num"] < r["num"]):
                around = body[max(0, m.start() - 160): m.end() + 160]
                rule = "body-ref-replace" if REPLACE_VERBS.search(around) or "superseded by" in around.lower() else "body-ref"
                add(r, tgt, rule, clean(around))
    # rule 3: newer ADR says replace/switch/migrate near a distinctive term that is
    # core to an older ADR in the same project
    for r in recs:
        text = r["context"] + " " + r["decision"] + " " + r["consequences"]
        for m in REPLACE_VERBS.finditer(text):
            window = text[max(0, m.start() - 200): m.end() + 200]
            wt = set(tokens(window))
            for o in recs:
                if o["project"] != r["project"] or o["id"] == r["id"] or not o["date"] or not r["date"] or o["date"] >= r["date"]:
                    continue
                core = tokens(o["title"]) + tokens(o["decision"])[:25]
                title_terms = sorted({t for t in core if idf.get(t, 0) > 2.6}, key=lambda t: (-idf[t], t))[:4]
                hit = [t for t in title_terms if t in wt]
                if hit:
                    add(r, o, "replace-term", f"{' '.join(hit)}: {window}")
    return list(links.values())


RULE_PRIOR = {"status-link": 0.9, "body-ref-replace": 0.45, "replace-term": 0.35, "body-ref": 0.2, "learned": 1.0}


def build():
    recs = load_records()
    per, idf = term_index(recs)
    links = candidate_links(recs, per, idf)
    return recs, per, idf, links


if __name__ == "__main__":
    recs, per, idf, links = build()
    print(len(recs), "records,", len(links), "candidate links")
    c = Counter(l["rule"] for l in links)
    print(c)
    for l in sorted(links, key=lambda x: (x["rule"], x["new"])):
        print(f"{l['rule']:17} {l['new']:28} -> {l['old']:28} | {l['evidence'][:90]}")
