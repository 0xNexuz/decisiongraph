"""DecisionGraph web API + UI."""
import json
import os
import threading
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .corpus import DATA, PROJECT_LABEL, build, chunks as make_chunks
from .embed import embed, embed_cached
from .evaluate import ROUNDS, count_tokens, load_questions, order
from .graph import DecisionGraph
from .hydra import Hydra
from .vector import VectorRAG, VectorRAGMemory

ROOT = os.path.dirname(os.path.dirname(__file__))
WEB = os.path.join(ROOT, "web")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

app = FastAPI(title="DecisionGraph")
lock = threading.Lock()
S = {}


def boot():
    t = time.time()
    recs, per, idf, links = build()
    ch = make_chunks(recs)
    cvecs = embed_cached([c["text"] for c in ch], "chunks")
    questions = load_questions()
    qv = embed_cached([q["q"] for q in questions], "questions")
    db = Hydra()
    if not db.wait(120):
        raise RuntimeError("HydraDB is not reachable")
    vec = VectorRAG(recs, ch, cvecs)
    mem = VectorRAGMemory(recs, ch, cvecs)
    graph = DecisionGraph(recs, per, idf, links, vec, db)
    loaded = graph.load()
    S.update(recs={r["id"]: r for r in recs}, graph=graph, mem=mem, questions=questions,
             qvecs={q["id"]: qv[i] for i, q in enumerate(questions)})
    trained = train()
    S.update(trained=trained, recs={r["id"]: r for r in recs}, vec=vec, mem=mem, graph=graph, questions=questions,
             qvecs={q["id"]: qv[i] for i, q in enumerate(questions)}, loaded=loaded,
             booted=time.time(), boot_s=round(time.time() - t, 1), feedback_count=0)


def train(seed=7):
    """Teach the graph with the 50 evaluation corrections (the same stream as the test),
    so the live demo shows a graph that has already learned from use."""
    g, mem = S["graph"], S["mem"]
    n = 0
    for item in order(S["questions"], seed):
        qv = S["qvecs"][item["id"]]
        g.feedback(item["q"], qv, g.answer(item["q"], qv), item["answer"])
        mem.feedback(item["q"], qv, None, item["answer"])
        n += 1
    return n


@app.on_event("startup")
def _startup():
    boot()


def card(rid):
    r = S["recs"][rid]
    return dict(id=rid, title=r["title"], project=PROJECT_LABEL[r["project"]], num=r["num"], date=r["date"],
                status=r["status"], decision=r["decision"][:600], url=r["url"])


def describe(changes):
    out = []
    for c in changes:
        c = dict(c)
        for k in ("frm", "to"):
            if c.get(k):
                r = S["recs"][c[k]]
                c[k + "_title"] = f"ADR {r['num']} {r['title']}"
        out.append(c)
    return out


class Ask(BaseModel):
    q: str


class Feedback(BaseModel):
    q: str
    correct: str


@app.get("/api/health")
def health():
    return dict(ok=bool(S), hydra=S["graph"].db.healthy() if S else False, boot_s=S.get("boot_s"))


@app.post("/api/ask")
def ask(body: Ask):
    q = body.q.strip()[:300]
    if not q:
        raise HTTPException(400, "empty question")
    qv = embed([q])[0]
    with lock:
        t0 = time.time()
        g = S["graph"].answer(q, qv)
        g_ms = (time.time() - t0) * 1000
        t1 = time.time()
        v = S["vec"].answer(q, qv)
        v_ms = (time.time() - t1) * 1000
    return dict(
        graph=dict(answer=card(g["answer"]), start=card(g["start"]), terms=g["terms"],
                   lineage=[dict(frm=card(p["frm"]), to=card(p["to"]), trust=p["trust"], rule=p["rule"], state=p["state"]) for p in g["lineage"]],
                   path=[dict(frm=card(p["frm"]), to=card(p["to"]), trust=p["trust"], rule=p["rule"], state=p["state"]) for p in g["path"]],
                   context=g["context"], tokens=count_tokens(g["context"]), ms=round(g_ms, 1)),
        vector=dict(answer=card(v["answer"]), retrieved=[card(x) for x in v["retrieved"]], context=v["context"],
                    tokens=count_tokens(v["context"]), ms=round(v_ms, 1)),
    )


@app.post("/api/feedback")
def feedback(body: Feedback):
    if body.correct not in S["recs"]:
        raise HTTPException(400, "unknown decision")
    q = body.q.strip()[:300]
    qv = embed([q])[0]
    with lock:
        res = S["graph"].answer(q, qv)
        changes = S["graph"].feedback(q, qv, res, body.correct)
        S["feedback_count"] += 1
        after = S["graph"].answer(q, qv)
    return dict(changes=describe(changes), now=card(after["answer"]),
                path=[dict(frm=card(p["frm"]), to=card(p["to"]), trust=p["trust"], rule=p["rule"]) for p in after["path"]])


@app.get("/api/decisions")
def decisions():
    rs = sorted(S["recs"].values(), key=lambda r: (r["project"], r["num"]))
    return [dict(id=r["id"], label=f"{PROJECT_LABEL[r['project']]} · ADR {r['num']} · {r['title']}", status=r["status"]) for r in rs]


SAMPLES = [
    "Are we still on Concourse for Cloud Platform pipelines?",
    "Is there one Kubernetes cluster for every environment on Cloud Platform?",
    "Do we run kOps for the Cloud Platform cluster?",
    "Where do I search my app's logs on Cloud Platform?",
    "Should Modernisation Platform lambdas be written in Python?",
    "What hostname format do GOV.UK internal services use?",
    "Should GOV.UK apps use MongoDB or DocumentDB on AWS?",
    "Does each Cloud Platform namespace get its own ingress controller?",
]


@app.get("/api/samples")
def samples():
    """New wordings that are not in the evaluation set."""
    return [dict(q=q) for q in SAMPLES]


@app.get("/api/results")
def results():
    p = os.path.join(DATA, "results.json")
    if not os.path.exists(p):
        return JSONResponse({"missing": True})
    r = json.load(open(p))
    r.pop("trace", None)
    return r


@app.get("/api/trace")
def trace():
    p = os.path.join(DATA, "results.json")
    return json.load(open(p)).get("trace", []) if os.path.exists(p) else []


@app.get("/api/graph")
def graph_info():
    with lock:
        return dict(stats=S["graph"].stats(), rules=S["graph"].rules(), loaded=S["loaded"], trained=S.get("trained", 0),
                    feedback_count=S["feedback_count"])


@app.get("/api/replay")
def replay(seed: int = 7):
    """Reset the graph and stream the test-then-learn run live, question by question."""

    def gen():
        with lock:
            g, vec, mem = S["graph"], S["vec"], S["mem"]
            g.load()
            mem.reset()
            stream = order(S["questions"], seed)
            per_round = len(stream) // ROUNDS
            seen = set()
            for i, item in enumerate(stream):
                rnd = min(i // per_round, ROUNDS - 1) + 1
                qv = S["qvecs"][item["id"]]
                row = dict(i=i + 1, n=len(stream), round=rnd, q=item["q"], want=item["answer"], first=item["topic"] not in seen)
                for sys in (vec, mem, g):
                    out = sys.answer(item["q"], qv)
                    ch = sys.feedback(item["q"], qv, out, item["answer"])
                    row[sys.name] = dict(got=out["answer"], ok=out["answer"] == item["answer"],
                                         stale=out["answer"] in item["stale"],
                                         learned=describe(ch) if sys.name == "decisiongraph" else None)
                seen.add(item["topic"])
                yield f"data: {json.dumps(row)}\n\n"
            S["feedback_count"] = 0
            yield f"data: {json.dumps({'done': True, 'rules': g.rules(), 'stats': g.stats()})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/reset")
def reset(x_admin_token: str = Header(default=""), x_blank: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403, "forbidden")
    with lock:
        S["graph"].load()
        S["mem"].reset()
        S["feedback_count"] = 0
        blank = x_blank == "1"
        S["trained"] = 0 if blank else train()
    return dict(ok=True, trained=S["trained"])


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/static", StaticFiles(directory=WEB), name="static")
