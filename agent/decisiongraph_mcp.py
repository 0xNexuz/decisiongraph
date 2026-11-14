"""DecisionGraph as an MCP server: gives coding agents (Claude Code, Cursor, ...) the decision
that still stands, plus its history, instead of a pile of search results.

Run:  DECISIONGRAPH_URL=https://your-app.onrender.com python agent/decisiongraph_mcp.py
Add to Claude Code:  claude mcp add decisiongraph -- python /path/to/agent/decisiongraph_mcp.py
"""
import json
import os
import urllib.error
import urllib.request

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

BASE = os.environ.get("DECISIONGRAPH_URL", "http://127.0.0.1:10000").rstrip("/")
mcp = MCPServer("decisiongraph")


class DecisionGraphError(ToolError):
    """Shown to the agent as the tool's error message."""


def _call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            detail = ""
        if detail == "unknown decision":
            raise DecisionGraphError("That decision id doesn't exist. Use find_decisions to get the right id.") from None
        raise DecisionGraphError(f"DecisionGraph answered {e.code}: {detail or e.reason}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise DecisionGraphError(f"Can't reach DecisionGraph at {BASE} ({e}). A free host may be waking up; try again in a minute.") from None


@mcp.tool()
def current_decision(question: str) -> str:
    """Return the team's current architecture decision for a question, with the older decisions
    it replaced. Use this before acting on any 'what did we decide about X' question. If no
    decision on record answers it, says so instead of guessing."""
    g = _call("/api/ask", {"q": question})["graph"]
    a = g["answer"]
    if not a:
        return "No decision on record answers this question. Don't assume one exists; ask the team."
    return f"{g['context']}\nSource: {a['url']}\nDecision id: {a['id']}"


@mcp.tool()
def find_decisions(search: str = "") -> str:
    """List decision records whose title contains the search words (all of them if empty), with
    the ids that correct_decision expects."""
    words = search.lower().split()
    rows = [d for d in _call("/api/decisions") if all(w in d["label"].lower() for w in words)]
    if not rows:
        return "No decision titles match that. Try fewer words."
    return "\n".join(f"{d['id']}  {d['label']}" for d in rows[:40]) + ("" if len(rows) <= 40 else f"\n... and {len(rows) - 40} more")


@mcp.tool()
def correct_decision(question: str, correct_decision_id: str) -> str:
    """Tell DecisionGraph which decision was actually right for a question (an id like
    'cloud-platform:48', from current_decision or find_decisions). The graph learns from it, so
    later questions on the same topic, in other words too, are more likely to get it right."""
    r = _call("/api/feedback", {"q": question, "correct": correct_decision_id})
    lines = [f"Learned {len(r['changes'])} change(s)."]
    lines += [f"- {c.get('kind')}: {c.get('frm_title') or c.get('frm', '')} -> {c.get('to_title') or c.get('to', '')}".rstrip(" ->")
              for c in r["changes"] if c.get("kind") != "hint"]
    now = r.get("now")
    lines.append(f"Now answers: ADR {now['num']} {now['title']}" if now else "Now answers: no decision on record.")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
