"""DecisionGraph as an MCP server: gives coding agents (Claude Code, Cursor, ...) the decision
that still stands, plus its history, instead of a pile of search results.

Run:  DECISIONGRAPH_URL=https://your-app.onrender.com python agent/decisiongraph_mcp.py
Add to Claude Code:  claude mcp add decisiongraph -- python /path/to/agent/decisiongraph_mcp.py
"""
import json
import os
import urllib.request

from mcp.server.mcpserver import MCPServer

BASE = os.environ.get("DECISIONGRAPH_URL", "http://127.0.0.1:10000").rstrip("/")
mcp = MCPServer("decisiongraph")


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


@mcp.tool()
def current_decision(question: str) -> str:
    """Return the team's current architecture decision for a question, with the older decisions
    it replaced. Use this before acting on any 'what did we decide about X' question."""
    g = _post("/api/ask", {"q": question})["graph"]
    a = g["answer"]
    return f"{g['context']}\nSource: {a['url']}\nDecision id: {a['id']}"


@mcp.tool()
def correct_decision(question: str, correct_decision_id: str) -> str:
    """Tell DecisionGraph which decision was actually right for a question (an id like
    'cloud-platform:48'). The graph learns from it, so the next agent gets it right."""
    r = _post("/api/feedback", {"q": question, "correct": correct_decision_id})
    return f"Learned {len(r['changes'])} change(s). Now answers: ADR {r['now']['num']} {r['now']['title']}"


if __name__ == "__main__":
    mcp.run()
