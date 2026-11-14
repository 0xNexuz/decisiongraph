const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;

const RULES = {
  "status-link": ["Named in a status line", "The old record says it was superseded, rejected or amended, and links the new one"],
  "body-ref-replace": ["Cited with replace wording", "A newer record cites the older one near words like replace, migrate or supersede"],
  "replace-term": ["Replaces what the old one chose", "A newer record says it replaces, switches from or retires the thing an older record chose"],
  "body-ref": ["Cited by a newer record", "A newer record mentions the older one by number"],
  learned: ["Learned from a correction", "Added by the graph when a correction showed a link was missing"],
};
const STATUS = { accepted: "Accepted", proposed: "Proposed", superseded: "Superseded", rejected: "Rejected", "partly-superseded": "Partly superseded", unknown: "No status" };

function onVisible(el, fn) {
  if (!el) return;
  let done = false;
  const run = () => { if (!done) { done = true; fn(); } };
  const r = el.getBoundingClientRect();
  if (r.top < innerHeight && r.bottom > 0) return setTimeout(run, 30);
  if (!("IntersectionObserver" in window)) return run();
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); run(); } }, { threshold: 0.15 });
  io.observe(el);
}

let lastQ = "";
let lastAnswer = null;

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

function slip(c, { stamp, past } = {}) {
  const stampHtml = stamp ? `<span class="stamp ${stamp[0]}">${esc(stamp[1])}</span>` : "";
  return `<article class="slip ${esc(c.status)}${past ? " past" : ""}">
    ${stampHtml}
    <p class="where">${esc(c.project)}, ADR ${esc(c.num)}${c.date ? `, ${esc(c.date)}` : ""}, ${esc(STATUS[c.status] || c.status)}</p>
    <h3 class="title">${esc(c.title)}</h3>
    <p class="text">${esc(c.decision)}</p>
    <p class="src"><a href="${esc(c.url)}" target="_blank" rel="noopener">Read the record</a></p>
  </article>`;
}

const NONE = (who) => `<article class="slip none"><p class="where">${esc(who)}</p><h3 class="title">No decision on record</h3><p class="text">Nothing in these decision records answers this question, so nothing is handed to the agent.</p></article>`;

function finalStamp(n) {
  if (n.status === "rejected") return ["replaced", "Rejected"];
  if (n.status === "superseded" || n.status === "partly-superseded") return ["replaced", "Superseded"];
  return ["current", "Current"];
}

function renderResult(res) {
  $("#result").hidden = false;
  const g = res.graph, v = res.vector;
  lastAnswer = g.answer;
  $("#pick").hidden = true;
  $("#learned").innerHTML = "";
  $("#yes").disabled = $("#no").disabled = false;
  if (!v.answer) {
    $("#vector-slip").innerHTML = NONE("Vector search");
    $("#vector-meter").innerHTML = "";
  }
  if (!g.answer) {
    $("#lineage").innerHTML = `<li>${NONE("DecisionGraph")}</li>`;
    $("#graph-meter").innerHTML = "The closest record was not close enough, so DecisionGraph says so instead of guessing.";
    $("#yes").disabled = true;
    if (!v.answer) return;
  }
  const chain = g.lineage.length ? g.lineage : g.path;
  const nodes = chain.length ? [chain[0].frm, ...chain.map((p) => p.to)] : [g.answer];
  const pastIds = new Set(nodes.slice(0, -1).map((n) => n.id));

  if (v.answer) {
    const wasReplaced = pastIds.has(v.answer.id) || v.replaced;
    let vStamp = ["status", STATUS[v.answer.status] || v.answer.status];
    if (wasReplaced) vStamp = ["replaced", "Replaced"];
    $("#vector-slip").innerHTML = slip(v.answer, { stamp: vStamp }) +
      (wasReplaced && g.answer && g.answer.id !== v.answer.id ? `<p class="stale-note">The graph knows this decision was later replaced. DecisionGraph answers with ADR ${esc(g.answer.num)}, ${esc(g.answer.title)}.</p>` : "");
    $("#vector-meter").innerHTML = `Hands the agent <b>${v.tokens}</b> tokens (top 3 passages).`;
  }
  if (!g.answer) return;

  let html = "";
  nodes.forEach((n, i) => {
    const last = i === nodes.length - 1;
    const stamp = last ? finalStamp(n) : ["replaced", "Replaced"];
    html += `<li class="enter" style="--i:${i * 2}">${slip(n, { stamp, past: !last }).replace('class="slip', `style="--i:${i}" class="slip${last ? "" : " tipped"}`)}</li>`;
    if (!last) {
      const p = chain[i];
      const [label, help] = RULES[p.rule] || [p.rule, ""];
      const how = p.learned_from ? `${p.rule === "learned" ? "learned" : "confirmed"} from the correction “${p.learned_from}”`
        : p.state === "confirmed" ? "confirmed by a correction" : label.toLowerCase();
      html += `<li class="hop enter" style="--i:${i * 2 + 1}" title="${esc(help)}"><span class="arrow">↓ replaced by</span><span>${esc(how)}</span><span class="trust">trust ${pct(p.trust)}</span></li>`;
    }
  });
  $("#lineage").innerHTML = html;
  const hops = g.path.length;
  const walked = hops ? `Started at ADR ${g.start.num} and walked ${hops} trusted ${hops === 1 ? "link" : "links"} with one algo.MSpaths call` : "The question landed on this decision directly, so there was nothing to walk";
  const hist = g.lineage.length ? ` The history comes from walking ${g.lineage.length} trusted ${g.lineage.length === 1 ? "link" : "links"} backwards.` : "";
  $("#graph-meter").innerHTML = `Hands the agent <b>${g.tokens}</b> tokens. ${walked}.${hist} Answered in ${g.ms} ms.${g.replaced ? " Warning: this record is marked replaced, but the graph does not trust any replacement yet." : ""}`;
}

async function ask(q) {
  lastQ = q;
  const btn = $("#ask button");
  btn.disabled = true; btn.textContent = "Asking";
  try {
    renderResult(await api("/api/ask", { q }));
    $("#result").scrollIntoView({ block: "start" });
  } catch (e) {
    $("#result").hidden = false;
    $("#lineage").innerHTML = `<li class="error">Couldn't answer: ${esc(e.message)}. The server may be waking up; try again in a minute.</li>`;
  } finally {
    btn.disabled = false; btn.textContent = "Ask";
  }
}

function describeChange(c) {
  switch (c.kind) {
    case "confirmed": return `Confirmed that <b>${esc(c.frm_title)}</b> was replaced by <b>${esc(c.to_title)}</b>.`;
    case "rejected": return `Stopped trusting the link from <b>${esc(c.frm_title)}</b> to <b>${esc(c.to_title)}</b>.`;
    case "learned": return `Added a new link: <b>${esc(c.to_title)}</b> replaces <b>${esc(c.frm_title)}</b>.`;
    case "proposed": return `Proposed a new link: <b>${esc(c.to_title)}</b> replaces <b>${esc(c.frm_title)}</b>. It is followed once a second correction needs it too.`;
    case "hint": return `Pointed the words “${esc(c.terms.join(", "))}” at the topic that starts with <b>${esc(c.to_title)}</b>, and stored this question in the graph.`;
    default: return esc(JSON.stringify(c));
  }
}

async function teach(correct) {
  $("#yes").disabled = $("#no").disabled = true;
  $("#learned").innerHTML = "<p>Teaching the graph…</p>";
  try {
    const r = await api("/api/feedback", { q: lastQ, correct });
    const items = r.changes.length ? r.changes.map((c, i) => `<li style="--i:${i}">${describeChange(c)}</li>`).join("") : "<li>Nothing needed changing.</li>";
    $("#learned").innerHTML = `<p>What the graph learned</p><ul>${items}</ul><p class="now">Asked again now, it answers: ADR ${esc(r.now.num)}, ${esc(r.now.title)}.</p>`;
    loadGraph();
  } catch (e) {
    $("#learned").innerHTML = `<p class="error">Couldn't save that: ${esc(e.message)}</p>`;
    $("#yes").disabled = $("#no").disabled = false;
  }
}

/* ---------- learning chart ---------- */
const SYS = [
  ["vector", "Vector search", "#c9d4ec", true],
  ["vector+memory", "Vector search + memory", "#e8cdbe", false],
  ["decisiongraph-frozen", "DecisionGraph, learning off", "#d9b44a", true],
  ["decisiongraph", "DecisionGraph, learning", "#7fd0d8", false],
];

function drawChart(mean) {
  const W = 720, H = 340, L = 52, R = 20, T = 18, B = 44;
  const x = (i) => L + (i * (W - L - R)) / 4;
  const y = (v) => T + (1 - v) * (H - T - B);
  let s = `<defs><clipPath id="sweep"><rect class="sweep" x="0" y="0" height="${H}" width="0"/></clipPath></defs>`;
  for (let v = 0; v <= 1.0001; v += 0.25) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 10}" y="${y(v) + 4}" text-anchor="end">${pct(v)}</text>`;
  for (let i = 0; i < 5; i++) s += `<text x="${x(i)}" y="${H - 16}" text-anchor="middle">Round ${i + 1}</text>`;
  const g = mean.decisiongraph.rounds;
  const band = g.map((r, i) => `${x(i)},${y(r.accuracy_max)}`).join(" ") + " " + g.slice().reverse().map((r, i) => `${x(4 - i)},${y(r.accuracy_min)}`).join(" ");
  s += `<g clip-path="url(#sweep)"><polygon class="band" points="${band}"/>`;
  for (const [key, label, color, dash] of SYS) {
    const rs = mean[key]?.rounds; if (!rs) continue;
    const pts = rs.map((r, i) => `${x(i)},${y(r.accuracy)}`).join(" ");
    s += `<g class="series"><title>${esc(label)}</title><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="${key === "decisiongraph" ? 4.5 : 2.5}" ${dash ? 'stroke-dasharray="7 6"' : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
    rs.forEach((r, i) => { s += `<circle cx="${x(i)}" cy="${y(r.accuracy)}" r="${key === "decisiongraph" ? 5 : 3.5}" fill="${color}"><title>${esc(label)}, round ${i + 1}: ${pct(r.accuracy)}</title></circle>`; });
    s += `</g>`;
  }
  const last = g[4];
  s += `<text x="${x(4) - 10}" y="${y(last.accuracy) - 14}" text-anchor="end" style="fill:#fff;font-weight:800;font-size:17px">${pct(last.accuracy)}</text></g>`;
  $("#chart").innerHTML = s;
  $("#legend").innerHTML = SYS.map(([k, label, c, dash]) => `<span class="${dash ? "dash" : ""}" style="--c:${c}">${esc(label)}</span>`).join("") +
    `<span style="--c:rgba(127,208,216,.4)">Range over five question orders</span>`;
  const rect = $("#chart .sweep");
  const sweep = () => { rect.style.transition = "width 1.8s cubic-bezier(.2,.8,.2,1)"; rect.setAttribute("width", W); };
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) rect.setAttribute("width", W);
  else onVisible($("#chart"), sweep);
}

function drawFacts(mean) {
  const g = mean.decisiongraph.overall, v = mean.vector.overall, m = mean["vector+memory"].overall;
  $("#facts").innerHTML = `
    <div class="fact"><h3>Answers that were outdated or rejected</h3><div class="pair"><span class="us">${pct(g.stale_rate)}</span><span class="them">vs ${pct(v.stale_rate)} for vector search</span></div><p>Average over all 50 questions and five question orders.</p></div>
    <div class="fact"><h3>Tokens handed to the agent</h3><div class="pair"><span class="us">${Math.round(g.tokens)}</span><span class="them">vs ${Math.round(v.tokens)} for vector search</span></div><p>The current decision and its history, not three loose passages.</p></div>
    <div class="fact"><h3>Right answer when a topic comes back in new words</h3><div class="pair"><span class="us">${pct(g.repeat_acc)}</span><span class="them">vs ${pct(m.repeat_acc)} with vector memory</span></div><p>A correction made once carries over to other wordings of the question.</p></div>`;
}

async function loadResults() {
  try {
    const r = await api("/api/results");
    if (r.missing) return;
    drawChart(r.mean); drawFacts(r.mean);
  } catch (e) { $("#facts").innerHTML = `<p class="error">Couldn't load results: ${esc(e.message)}</p>`; }
}

/* ---------- inside the graph ---------- */
async function loadGraph() {
  try {
    const r = await api("/api/graph");
    const order = ["status-link", "body-ref-replace", "replace-term", "body-ref", "learned"];
    $("#trust").innerHTML = order.filter((k) => r.rules[k]).map((k) => {
      const t = r.rules[k].trust, [label, help] = RULES[k];
      return `<div class="tr${t >= 0.5 ? " trusted" : ""}" data-rule="${k}" style="--w:${(t * 100).toFixed(1)}%;--d:${order.indexOf(k) * 120}ms"><div class="name">${esc(label)}<small>${esc(help)}</small></div>
        <div class="track" title="Links are followed at 50% trust or more"><div class="fill"></div><div class="line"></div></div>
        <div class="pct">${pct(t)}</div></div>`;
    }).join("");
    document.querySelectorAll(".tr[data-rule]").forEach(markStart);
    onVisible($("#trust"), () => $("#trust").classList.add("on"));
    const s = r.stats;
    $("#counts").textContent = `${r.trained ? `This graph has learned from ${r.trained} corrections. ` : ""}HydraDB holds ${s.Decision} decisions, ${s.Term} terms, ${s.CORE_OF} term-to-decision links, ${s.REPLACED_BY} “replaced by” links (${s.trusted_links} trusted), ${s.HINTS} learned hints and ${s.Question} corrected questions. Right now it knows that ${r.replaced_known} decisions were replaced, found with one all-pairs path search per project. The solid line marks 50%: below it, links from that rule are not followed unless a correction has confirmed them. The dashed tick marks where each rule started, before any correction. ${r.feedback_count ? `Visitors have corrected it ${r.feedback_count} times since it last started.` : ""}`;
  } catch (e) { $("#counts").textContent = "Couldn't reach the graph."; }
}

/* ---------- what the corrections changed ---------- */
const STATE = {
  confirmed: ["Confirmed", "now followed"],
  pending: ["Proposed", "waiting for a second correction"],
  rejected: ["Rejected", "never followed"],
};
let ledgerAll = false;
async function loadLearned() {
  try {
    const r = await api("/api/learned");
    const s = r.summary;
    $("#tally").innerHTML = [
      [s.confirmed, s.confirmed === 1 ? "link confirmed" : "links confirmed", "confirmed"],
      [s.rejected, s.rejected === 1 ? "link rejected" : "links rejected", "rejected"],
      [s.pending, s.pending === 1 ? "missing link proposed" : "missing links proposed", "pending"],
      [s.questions, s.questions === 1 ? "question remembered" : "questions remembered", "q"],
    ].map(([n, l, k]) => `<div class="t ${k}"><b>${n}</b><span>${esc(l)}</span></div>`).join("");
    const rows = r.links.map((l) => {
      const [label] = RULES[l.rule] || [l.rule];
      const before = l.before == null ? "No link at all" : `${label}: ${pct(l.before)} trust, ${l.before >= 0.5 ? "followed" : "not followed"}`;
      const [st, what] = STATE[l.state];
      const q = l.question ? `<p class="by">Changed by the question “${esc(l.question)}”</p>` : "";
      return `<li class="lrow ${l.state}">
        <div class="slip from"><small>ADR ${l.frm_num}</small>${esc(l.frm_title)}</div>
        <div class="link"><div class="was"><span>Before</span><i></i><em>${esc(before)}</em></div><div class="now"><span>Now</span><i></i><em><b>${st}</b>, ${what}</em></div></div>
        <div class="slip to"><small>ADR ${l.to_num}</small>${esc(l.to_title)}</div>${q}</li>`;
    });
    const cut = 6;
    $("#ledger").innerHTML = (ledgerAll ? rows : rows.slice(0, cut)).join("") || `<li class="empty">No corrections yet. Ask a question above and correct it.</li>`;
    const more = $("#ledger-more");
    more.hidden = rows.length <= cut || ledgerAll;
    more.textContent = `Show all ${rows.length} changed links`;
    window.__ruleStart = r.rules;
    document.querySelectorAll(".tr[data-rule]").forEach(markStart);
  } catch (e) { $("#ledger").innerHTML = `<li class="empty error">Couldn't read the changes from the graph.</li>`; }
}
function markStart(el) {
  const st = window.__ruleStart && window.__ruleStart[el.dataset.rule];
  if (!st || el.dataset.rule === "learned" || el.querySelector(".start")) return;
  el.querySelector(".track").insertAdjacentHTML("beforeend", `<div class="start" style="left:${(st.start * 100).toFixed(1)}%" title="Started at ${pct(st.start)}"></div>`);
  el.querySelector(".pct").insertAdjacentHTML("beforeend", `<small>from ${pct(st.start)}</small>`);
}
$("#ledger-more").addEventListener("click", () => { ledgerAll = true; loadLearned(); });

/* ---------- replay ---------- */
function replay() {
  const btn = $("#replay");
  btn.disabled = true; btn.textContent = "Running";
  $("#replay-out").hidden = false;
  const tally = { vector: [0, 0], "vector+memory": [0, 0], decisiongraph: [0, 0] };
  const show = [["vector", "Vector search", "#c9d4ec"], ["vector+memory", "Vector search + memory", "#e8cdbe"], ["decisiongraph", "DecisionGraph, learning", "#7fd0d8"]];
  $("#replay-bars").innerHTML = show.map(([k, l, c]) => `<div class="rbar" style="--c:${c}"><span class="label">${esc(l)}</span><div class="track"><div class="fill" id="rb-${k.replace(/\W/g, "")}"></div></div><span class="pct" id="rp-${k.replace(/\W/g, "")}">0%</span></div>`).join("");
  const es = new EventSource("/api/replay");
  es.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.busy) { es.close(); btn.disabled = false; btn.textContent = "Replay the 50-question run live"; $("#replay-now").textContent = "Someone else is running a replay right now. Try again in a minute."; return; }
    if (d.done) {
      es.close(); btn.disabled = false; btn.textContent = "Replay again";
      $("#replay-now").innerHTML += " Done.";
      loadGraph(); loadLearned(); return;
    }
    for (const [k] of show) {
      tally[k][0] += d[k].ok ? 1 : 0; tally[k][1] += 1;
      const a = tally[k][0] / tally[k][1], id = k.replace(/\W/g, "");
      $(`#rb-${id}`).style.width = `${a * 100}%`; $(`#rp-${id}`).textContent = pct(a);
    }
    const g = d.decisiongraph;
    const learned = (g.learned || []).filter((c) => c.kind !== "hint").map(describeChange).join(" ");
    $("#replay-now").innerHTML = `Question ${d.i} of ${d.n}, round ${d.round}: “${esc(d.q)}” DecisionGraph was ${g.ok ? "right" : "wrong"}. ${learned}`;
  };
  es.onerror = () => { es.close(); btn.disabled = false; btn.textContent = "Replay the 50-question run live"; $("#replay-now").innerHTML = '<span class="error">The replay stopped. The server may be busy; try again.</span>'; };
}

/* ---------- boot ---------- */
$("#ask").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#q").value.trim(); if (q) ask(q); });
$("#yes").addEventListener("click", () => lastAnswer && teach(lastAnswer.id));
$("#no").addEventListener("click", () => { $("#pick").hidden = false; $("#correct").focus(); });
$("#teach").addEventListener("click", () => { const v = $("#correct").value; if (v) teach(v); });
$("#replay").addEventListener("click", replay);

document.querySelectorAll(".reveal").forEach((el) => onVisible(el, () => el.classList.add("in")));
fetch("/static/hero.svg").then((r) => r.text()).then((t) => { $("#hero-art").innerHTML = t; }).catch(() => {});

(async () => {
  try {
    const samples = await api("/api/samples");
    $("#samples").innerHTML = samples.map((s) => `<button type="button" data-q="${esc(s.q)}">${esc(s.q)}</button>`).join("");
    $("#samples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { $("#q").value = b.dataset.q; ask(b.dataset.q); } });
    const ds = await api("/api/decisions");
    $("#correct").innerHTML = `<option value="">Choose a decision</option>` + ds.map((d) => `<option value="${esc(d.id)}">${esc(d.label)}</option>`).join("");
  } catch (e) { /* page still works without samples */ }
  loadResults(); loadGraph(); loadLearned();
  const pre = new URLSearchParams(location.search).get("q");
  if (pre) { $("#q").value = pre; ask(pre); }
})();

