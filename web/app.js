/* LegalCourt AI 前端：零构建、零依赖，直接消费 REST + SSE。
 * 关键交互：流式庭审时间线 → 证据溯源抽屉 → 结构化意见书 → HITL 签署/驳回 → 审计校验。
 */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (iso) => {
  try { return new Date(iso).toLocaleTimeString("zh-CN", { hour12: false }); } catch { return ""; }
};

const state = {
  caseId: null,
  hearingId: null,
  running: false,
  evidence: new Map(),   // evidence_id -> evidence 对象
  opinion: null,
  samples: [],
  pendingHitl: null,
};

/* ---------------- 基础请求 ---------------- */

function headers(role) {
  return {
    "Content-Type": "application/json",
    "X-User": ($("#userInput").value || "alice").trim(),
    "X-Role": role || $("#roleSel").value,
  };
}

async function api(path, { method = "GET", body = null, role = null } = {}) {
  const res = await fetch(path, { method, headers: headers(role), body: body ? JSON.stringify(body) : null });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || text || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

let toastTimer = null;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3800);
}

/* ---------------- 初始化 ---------------- */

async function loadEnv() {
  try {
    const h = await api("/healthz");
    $("#chipLLM").textContent = `模型 · ${h.llm_provider}/${h.llm_model}`;
    $("#chipIndex").textContent = `索引 · ${h.index_size} 条 · ${h.vector_backend}`;
    $("#chipThr").textContent = `复核阈值 · ${h.hitl_threshold}`;
  } catch (e) { toast("环境信息读取失败：" + e.message, true); }

  try {
    state.samples = await api("/api/v1/samples");
    const sel = $("#sampleSel");
    state.samples.forEach((s) => {
      const o = el("option", null, `${s.id} · ${s.title || "未命名"}`);
      o.value = s.id;
      sel.appendChild(o);
    });
  } catch (e) { /* 示例缺失不阻断 */ }
}

function loadSample(id) {
  const s = state.samples.find((x) => x.id === id);
  if (!s) return;
  $("#titleInput").value = s.title || "";
  $("#factsInput").value = s.facts || "";
  $("#contractInput").value = s.contract_text || "";
  toast(`已载入示例 ${s.id}：${s.title || ""}`);
}

/* ---------------- 运行与 SSE ---------------- */

function setRunState(kind, text) {
  const box = $("#runState");
  box.innerHTML = "";
  box.appendChild(el("span", "dot " + kind));
  box.appendChild(document.createTextNode(text));
}

function clearStage() {
  $("#timeline").innerHTML = "";
  const empty = $("#timelineEmpty");
  if (empty) empty.remove();
  $("#opinionBody").innerHTML = "";
  $("#hitlPanel").hidden = true;
  state.evidence.clear();
  state.opinion = null;
  state.pendingHitl = null;
}

async function startHearing() {
  if (state.running) return;
  const title = $("#titleInput").value.trim();
  const facts = $("#factsInput").value.trim();
  const contract = $("#contractInput").value.trim();
  if (title.length < 2) return toast("请填写案件标题（至少 2 个字）", true);
  if (facts.length < 10) return toast("争议事实太短，至少 10 个字", true);

  $("#runBtn").disabled = true;
  state.running = true;
  clearStage();
  switchTab("timeline");
  setRunState("run", "创建案件…");

  try {
    const created = await api("/api/v1/cases", {
      method: "POST",
      role: "lawyer",
      body: { title, facts, contract_text: contract || null, contract_filename: null },
    });
    state.caseId = created.case_id;
    setRunState("run", "推演中…");
    await streamHearing();
  } catch (e) {
    setRunState("wait", "失败");
    toast(e.message, true);
  } finally {
    state.running = false;
    $("#runBtn").disabled = false;
  }
}

async function streamHearing() {
  const res = await fetch("/api/v1/hearings/stream", {
    method: "POST",
    headers: headers("lawyer"),
    body: JSON.stringify({
      case_id: state.caseId,
      max_rounds: Number($("#roundsInput").value) || 3,
      auto_review: $("#autoReview").checked,
    }),
  });
  if (!res.ok || !res.body) {
    const text = await res.text();
    throw new Error(text || `推演请求失败（${res.status}）`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      handleSseBlock(block);
    }
  }
  if (buffer.trim()) handleSseBlock(buffer);
  await refreshDetail();
}

function handleSseBlock(block) {
  let event = "message";
  const dataLines = [];
  block.split(/\r?\n/).forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  });
  if (!dataLines.length) return;
  let data = null;
  try { data = JSON.parse(dataLines.join("\n")); } catch { return; }

  switch (event) {
    case "init":
      state.hearingId = data.hearing_id;
      setRunState("run", `庭审 ${data.hearing_id.slice(-6)} · 推演中`);
      break;
    case "timeline":
      addNode(data);
      break;
    case "interrupt":
      showHitl(data);
      setRunState("wait", "等待人工复核");
      break;
    case "awaiting":
      setRunState("wait", "等待人工复核");
      break;
    case "done":
      setRunState("ok", `已完成 · 风险 ${data.risk_level} · 置信 ${data.confidence}`);
      break;
    case "error":
      setRunState("wait", "推演失败");
      toast(data.message || "推演失败", true);
      break;
    default:
      break;
  }
}

async function refreshDetail() {
  if (!state.hearingId) return;
  try {
    const detail = await api(`/api/v1/hearings/${state.hearingId}`);
    renderOpinion(detail);
    if (detail.hitl && detail.hitl.status === "pending") {
      state.pendingHitl = detail.hitl;
      showHitl(detail.hitl);
    } else {
      $("#hitlPanel").hidden = true;
    }
    if (detail.status === "completed") switchTab("opinion");
  } catch (e) { toast("意见书读取失败：" + e.message, true); }
}

/* ---------------- 时间线渲染 ---------------- */

function collectEvidence(list) {
  (list || []).forEach((e) => {
    if (e && e.evidence_id) state.evidence.set(e.evidence_id, e);
  });
}

function evidenceChip(id) {
  const e = state.evidence.get(id);
  const b = el("button", "ev", id + (e ? ` · ${String(e.title || "").slice(0, 14)}` : ""));
  b.dataset.tier = e ? e.tier : "5";
  b.title = e ? e.locator : id;
  b.addEventListener("click", () => openEvidence(id));
  return b;
}

function addNode(ev) {
  const li = el("li", "node");
  li.dataset.node = ev.node || "";

  const top = el("div", "node-top");
  top.appendChild(el("span", "node-actor", ev.actor || ev.node));
  top.appendChild(el("span", null, `#${ev.seq} · ${fmtTime(ev.ts)}`));
  li.appendChild(top);

  li.appendChild(el("div", "node-title", ev.title || ""));
  if (ev.detail) li.appendChild(el("div", "node-detail", ev.detail));

  const p = ev.payload || {};
  collectEvidence(p.evidence);

  const box = el("div", "node-payload");
  let has = false;

  if (p.evidence && p.evidence.length) {
    has = true;
    const chips = el("div", "ev-chips");
    p.evidence.forEach((e) => chips.appendChild(evidenceChip(e.evidence_id)));
    box.appendChild(chips);
  }

  if (p.arguments && p.arguments.length) {
    has = true;
    p.arguments.forEach((a) => {
      const card = el("div", "arg");
      card.appendChild(el("div", "arg-claim", a.claim || a.rebuttal || ""));
      if (a.basis && a.basis.length) {
        const chips = el("div", "ev-chips");
        a.basis.forEach((id) => chips.appendChild(evidenceChip(id)));
        card.appendChild(chips);
      }
      box.appendChild(card);
    });
  }

  if (p.focuses && p.focuses.length && ev.node === "judge") {
    has = true;
    p.focuses.forEach((f) => {
      const row = el("div", "arg");
      row.appendChild(el("div", "arg-claim", `${f.key} · 原告 ${f.plaintiff_strength} / 被告 ${f.defendant_strength} · ${f.status}`));
      if (f.evidence_ids && f.evidence_ids.length) {
        const chips = el("div", "ev-chips");
        f.evidence_ids.forEach((id) => chips.appendChild(evidenceChip(id)));
        row.appendChild(chips);
      }
      box.appendChild(row);
    });
  }

  if (p.risks && p.risks.length) {
    has = true;
    p.risks.forEach((r) => box.appendChild(el("div", "arg-claim", `【${r.level}】${r.name}：${r.advice || ""}`)));
  }

  if (p.missing && p.missing.length) {
    has = true;
    const chips = el("div", "chips");
    p.missing.forEach((m) => chips.appendChild(el("span", "tag", m)));
    box.appendChild(chips);
  }

  if (p.triggers && p.triggers.length) {
    has = true;
    p.triggers.forEach((t) => box.appendChild(el("div", "arg-claim", "· " + t)));
  }

  if (has) li.appendChild(box);
  $("#timeline").appendChild(li);
  li.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------------- 意见书 ---------------- */

function riskBadge(level) {
  const cls = level === "高" ? "high" : level === "中" ? "mid" : "low";
  return `<span class="badge ${cls}">风险 ${esc(level)}</span>`;
}

function statusBadge(s) {
  const map = { resolved: ["gray", "已决"], conflict: ["high", "冲突未决"], undetermined: ["mid", "证据不足"] };
  const [cls, text] = map[s] || ["gray", s];
  return `<span class="badge ${cls}">${text}</span>`;
}

function partyText(p) {
  return { plaintiff: "有利于原告/我方", defendant: "有利于被告/对方", undetermined: "暂不作倾向判断" }[p] || p;
}

function renderOpinion(detail) {
  const box = $("#opinionBody");
  if (!detail || !detail.opinion) {
    box.innerHTML = "";
    return;
  }
  const op = detail.opinion;
  state.opinion = op;
  collectEvidence(op.evidence_snapshot);

  const focuses = (op.focuses || []).map((f) => {
    const chips = (f.evidence_ids || []).map((id) => `<button class="ev" data-ev="${esc(id)}">${esc(id)}</button>`).join("");
    return `
      <div class="focus" data-focus="${esc(f.key)}">
        <div class="focus-head">
          <h4>${esc(f.title)}</h4>
          ${statusBadge(f.status)}
          <span class="badge gray">${esc(f.key)}</span>
        </div>
        <div class="bars">
          <div><div class="bar-row">原告强度 ${f.plaintiff_strength}</div><div class="bar p"><i style="width:${Math.round((f.plaintiff_strength || 0) * 100)}%"></i></div></div>
          <div><div class="bar-row">被告强度 ${f.defendant_strength}</div><div class="bar d"><i style="width:${Math.round((f.defendant_strength || 0) * 100)}%"></i></div></div>
        </div>
        <div class="focus-views">
          <div class="view"><span class="view-label">原告主张</span>${esc(f.plaintiff_view || "—")}</div>
          <div class="view"><span class="view-label">被告抗辩</span>${esc(f.defendant_view || "—")}</div>
        </div>
        ${f.note ? `<div class="note">${esc(f.note)}</div>` : ""}
        <div class="ev-chips">${chips}</div>
      </div>`;
  }).join("");

  const conflicts = (op.conflicts || []).map((c) => `
    <div class="conflict ${c.resolution === "unresolved" ? "unresolved" : ""}">
      <b>${esc(c.focus_key)}</b> · ${esc(c.reason)}<br/>
      证据 <button class="ev" data-ev="${esc(c.evidence_a)}">${esc(c.evidence_a)}</button>
      ↔ <button class="ev" data-ev="${esc(c.evidence_b)}">${esc(c.evidence_b)}</button>
      · 处置：${c.resolution === "unresolved" ? "<b>未决（转人工）</b>" : esc(c.resolved_by || c.resolution)}
    </div>`).join("") || `<div class="node-detail">本次推演未检出同层级证据冲突。</div>`;

  const missing = (op.missing_clauses || []).map((m) => `<span class="tag">${esc(m)}</span>`).join("");
  const remed = (op.remediation || []).map((r) => `<li>${esc(r)}</li>`).join("");
  const cites = (op.citations || []).map((id) => `<button class="ev" data-ev="${esc(id)}">${esc(id)}</button>`).join("");

  box.innerHTML = `
    <div class="verdict">
      <div class="kv">${riskBadge(op.risk_level)}<span>风险分 ${op.risk_score}</span></div>
      <div class="kv"><b>${Math.round((op.confidence || 0) * 100)}%</b><span>置信度</span></div>
      <div class="kv"><b>${op.round_count}</b><span>辩论轮次</span></div>
      <div class="kv"><b>${op.llm_calls}</b><span>LLM 调用</span></div>
      <div class="kv"><b>${esc(partyText(op.inclination_party))}</b><span>倾向性意见</span></div>
      <div class="kv"><b>${esc(op.model)}</b><span>模型 / 版本 ${esc(op.schema_version)}</span></div>
    </div>
    <div class="section-title">结论摘要</div>
    <div class="summary-text">${esc(op.summary || "（无）")}<br/><br/>${esc(op.inclination || "")}</div>
    <div class="section-title">争议焦点（${(op.focuses || []).length}）</div>
    ${focuses || `<div class="node-detail">未识别到争议焦点。</div>`}
    <div class="section-title">证据冲突与处置</div>
    ${conflicts}
    <div class="section-title">缺失条款</div>
    <div class="chips">${missing || `<span class="node-detail">必备条款齐备。</span>`}</div>
    <div class="section-title">补强建议</div>
    <ol class="remediation">${remed || "<li>（无）</li>"}</ol>
    <div class="section-title">引用证据（${(op.citations || []).length}）</div>
    <div class="ev-chips">${cites || "—"}</div>
    <div class="section-title">归档信息</div>
    <div class="mono">opinion_id: ${esc(op.opinion_id)}<br/>audit_tail: ${esc(op.audit_tail || "")}<br/>generated_at: ${esc(op.generated_at)}</div>
    <div class="hitl-buttons" style="margin-top:18px">
      <button class="ghost" id="copyOpinion">复制意见书 JSON</button>
      <button class="ghost" id="downloadOpinion">下载 JSON</button>
    </div>`;

  box.querySelectorAll("[data-ev]").forEach((b) => b.addEventListener("click", () => openEvidence(b.dataset.ev)));
  const copyBtn = box.querySelector("#copyOpinion");
  if (copyBtn) copyBtn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(JSON.stringify(op, null, 2)); toast("已复制到剪贴板"); }
    catch { toast("复制失败，请用下载", true); }
  });
  const dlBtn = box.querySelector("#downloadOpinion");
  if (dlBtn) dlBtn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(op, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${op.opinion_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

/* ---------------- HITL ---------------- */

function showHitl(payload) {
  state.pendingHitl = payload;
  $("#hitlPanel").hidden = false;
  const triggers = payload.triggers || [];
  $("#hitlTriggers").textContent = triggers.length ? triggers.join("；") : "命中复核条件";
  const focusRows = (payload.focuses || []).map((f) =>
    `· ${esc(f.title)}：原告 ${f.plaintiff_strength} / 被告 ${f.defendant_strength}（${esc(f.status)}）`).join("<br/>");
  const conflictRows = (payload.conflicts || []).map((c) => `· ${esc(c.focus_key)}：${esc(c.resolution)}`).join("<br/>");
  $("#hitlSummary").innerHTML = `
    <div>置信度 ${payload.confidence} · 风险 ${esc(payload.risk_level || "")} · ${esc(payload.instruction || "")}</div>
    ${focusRows ? `<div style="margin-top:6px">${focusRows}</div>` : ""}
    ${conflictRows ? `<div style="margin-top:6px">冲突：<br/>${conflictRows}</div>` : ""}`;
  $("#hitlPanel").scrollIntoView({ behavior: "smooth", block: "center" });
}

async function submitHitl(decision) {
  if (!state.hearingId) return toast("尚无可复核的庭审", true);
  const reviewer = $("#reviewerInput").value.trim();
  const comment = $("#reviewComment").value.trim();
  if (reviewer.length < 2) return toast("复核人姓名至少 2 个字", true);
  if (decision === "reject" && !comment) return toast("驳回必须填写复核理由", true);

  $("#approveBtn").disabled = true;
  $("#rejectBtn").disabled = true;
  try {
    const detail = await api(`/api/v1/hearings/${state.hearingId}/hitl`, {
      method: "POST",
      role: "reviewer",
      body: { decision, reviewer, comment },
    });
    $("#hitlPanel").hidden = true;
    renderOpinion(detail);
    if (detail.hitl && detail.hitl.status === "pending") {
      showHitl(detail.hitl);
      toast("已驳回：已进入修订轮次，再次命中复核");
    } else {
      switchTab("opinion");
      toast(decision === "approve" ? "已签署通过，意见书归档" : "已驳回，意见书未归档");
      loadAudit();
    }
    state.pendingHitl = null;
  } catch (e) {
    toast(`复核失败：${e.message}`, true);
  } finally {
    $("#approveBtn").disabled = false;
    $("#rejectBtn").disabled = false;
  }
}

/* ---------------- 证据抽屉 ---------------- */

function openEvidence(id) {
  const e = state.evidence.get(id) || (state.opinion && (state.opinion.evidence_snapshot || []).find((x) => x.evidence_id === id));
  const body = $("#drawerBody");
  if (!e) {
    body.innerHTML = `<div class="ev-title">${esc(id)}</div><div class="node-detail">该证据未随本次响应返回，可在「审计链」中核对引用记录。</div>`;
  } else {
    const tierLabel = { 1: "法律条文", 2: "司法解释", 3: "类案裁判", 4: "合同条款", 5: "事实陈述" }[e.tier] || "其他";
    body.innerHTML = `
      <div class="ev-title">${esc(e.title)}</div>
      <div class="ev-meta">
        <span class="badge gray">${esc(tierLabel)}</span>
        <span class="badge gray">tier ${e.tier}</span>
        <span class="badge gray">score ${Number(e.score || 0).toFixed(3)}</span>
        ${e.favor ? `<span class="badge gray">立场 ${esc(e.favor)}</span>` : ""}
      </div>
      <div class="ev-locator">${esc(e.locator)}</div>
      <div class="ev-text">${esc(e.snippet)}</div>
      ${(e.tags || []).length ? `<div class="chips" style="margin-top:10px">${e.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>` : ""}
      <div class="node-detail" style="margin-top:12px">证据编号 ${esc(e.evidence_id)}；该编号在全链路唯一，可直接用于审计反查。</div>`;
  }
  $("#drawer").hidden = false;
  $(".layout").classList.add("drawer-open");
}

function closeDrawer() {
  $("#drawer").hidden = true;
  $(".layout").classList.remove("drawer-open");
}

/* ---------------- 审计 ---------------- */

async function loadAudit() {
  try {
    const rows = await api("/api/v1/audit?limit=60", { role: $("#roleSel").value });
    const tbody = $("#auditTable").querySelector("tbody");
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td class="mono">${esc((r.ts || "").replace("T", " ").slice(0, 19))}</td>
        <td>${esc(r.action)}</td>
        <td>${esc(r.actor)}</td>
        <td class="mono">${esc(r.case_id || r.hearing_id || "")}</td>
        <td class="mono" title="${esc(r.hash)}">${esc((r.hash || "").slice(0, 12))}…</td>
      </tr>`).join("");
    $("#auditResult").textContent = `已加载 ${rows.length} 条（最近的在前）`;
  } catch (e) { toast("审计读取失败：" + e.message, true); }
}

async function verifyAudit() {
  try {
    const r = await api("/api/v1/audit/verify");
    $("#auditResult").textContent = r.ok
      ? `哈希链完整：校验 ${r.checked} 条，无断裂`
      : `哈希链断裂：位置 ${r.broken_at}（${r.reason || ""}）`;
    toast(r.ok ? "审计链校验通过" : "审计链校验失败", !r.ok);
  } catch (e) { toast("校验失败：" + e.message, true); }
}

/* ---------------- 交互绑定 ---------------- */

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.id === `pane-${name}`));
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("#runBtn").addEventListener("click", startHearing);
  $("#sampleSel").addEventListener("change", (e) => loadSample(e.target.value));
  $("#approveBtn").addEventListener("click", () => submitHitl("approve"));
  $("#rejectBtn").addEventListener("click", () => submitHitl("reject"));
  $("#auditLoad").addEventListener("click", loadAudit);
  $("#auditVerify").addEventListener("click", verifyAudit);
  $("#drawerClose").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  loadEnv();
});
