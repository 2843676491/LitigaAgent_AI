"""REST + SSE 接口层。

鉴权：RBAC（角色→权限）+ ABAC（案件归属部门），见 security.py。
流式：推演过程以 SSE 逐节点推送，前端可实时看到"原告→被告→法官"的交锋。
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import audit, db, service
from ..config import settings
from ..rag.hybrid import get_retriever
from ..rag.ingest import md5_of, split_contract
from ..schemas import (
    AuditRecord,
    CaseCreate,
    CaseView,
    HearingDetail,
    HearingRequest,
    HitlDecision,
    SearchHit,
)
from ..security import Principal, ensure_case_access, require

API = settings.api_prefix


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 预热：装载语料、建立向量与 BM25 索引
    retriever = get_retriever()
    audit.append("system", "app.start", "system", {"index_size": retriever.index.size,
                                                   "llm": settings.llm_provider,
                                                   "vector_backend": getattr(retriever.index.vector, "kind", "unknown")})
    yield


app = FastAPI(title="LegalCourt AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    retriever = get_retriever()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": __import__("legalcourt.llm.base", fromlist=["get_llm"]).get_llm().model,
        "index_size": retriever.index.size,
        "vector_backend": getattr(retriever.index.vector, "kind", "unknown"),
        "max_rounds": settings.max_rounds,
        "hitl_threshold": settings.hitl_confidence_threshold,
    }


@app.get(f"{API}/config")
def get_config() -> dict[str, Any]:
    return {
        "app": settings.app_name,
        "llm_provider": settings.llm_provider,
        "max_rounds": settings.max_rounds,
        "hitl_threshold": settings.hitl_confidence_threshold,
        "top_k": settings.top_k_final,
        "roles": ["lawyer", "reviewer", "auditor", "admin"],
    }


# ---------------- 案件 ----------------

@app.post(f"{API}/cases", response_model=CaseView)
def create_case(payload: CaseCreate, principal: Principal = Depends(require("case:create"))) -> CaseView:
    contract = payload.contract_text or ""
    clause_count = len(split_contract(contract, "preview", payload.contract_filename)) if contract else 0
    row = db.create_case(
        title=payload.title,
        facts=payload.facts,
        contract_text=contract or None,
        contract_filename=payload.contract_filename,
        owner=principal.user,
        dept=payload.dept or principal.dept,
        clause_count=clause_count,
    )
    audit.append(row["id"], "case.created", principal.user,
                 {"title": payload.title, "clauses": clause_count, "md5": md5_of(contract)})
    return CaseView(**{k: row[k] for k in ("id", "title", "facts", "contract_filename", "clause_count", "created_at", "owner", "dept")} | {"case_id": row["id"]})


@app.post(f"{API}/cases/upload", response_model=CaseView)
async def create_case_upload(
    title: str = Form(...),
    facts: str = Form(...),
    file: UploadFile | None = File(default=None),
    principal: Principal = Depends(require("case:create")),
) -> CaseView:
    filename = file.filename if file else None
    text = ""
    if file:
        raw = await file.read()
        if Path(filename or "").suffix.lower() == ".pdf":
            raise HTTPException(status_code=400, detail="PDF 暂不解析，请粘贴文本或上传 .md/.txt（原件已不入库时可重试）")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gbk", errors="ignore")
        obj_path = settings.object_dir / f"{md5_of(raw)}-{filename}"
        obj_path.write_bytes(raw)
        # 由 create_case 统一落库后再关联文书，这里先创建案件
    clause_count = len(split_contract(text, "preview", filename)) if text else 0
    row = db.create_case(title=title, facts=facts, contract_text=text or None, contract_filename=filename,
                         owner=principal.user, dept=principal.dept, clause_count=clause_count)
    if file:
        db.save_document(row["id"], filename or "upload", raw, str(obj_path))
    audit.append(row["id"], "case.created.upload", principal.user,
                 {"filename": filename, "clauses": clause_count})
    return CaseView(**{k: row[k] for k in ("id", "title", "facts", "contract_filename", "clause_count", "created_at", "owner", "dept")} | {"case_id": row["id"]})


@app.get(f"{API}/cases", response_model=list[CaseView])
def list_cases(limit: int = 30, principal: Principal = Depends(require("case:read"))) -> list[CaseView]:
    scope = None if principal.role in {"admin", "auditor"} else principal.dept
    rows = db.list_cases(limit=limit, dept=scope)
    return [CaseView(**{k: r.get(k) for k in ("title", "facts", "contract_filename", "clause_count", "created_at", "owner", "dept")} | {"case_id": r["id"]}) for r in rows]


@app.get(f"{API}/cases/{{case_id}}", response_model=CaseView)
def get_case(case_id: str, principal: Principal = Depends(require("case:read"))) -> CaseView:
    row = ensure_case_access(principal, db.get_case(case_id))
    return CaseView(**{k: row.get(k) for k in ("title", "facts", "contract_filename", "clause_count", "created_at", "owner", "dept")} | {"case_id": row["id"]})


# ---------------- 推演 ----------------

@app.post(f"{API}/hearings/stream")
def run_hearing_stream(payload: HearingRequest, principal: Principal = Depends(require("hearing:run"))):
    case = db.get_case(payload.case_id)
    ensure_case_access(principal, case, perm="hearing:run")

    def gen() -> Iterator[str]:
        for item in service.run_streaming(
            payload.case_id,
            max_rounds=payload.max_rounds,
            auto_review=payload.auto_review,
            actor=principal.user,
        ):
            yield _sse(item["event"], item["data"])

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get(f"{API}/hearings/{{hearing_id}}", response_model=HearingDetail)
def get_hearing(hearing_id: str, principal: Principal = Depends(require("hearing:read"))) -> HearingDetail:
    row = db.get_hearing(hearing_id)
    if not row:
        raise HTTPException(status_code=404, detail="庭审不存在")
    ensure_case_access(principal, db.get_case(row["case_id"]))
    return service.get_detail(hearing_id)


@app.post(f"{API}/hearings/{{hearing_id}}/hitl", response_model=HearingDetail)
def submit_hitl(hearing_id: str, decision: HitlDecision,
                principal: Principal = Depends(require("hitl:decide"))) -> HearingDetail:
    row = db.get_hearing(hearing_id)
    if not row:
        raise HTTPException(status_code=404, detail="庭审不存在")
    ensure_case_access(principal, db.get_case(row["case_id"]), perm="hitl:decide")
    if row["status"] != "awaiting_review":
        raise HTTPException(status_code=409, detail=f"当前状态 {row['status']} 不接受复核（可能已被其他复核人处理）")
    try:
        service.resume(hearing_id, {"decision": decision.decision, "reviewer": decision.reviewer,
                                    "comment": decision.comment}, actor=decision.reviewer)
    except service.HearingStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.get_detail(hearing_id)


@app.get(f"{API}/cases/{{case_id}}/hearings")
def list_case_hearings(case_id: str, principal: Principal = Depends(require("hearing:read"))) -> list[dict[str, Any]]:
    ensure_case_access(principal, db.get_case(case_id))
    return db.list_hearings(case_id)


# ---------------- 审计 ----------------

@app.get(f"{API}/audit", response_model=list[AuditRecord])
def list_audit(case_id: str | None = None, hearing_id: str | None = None, limit: int = 200,
               principal: Principal = Depends(require("audit:read"))) -> list[AuditRecord]:
    if case_id:
        ensure_case_access(principal, db.get_case(case_id))
    return [AuditRecord(**r) for r in db.list_audit(case_id=case_id, hearing_id=hearing_id, limit=limit)]


@app.get(f"{API}/audit/verify")
def verify_audit(case_id: str | None = None, principal: Principal = Depends(require("audit:read"))) -> dict[str, Any]:
    return audit.verify_chain(case_id=case_id)


# ---------------- 语料与样例 ----------------

@app.get(f"{API}/corpus/search", response_model=list[SearchHit])
def search_corpus(q: str = Query(min_length=1), k: int = 6,
                  principal: Principal = Depends(require("corpus:read"))) -> list[SearchHit]:
    return [SearchHit(evidence=e) for e in get_retriever().search(q, top_k=min(k, 20))]


@app.get(f"{API}/samples")
def list_samples(principal: Principal = Depends(require("case:read"))) -> list[dict[str, Any]]:
    """示例场景：按文件名前缀配对『合同』与『事实』。"""
    out: list[dict[str, Any]] = []
    if not settings.sample_dir.exists():
        return out
    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(settings.sample_dir.glob("*.md")):
        m = re.match(r"(\d+)_(.*)", path.stem)
        key = m.group(1) if m else path.stem
        kind = "contract" if "合同" in path.stem else ("facts" if "事实" in path.stem else "other")
        g = groups.setdefault(key, {"id": key, "contract_text": "", "facts": "", "contract_filename": "", "title": ""})
        content = path.read_text(encoding="utf-8")
        if kind == "contract":
            g["contract_text"] = content
            g["contract_filename"] = path.name
            g["title"] = (path.stem.split("_", 1)[-1])
        elif kind == "facts":
            g["facts"] = content
    for key, g in sorted(groups.items()):
        if g["contract_text"] or g["facts"]:
            out.append(g)
    return out


# ---------------- 静态前端 ----------------
if Path(settings.web_dir).exists():
    app.mount("/", StaticFiles(directory=str(settings.web_dir), html=True), name="web")
