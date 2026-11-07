"""LangGraph 状态图装配。

路由一览：
    consult  : retrieve → consult_answer → END
    review   : retrieve → clause_risk → hitl? → finalize → END
    dispute  : retrieve → plaintiff → defendant → judge →（未收敛则回到 plaintiff）
               → cite_guard → hitl? → finalize → END
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command  # noqa: F401  (对外暴露，便于 service 调用)

from ..config import settings
from . import nodes
from .state import HearingState

_SAVER: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    global _SAVER
    if _SAVER is None:
        path = settings.db_path.parent / "checkpoints.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        _SAVER = SqliteSaver(conn)
    return _SAVER


def route_after_retrieve(state: HearingState) -> str:
    route = state.get("route", "dispute_simulation")
    if route == "statute_consult":
        return "consult_answer"
    if route == "contract_review":
        return "clause_risk"
    return "plaintiff"


def route_after_judge(state: HearingState) -> str:
    return "cite_guard" if state.get("converged") else "plaintiff"


def route_after_hitl(state: HearingState) -> str:
    hitl = state.get("hitl") or {}
    if hitl.get("decision") == "reject" and int(state.get("revision_count", 0)) <= settings.max_revisions:
        return "plaintiff"
    return "finalize"


def build_graph(checkpointer: Any | None = None):
    g: StateGraph = StateGraph(HearingState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("orchestrate", nodes.orchestrate)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("plaintiff", nodes.plaintiff)
    g.add_node("defendant", nodes.defendant)
    g.add_node("judge", nodes.judge)
    g.add_node("cite_guard", nodes.cite_guard)
    g.add_node("hitl_gate", nodes.hitl_gate)
    g.add_node("clause_risk", nodes.clause_risk)
    g.add_node("consult_answer", nodes.consult_answer)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "orchestrate")
    g.add_edge("orchestrate", "retrieve")

    g.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {"plaintiff": "plaintiff", "clause_risk": "clause_risk", "consult_answer": "consult_answer"},
    )
    g.add_edge("plaintiff", "defendant")
    g.add_edge("defendant", "judge")
    g.add_conditional_edges("judge", route_after_judge, {"cite_guard": "cite_guard", "plaintiff": "plaintiff"})
    g.add_edge("cite_guard", "hitl_gate")
    g.add_edge("clause_risk", "hitl_gate")
    g.add_conditional_edges("hitl_gate", route_after_hitl, {"plaintiff": "plaintiff", "finalize": "finalize"})
    g.add_edge("finalize", END)
    g.add_edge("consult_answer", END)

    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph(get_checkpointer())
