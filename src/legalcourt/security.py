"""RBAC + ABAC：角色→权限矩阵 + 案件归属校验。

演示用身份走请求头（X-User / X-Role / X-Dept），生产替换为 JWT/OIDC，
只需替换 `get_principal`，其余鉴权逻辑不变。
"""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import unquote

from fastapi import Header, HTTPException

from . import audit

ROLE_PERMS: dict[str, set[str]] = {
    "lawyer": {"case:create", "case:read", "hearing:run", "hearing:read", "audit:read", "corpus:read"},
    "reviewer": {"case:read", "hearing:read", "hitl:decide", "audit:read", "corpus:read"},
    "auditor": {"case:read", "hearing:read", "audit:read", "corpus:read"},
    "admin": {
        "case:create", "case:read", "hearing:run", "hearing:read",
        "hitl:decide", "audit:read", "corpus:read", "user:manage",
    },
}

DEFAULT_ROLE = "lawyer"


class Principal:
    def __init__(self, user: str, role: str, dept: str) -> None:
        self.user = user
        self.role = role if role in ROLE_PERMS else DEFAULT_ROLE
        self.dept = dept

    @property
    def perms(self) -> set[str]:
        return ROLE_PERMS[self.role]

    def can(self, perm: str) -> bool:
        return perm in self.perms

    def __repr__(self) -> str:  # pragma: no cover
        return f"Principal(user={self.user}, role={self.role}, dept={self.dept})"


def get_principal(
    x_user: str = Header(default="alice", alias="X-User"),
    x_role: str = Header(default=DEFAULT_ROLE, alias="X-Role"),
    x_dept: str = Header(default="法务一部", alias="X-Dept"),
) -> Principal:
    # 请求头只允许 latin-1，中文部门名请用 percent-encoding（前端/网关负责编码）
    return Principal(user=unquote(x_user).strip() or "anonymous", role=unquote(x_role).strip(),
                     dept=unquote(x_dept).strip() or "法务一部")


def require(perm: str) -> Callable[..., Principal]:
    def _dep(principal: Principal = None) -> Principal:  # type: ignore[assignment]
        raise RuntimeError("dependency must be overridden by FastAPI")

    def _checker(x_user: str = Header(default="alice", alias="X-User"),
                 x_role: str = Header(default=DEFAULT_ROLE, alias="X-Role"),
                 x_dept: str = Header(default="法务一部", alias="X-Dept")) -> Principal:
        p = Principal(unquote(x_user).strip() or "anonymous", unquote(x_role).strip(),
                      unquote(x_dept).strip() or "法务一部")
        if not p.can(perm):
            audit.append("system", "rbac.denied", p.user, {"perm": perm, "role": p.role})
            raise HTTPException(status_code=403, detail=f"角色 {p.role} 无权限：{perm}")
        return p

    return _checker


def ensure_case_access(principal: Principal, case_row: dict[str, Any] | None, perm: str = "case:read") -> dict[str, Any]:
    """ABAC：本人/本部门可读；审计员与管理员可跨部门只读。"""
    if not case_row:
        raise HTTPException(status_code=404, detail="案件不存在")
    if not principal.can(perm):
        audit.append(case_row["id"], "rbac.denied", principal.user, {"perm": perm})
        raise HTTPException(status_code=403, detail=f"无权限：{perm}")
    if principal.role in {"admin", "auditor"}:
        return case_row
    if case_row.get("owner") == principal.user or case_row.get("dept") == principal.dept:
        return case_row
    audit.append(case_row["id"], "abac.denied", principal.user,
                 {"dept": principal.dept, "case_dept": case_row.get("dept")})
    raise HTTPException(status_code=403, detail="无权访问其他部门案件")
