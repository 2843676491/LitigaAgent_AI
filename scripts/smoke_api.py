"""端到端冒烟：不依赖浏览器，直接打 HTTP 接口走一遍「建案 → SSE 推演 → HITL → 意见书」。

用法：先启动服务，再 `python scripts/smoke_api.py`（可加 --weak 走证据不足场景）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
H = {"X-User": "smoke", "X-Role": "lawyer"}


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", action="store_true", help="跑无书面合同的弱证据场景")
    ap.add_argument("--sample", default="01")
    args = ap.parse_args()

    with httpx.Client(base_url=BASE, timeout=60) as c:
        env = c.get("/healthz").json()
        print(f"[health] {env}")

        if args.weak:
            payload = {
                "title": "口头咨询服务费争议",
                "facts": "我方为对方提供企业管理咨询服务，双方口头约定服务费20万元。对方仅支付5万元后拒付，"
                         "称咨询成果未达标，但未提供任何书面验收或异议记录。我方仅有微信聊天记录，未签订书面合同，多次催告无果。",
                "contract_text": None,
            }
        else:
            payload = {
                "title": f"示例{args.sample}·端到端冒烟",
                "facts": read(f"data/samples/{args.sample}_争议事实.md"),
                "contract_text": read(f"data/samples/{args.sample}_设备采购合同.md")
                if args.sample == "01" else None,
            }
            if args.sample == "02":
                payload["contract_text"] = read("data/samples/02_软件开发合同.md")
            if args.sample == "03":
                payload["contract_text"] = read("data/samples/03_建设工程施工合同.md")

        case = c.post("/api/v1/cases", json=payload, headers=H).json()
        print(f"[case] {case['case_id']} clauses={case['clause_count']}")

        events = []
        with c.stream("POST", "/api/v1/hearings/stream",
                      json={"case_id": case["case_id"], "max_rounds": 3, "auto_review": False},
                      headers=H) as r:
            event = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    events.append((event, data))
                    if event == "timeline":
                        print(f"  · {data['node']:<12} {data['title']}")
                    elif event == "interrupt":
                        print(f"  ! 复核命中：{data.get('triggers')}")
                    elif event in ("done", "error", "awaiting"):
                        print(f"  · {event}: {json.dumps(data, ensure_ascii=False)}")

        hearing_id = next(d for e, d in events if e == "init")["hearing_id"]
        detail = c.get(f"/api/v1/hearings/{hearing_id}").json()
        print(f"[status] {detail['status']}")

        if detail["status"] == "awaiting_review":
            comment = "证据不足，需补充书面合同与验收记录" if args.weak else "核验通过"
            r = c.post(f"/api/v1/hearings/{hearing_id}/hitl",
                       json={"decision": "approve", "reviewer": "冒烟复核", "comment": comment},
                       headers={"X-User": "smoke", "X-Role": "reviewer"})
            print(f"[hitl] {r.status_code}")
            detail = r.json()

        op = detail.get("opinion")
        if op:
            print(f"[opinion] risk={op['risk_level']} conf={op['confidence']} "
                  f"focus={len(op['focuses'])} citations={len(op['citations'])}")
        verify = c.get("/api/v1/audit/verify").json()
        print(f"[audit] {verify}")
        if not verify.get("ok"):
            sys.exit(1)


if __name__ == "__main__":
    main()
