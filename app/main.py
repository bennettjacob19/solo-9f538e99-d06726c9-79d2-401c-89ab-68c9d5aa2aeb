"""FastAPI 入口：本地单卡夜班训练 + 定时评测演练。

端点：
* ``POST /plans``          提交排程请求；有解才创建演练会话，无解不启动
* ``GET  /sessions``       列出会话
* ``GET  /sessions/{id}``  查询模拟时钟与已执行时段
* ``POST /sessions/{id}/advance``  推进模拟时钟，重放训练/存档/评测
* ``GET  /sessions/{id}/results``  最终权重与不间断训练一致性结论
* ``GET  /docs``           Swagger UI（自动生成）
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from . import ml
from .schemas import PlanRequest
from .scheduler import EvaluationSpec, solve
from .simulator import Session

DATA_DIR = os.environ.get("APP_DATA_DIR", os.path.join(os.getcwd(), "sessions_data"))
os.makedirs(DATA_DIR, exist_ok=True)

SESSIONS: dict[str, Session] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 内存会话随进程结束；检查点文件保留在 DATA_DIR 供核查。


app = FastAPI(
    title="单卡夜班训练与定时评测演练",
    version="1.0.0",
    description=(
        "提交至多 24 批训练耗时、至多 4 场互不重叠评测及检查点/重跑预算，"
        "求解器在满足全部独占与续训约束时受理并启动确定性模拟；"
        "推进模拟时钟可查实跑时段、重跑批数与推理结果，最终权重与"
        "不间断训练逐元素一致。"
    ),
    lifespan=lifespan,
)


@app.post("/plans", status_code=201)
def create_plan(req: PlanRequest) -> dict:
    specs = [
        EvaluationSpec(
            eval_id=e.eval_id,
            start=e.start,
            end=e.end,
            min_checkpoint_batches=e.min_checkpoint_batches,
        )
        for e in req.evaluations
    ]
    result = solve(
        durations=list(req.durations),
        evaluations=specs,
        max_checkpoints=req.max_checkpoints,
        max_rerun_batches=req.max_rerun_batches,
    )
    if not result.feasible:
        # 无解不启动：不创建任何会话/目录，返回 422 说明原因。
        raise HTTPException(
            status_code=422,
            detail={
                "accepted": False,
                "message": "无满足限制的排法，演练未启动",
                "reasons": result.reasons,
            },
        )

    session_id = uuid.uuid4().hex[:12]
    session = Session(
        session_id=session_id,
        plan=result,
        data_dir=DATA_DIR,
        num_batches=len(req.durations),
    )
    SESSIONS[session_id] = session
    return {
        "accepted": True,
        "session_id": session_id,
        "checkpoints_used": result.checkpoints_used,
        "reruns_total": result.reruns_total,
        "makespan": result.makespan,
        "segments": [s.as_dict() for s in result.segments],
    }


@app.get("/sessions")
def list_sessions() -> dict:
    return {
        "sessions": [
            {"session_id": sid, **{k: v for k, v in s.status().items()
                                   if k != "session_id"}}
            for sid, s in sorted(SESSIONS.items())
        ]
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = _require_session(session_id)
    return session.status()


@app.post("/sessions/{session_id}/advance")
def advance_clock(session_id: str, seconds: int) -> dict:
    session = _require_session(session_id)
    if seconds <= 0:
        raise HTTPException(status_code=400, detail="seconds 必须为正整数")
    return session.advance(seconds)


@app.get("/sessions/{session_id}/results")
def get_results(session_id: str) -> dict:
    session = _require_session(session_id)
    out = session.results()
    out["session_id"] = session_id
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "numpy": np.__version__}


def _require_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="会话不存在或未受理")
    return SESSIONS[session_id]
