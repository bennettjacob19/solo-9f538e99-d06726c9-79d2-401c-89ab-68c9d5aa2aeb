"""FastAPI 请求/响应模型与入参校验。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from .scheduler import MAX_BATCHES, MAX_EVALUATIONS


class EvaluationInput(BaseModel):
    eval_id: int = Field(..., description="评测场次编号（任意正整数，仅作标识）")
    start: int = Field(..., ge=0, description="评测开始的模拟秒（含）")
    end: int = Field(..., ge=1, description="评测结束的模拟秒（不含）")
    min_checkpoint_batches: int = Field(
        ..., ge=1, description="该场要求最近检查点至少覆盖的批次数"
    )

    @model_validator(mode="after")
    def _check_window(self) -> "EvaluationInput":
        if self.end <= self.start:
            raise ValueError("评测结束秒必须严格大于开始秒")
        return self


class PlanRequest(BaseModel):
    durations: list[int] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCHES,
        description="逐批训练耗时（正整数秒），至多 24 批",
    )
    evaluations: list[EvaluationInput] = Field(
        default_factory=list,
        max_length=MAX_EVALUATIONS,
        description="至多 4 场互不重叠的评测",
    )
    max_checkpoints: int = Field(..., ge=0, description="检查点写入次数上限")
    max_rerun_batches: int = Field(..., ge=0, description="累计重跑批数上限")

    @field_validator("durations")
    @classmethod
    def _durations_positive(cls, v: list[int]) -> list[int]:
        if any(d <= 0 for d in v):
            raise ValueError("每批耗时必须为正整数秒")
        return v

    @model_validator(mode="after")
    def _check_evaluations(self) -> "PlanRequest":
        evals = sorted(self.evaluations, key=lambda e: e.start)
        ids = [e.eval_id for e in evals]
        if len(set(ids)) != len(ids):
            raise ValueError("评测场次编号 eval_id 必须唯一")
        for prev, cur in zip(evals, evals[1:]):
            if cur.start < prev.end:
                raise ValueError(
                    f"评测 {prev.eval_id} 与 {cur.eval_id} 时段重叠"
                )
        for e in evals:
            if e.min_checkpoint_batches > len(self.durations):
                raise ValueError(
                    f"评测 {e.eval_id} 要求的最低存档批次数超过总批次数"
                )
        return self
