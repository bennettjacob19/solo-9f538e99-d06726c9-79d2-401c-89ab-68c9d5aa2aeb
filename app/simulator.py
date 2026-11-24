"""模拟时钟驱动的演练会话。

排程由 :mod:`app.scheduler` 预先求得并保证可行；本模块负责按模拟时钟
重放排程，并在对应时段真正执行 NumPy 训练、检查点落盘与推理：

* 训练时段调用 :func:`app.ml.train_one_batch`，重跑批次使用相同数据与
  相同起点权重，更新结果与首跑逐元素一致；
* 检查点时段把当前权重写入本地 .npz 文件（模拟独占卡 2 秒）；
* 评测时段加载最近检查点权重，在固定评测集上推理，随后把训练权重
  回滚到该检查点（从同一检查点续训）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ml
from .scheduler import PlanSegment, SolveResult


@dataclass
class ExecutedEvent:
    segment: PlanSegment
    inference: Optional[dict] = None
    checkpoint_path: Optional[str] = None

    def as_dict(self) -> dict:
        out = self.segment.as_dict()
        if self.inference is not None:
            out["inference"] = self.inference
        if self.checkpoint_path is not None:
            out["checkpoint_path"] = self.checkpoint_path
        return out


@dataclass
class Session:
    session_id: str
    plan: SolveResult
    data_dir: str
    num_batches: int

    clock: int = 0
    cursor: int = 0
    current_weights: np.ndarray = field(default_factory=ml.initial_weights)
    # 已首跑完成的最大批次数（用于判定某段是否为重跑）。
    peak_batches: int = 0
    # 当前已完成（含重跑）到的批次序号。
    trained_cursor: int = 0
    reruns_executed: int = 0
    checkpoints_written: int = 0
    events: list[ExecutedEvent] = field(default_factory=list)
    # checkpoint_no -> Checkpoint 对象。
    checkpoint_store: dict = field(default_factory=dict)
    finished: bool = False

    def __post_init__(self) -> None:
        self.ckpt_dir = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.reference_weights = ml.reference_uninterrupted(self.num_batches)

    # ------------------------------------------------------------------
    # 时钟推进
    # ------------------------------------------------------------------
    def advance(self, seconds: int) -> dict:
        if seconds <= 0:
            raise ValueError("推进时长必须为正整数秒")
        if self.finished:
            return {
                "clock": self.clock,
                "advanced_events": [],
                "finished": True,
            }

        target = self.clock + seconds
        produced: list[ExecutedEvent] = []

        while self.cursor < len(self.plan.segments):
            seg = self.plan.segments[self.cursor]
            if seg.start >= target or seg.end > target:
                # 下一时段尚未开始，或时段不可拆、当前跑不完：
                # 不产生任何副作用，时钟先推进到请求时刻。
                break
            event = self._execute(seg)
            produced.append(event)
            self.events.append(event)
            self.cursor += 1
            self.clock = seg.end

        if self.cursor >= len(self.plan.segments):
            self.finished = True
            self.clock = self.plan.makespan
        else:
            # 未完成：时钟走到请求时刻，但不超过排程总时长。
            self.clock = min(target, self.plan.makespan)

        return {
            "clock": self.clock,
            "advanced_events": [e.as_dict() for e in produced],
            "finished": self.finished,
        }

    # ------------------------------------------------------------------
    # 各时段的实际副作用
    # ------------------------------------------------------------------
    def _execute(self, seg: PlanSegment) -> ExecutedEvent:
        if seg.kind == "train":
            assert seg.batch_index == self.trained_cursor, (
                f"排程批次序列断裂：期望 {self.trained_cursor}，"
                f"实际 {seg.batch_index}"
            )
            self.current_weights = ml.train_one_batch(
                self.current_weights, seg.batch_index
            )
            self.trained_cursor += 1
            if seg.rerun:
                self.reruns_executed += 1
            else:
                self.peak_batches = max(self.peak_batches, self.trained_cursor)
            return ExecutedEvent(seg)

        if seg.kind == "checkpoint":
            self.checkpoints_written += 1
            ckpt = ml.save_checkpoint(
                self.ckpt_dir,
                checkpoint_id=self.checkpoints_written,
                batches_trained=self.trained_cursor,
                weights=self.current_weights,
            )
            self.checkpoint_store[self.checkpoints_written] = ckpt
            return ExecutedEvent(seg, checkpoint_path=ckpt.path)

        if seg.kind == "evaluation":
            assert seg.checkpoint_no in self.checkpoint_store
            ckpt = self.checkpoint_store[seg.checkpoint_no]
            # 用最近写完的检查点“实际推理”：从磁盘加载后再推理。
            loaded_weights, loaded_batches = ml.load_checkpoint(ckpt.path)
            assert loaded_batches == ckpt.batches_trained
            inference = ml.run_inference(loaded_weights)
            inference["used_checkpoint_no"] = seg.checkpoint_no
            inference["used_checkpoint_batches"] = ckpt.batches_trained
            event = ExecutedEvent(seg, inference=inference)
            # 评测结束后从同一检查点续训：权重与进度回滚。
            rolled_back = self.trained_cursor - ckpt.batches_trained
            assert rolled_back >= 0
            self.current_weights = loaded_weights.copy()
            self.trained_cursor = ckpt.batches_trained
            return event

        if seg.kind == "idle":
            return ExecutedEvent(seg)

        raise ValueError(f"未知时段类型: {seg.kind}")

    # ------------------------------------------------------------------
    # 状态与结果
    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "clock": self.clock,
            "finished": self.finished,
            "trained_cursor": self.trained_cursor,
            "peak_batches_trained": self.peak_batches,
            "reruns_executed": self.reruns_executed,
            "checkpoints_written": self.checkpoints_written,
            "limits": {
                "checkpoints_planned": self.plan.checkpoints_used,
                "reruns_planned": self.plan.reruns_total,
                "makespan": self.plan.makespan,
            },
            "executed_segments": [e.as_dict() for e in self.events],
        }

    def results(self) -> dict:
        out = {
            "finished": self.finished,
            "clock": self.clock,
            "reruns_executed": self.reruns_executed,
            "checkpoints_written": self.checkpoints_written,
        }
        if self.finished:
            final_w = self.current_weights
            ref_w = self.reference_weights
            out["final_weights"] = [float(v) for v in final_w]
            out["reference_uninterrupted_weights"] = [
                float(v) for v in ref_w
            ]
            out["max_abs_diff"] = float(np.max(np.abs(final_w - ref_w)))
            out["consistent_with_uninterrupted"] = bool(
                np.array_equal(final_w, ref_w)
            )
        else:
            out["final_weights"] = None
            out["consistent_with_uninterrupted"] = None
        return out
