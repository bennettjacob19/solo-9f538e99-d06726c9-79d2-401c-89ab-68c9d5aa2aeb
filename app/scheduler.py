"""单卡排程求解器。

约束（与需求一一对应）：

* 训练批次不可拆、不可抢占；写本地检查点独占卡固定 2 秒；评测准点
  独占卡，时段固定，不可被训练/存档占用。
* 评测使用最近写完的检查点推理；推理后训练回滚到该检查点，检查点
  之后完成的批次全部重跑，累计重跑批数受上限约束。
* 每场评测要求最近检查点覆盖的批次数达到该场的最低存档批次。
* 检查点写入次数、累计重跑批数均有上限。
* 无评测的空档里，只要放得下“下一批”，该时段必须用于训练或存档
  （仅当训练和存档都放不下/无意义时才允许空闲）。
* 存在满足全部限制的排法才受理，否则返回无解原因。

求解方式：按评测窗口把时间线切成有限空档 + 评测段，用带回滚状态
记忆化的深度优先搜索枚举“训练 / 存档”决策；同一空档内相同进度
只保留最早到达（支配剪枝）。规模受 24 批 / 4 场评测约束，可穷举。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CHECKPOINT_SECONDS = 2
MAX_BATCHES = 24
MAX_EVALUATIONS = 4


@dataclass(frozen=True)
class EvaluationSpec:
    eval_id: int
    start: int
    end: int
    min_checkpoint_batches: int


@dataclass
class PlanSegment:
    kind: str  # train | checkpoint | evaluation | idle
    start: int
    end: int
    # train 专用
    batch_index: Optional[int] = None
    rerun: bool = False
    # checkpoint 专用（第几个检查点，1 基；及其覆盖的批次数）
    checkpoint_no: Optional[int] = None
    batches_trained: Optional[int] = None
    # evaluation 专用
    eval_id: Optional[int] = None
    min_checkpoint_batches: Optional[int] = None

    def as_dict(self) -> dict:
        out: dict = {"kind": self.kind, "start": self.start, "end": self.end}
        if self.batch_index is not None:
            out["batch_index"] = self.batch_index
            out["rerun"] = self.rerun
        if self.checkpoint_no is not None:
            out["checkpoint_no"] = self.checkpoint_no
        if self.batches_trained is not None:
            out["batches_trained"] = self.batches_trained
        if self.eval_id is not None:
            out["eval_id"] = self.eval_id
        if self.min_checkpoint_batches is not None:
            out["min_checkpoint_batches"] = self.min_checkpoint_batches
        return out


@dataclass
class SolveResult:
    feasible: bool
    segments: list[PlanSegment] = field(default_factory=list)
    checkpoints_used: int = 0
    reruns_total: int = 0
    makespan: int = 0
    reasons: list[str] = field(default_factory=list)


def solve(
    durations: list[int],
    evaluations: list[EvaluationSpec],
    max_checkpoints: int,
    max_rerun_batches: int,
) -> SolveResult:
    n = len(durations)
    evals = sorted(evaluations, key=lambda e: e.start)
    # 每场评测只保留信息量最大的一条失败原因，避免 DFS 重复刷屏。
    reason_map: dict[int, str] = {}

    def fail(eval_index: int, reason: str) -> None:
        reason_map[eval_index] = reason

    # 无评测：顺序训练即可，天然满足“空档必须利用”。
    if not evals:
        segments: list[PlanSegment] = []
        t = 0
        for b in range(n):
            segments.append(PlanSegment("train", t, t + durations[b],
                                        batch_index=b))
            t += durations[b]
        return SolveResult(True, segments, 0, 0, t)

    # 每个空档各自的支配记忆：状态 -> 该状态最早到达时刻。
    gap_memos: list[dict[tuple, int]] = [dict() for _ in range(len(evals))]

    def do_eval(j: int, cur: int, peak: int, last_cp: int, cp_used: int,
                reruns_used: int) -> Optional[tuple[list[PlanSegment], int,
                                                    int, int]]:
        spec = evals[j]
        if last_cp < spec.min_checkpoint_batches:
            fail(
                j,
                f"评测 {spec.eval_id} 开始时最近检查点仅覆盖 {last_cp} 批，"
                f"低于要求的 {spec.min_checkpoint_batches} 批",
            )
            return None
        rolled_back = cur - last_cp
        new_reruns = reruns_used + rolled_back
        if new_reruns > max_rerun_batches:
            fail(
                j,
                f"评测 {spec.eval_id} 将重跑 {rolled_back} 批，"
                f"累计 {new_reruns} 批超过上限 {max_rerun_batches}",
            )
            return None

        eval_seg = PlanSegment(
            "evaluation",
            spec.start,
            spec.end,
            eval_id=spec.eval_id,
            checkpoint_no=cp_used,  # 最近写完的检查点序号
            batches_trained=last_cp,
            min_checkpoint_batches=spec.min_checkpoint_batches,
        )

        if j + 1 < len(evals):
            rest = gap_dfs(
                j + 1,
                spec.end,
                evals[j + 1].start,
                last_cp,
                peak,
                last_cp,
                cp_used,
                new_reruns,
            )
            if rest is None:
                return None
            rest_segments, r_cp, r_reruns, r_makespan = rest
            return [eval_seg] + rest_segments, r_cp, r_reruns, r_makespan

        # 最后一场评测之后：顺序训完剩余批次，不再存档。
        t = spec.end
        cur2 = last_cp
        tail: list[PlanSegment] = []
        while cur2 < n:
            tail.append(
                PlanSegment(
                    "train",
                    t,
                    t + durations[cur2],
                    batch_index=cur2,
                    rerun=cur2 < peak,
                )
            )
            t += durations[cur2]
            cur2 += 1
        result = ([eval_seg] + tail, cp_used, new_reruns, t)
        return result

    def gap_dfs(j: int, x: int, gap_end: int, cur: int, peak: int,
                last_cp: int, cp_used: int, reruns_used: int
                ) -> Optional[tuple[list[PlanSegment], int, int, int]]:
        key = (cur, peak, last_cp, cp_used, reruns_used)
        prev = gap_memos[j].get(key)
        if prev is not None and prev <= x:
            return None  # 同进度更早就能到达，当前路径被支配
        gap_memos[j][key] = x

        if x == gap_end:
            return do_eval(j, cur, peak, last_cp, cp_used, reruns_used)

        can_train = cur < n and x + durations[cur] <= gap_end
        can_checkpoint = (
            x + CHECKPOINT_SECONDS <= gap_end
            and cp_used < max_checkpoints
            and last_cp < cur
        )

        if can_train:
            # 放得下下一批 => 该时段必须用于训练或存档（不允许空闲）。
            # 先尝试训练；训练分支整体无解时再尝试存档。
            head = PlanSegment(
                "train",
                x,
                x + durations[cur],
                batch_index=cur,
                rerun=cur < peak,
            )
            rest = gap_dfs(
                j, x + durations[cur], gap_end, cur + 1,
                max(peak, cur + 1), last_cp, cp_used, reruns_used,
            )
            if rest is not None:
                return [head] + rest[0], rest[1], rest[2], rest[3]

            if can_checkpoint:
                cp = PlanSegment(
                    "checkpoint",
                    x,
                    x + CHECKPOINT_SECONDS,
                    checkpoint_no=cp_used + 1,
                    batches_trained=cur,
                )
                rest2 = gap_dfs(
                    j, x + CHECKPOINT_SECONDS, gap_end, cur, peak, cur,
                    cp_used + 1, reruns_used,
                )
                if rest2 is not None:
                    return [cp] + rest2[0], rest2[1], rest2[2], rest2[3]
            return None

        # 放不下下一批：能存档则尝试存档，否则只能空闲到空档结束。
        if can_checkpoint:
            cp = PlanSegment(
                "checkpoint",
                x,
                x + CHECKPOINT_SECONDS,
                checkpoint_no=cp_used + 1,
                batches_trained=cur,
            )
            rest = gap_dfs(
                j, x + CHECKPOINT_SECONDS, gap_end, cur, peak, cur,
                cp_used + 1, reruns_used,
            )
            if rest is not None:
                return [cp] + rest[0], rest[1], rest[2], rest[3]

        if x < gap_end:
            idle = PlanSegment("idle", x, gap_end)
            rest = do_eval(j, cur, peak, last_cp, cp_used, reruns_used)
            if rest is not None:
                return [idle] + rest[0], rest[1], rest[2], rest[3]
        return None

    result = gap_dfs(0, 0, evals[0].start, 0, 0, 0, 0, 0)
    if result is None:
        reasons = [reason_map[j] for j in sorted(reason_map)]
        return SolveResult(
            False,
            reasons=reasons or ["不存在满足全部限制的排法"],
        )

    plan_segments, cp_used, reruns_total, makespan = result
    return SolveResult(
        True,
        plan_segments,
        checkpoints_used=cp_used,
        reruns_total=reruns_total,
        makespan=makespan,
    )
