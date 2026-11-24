"""排程求解器规则测试。"""

import pytest

from app.scheduler import (
    CHECKPOINT_SECONDS,
    EvaluationSpec,
    solve,
)


def kinds(segments):
    return [s.kind for s in segments]


def test_no_evaluation_trains_everything_back_to_back():
    res = solve([3, 5, 2], [], max_checkpoints=2, max_rerun_batches=0)
    assert res.feasible
    assert kinds(res.segments) == ["train", "train", "train"]
    assert res.makespan == 10
    assert res.checkpoints_used == 0
    assert res.reruns_total == 0


def test_checkpoint_must_be_written_before_first_evaluation():
    # 5 批，每批 2 秒；第 20 秒评测，要求至少 3 批存档。
    ev = EvaluationSpec(1, start=20, end=22, min_checkpoint_batches=3)
    # 存档预算为 0 => 无解。
    res = solve([2] * 5, [ev], max_checkpoints=0, max_rerun_batches=10)
    assert not res.feasible
    assert any("低于要求" in r for r in res.reasons)

    res = solve([2] * 5, [ev], max_checkpoints=1, max_rerun_batches=10)
    assert res.feasible
    cp = [s for s in res.segments if s.kind == "checkpoint"]
    assert len(cp) == 1
    assert cp[0].batches_trained >= 3
    # 存档独占 2 秒且不与任何时段重叠。
    assert cp[0].end - cp[0].start == CHECKPOINT_SECONDS


def test_no_overlap_and_evaluation_on_time():
    ev = EvaluationSpec(7, start=10, end=13, min_checkpoint_batches=1)
    res = solve([4, 4, 4], [ev], max_checkpoints=2, max_rerun_batches=5)
    assert res.feasible
    eval_seg = next(s for s in res.segments if s.kind == "evaluation")
    assert (eval_seg.start, eval_seg.end) == (10, 13)
    # 所有时段相邻不重叠、按时间排序。
    timeline = sorted(res.segments, key=lambda s: s.start)
    for a, b in zip(timeline, timeline[1:]):
        assert a.end <= b.start


def test_rerun_budget_enforced():
    # 两场评测只给 1 个检查点 => 第二场复用第一场的检查点，
    # 两场之间训练的批次在第二场后必须重跑。
    evals = [
        EvaluationSpec(1, start=10, end=12, min_checkpoint_batches=4),
        EvaluationSpec(2, start=16, end=18, min_checkpoint_batches=4),
    ]
    # 第一场前训 4 批、存档；两场间再训 2 批；第二场后这 2 批重跑。
    res = solve([2] * 8, evals, max_checkpoints=1, max_rerun_batches=1)
    assert not res.feasible
    assert any("重跑" in r for r in res.reasons)

    res = solve([2] * 8, evals, max_checkpoints=1, max_rerun_batches=2)
    assert res.feasible
    assert res.reruns_total == 2

    # 给足检查点预算时，每场评测前存档 => 无需重跑。
    res = solve([2] * 8, evals, max_checkpoints=2, max_rerun_batches=0)
    assert res.feasible
    assert res.reruns_total == 0


def test_no_idle_when_batch_fits():
    # 空档 0..11，批 3 秒、存档 2 秒：训 3 批到 9 秒，尾段 2 秒
    # 下一批放不下但放得下存档 => 必须存档，0..11 全程无空闲。
    ev = EvaluationSpec(3, start=11, end=13, min_checkpoint_batches=1)
    res = solve([3, 3, 3, 3], [ev], max_checkpoints=2, max_rerun_batches=4)
    assert res.feasible
    pre = [s for s in res.segments if s.start < 11]
    assert all(s.kind != "idle" for s in pre)
    assert pre[-1].end == 11
    assert pre[-1].kind == "checkpoint"


def test_idle_allowed_only_when_nothing_fits():
    # 空档 0..9，批 5 秒、存档 2 秒；训一批后尾段 4 秒：下一批放不下，
    # 存档放得下 => 存档（5..7），剩 2 秒仍可... 重复存档被禁止，
    # 故 7..9 空闲（此时训练与有效存档都放不下）。
    ev = EvaluationSpec(4, start=9, end=11, min_checkpoint_batches=1)
    res = solve([5, 5], [ev], max_checkpoints=1, max_rerun_batches=2)
    assert res.feasible
    idle = [s for s in res.segments if s.kind == "idle"]
    assert idle, "尾段应出现空闲"
    seg = idle[0]
    # 空闲时长 < 单批耗时 => 训练放不下；存档虽物理放得下，但当前进度
    # 已有同内容检查点，重复存档被禁止 => 不存在任何有效动作。
    assert seg.end - seg.start < 5
    last_cp = max(
        s.batches_trained for s in res.segments if s.kind == "checkpoint"
    )
    assert last_cp == 1  # 空闲前进度已被检查点覆盖，无法再有效存档


def test_two_evaluations_cross_rerun_accounting():
    evals = [
        EvaluationSpec(1, start=12, end=14, min_checkpoint_batches=2),
        EvaluationSpec(2, start=30, end=32, min_checkpoint_batches=3),
    ]
    # 只给 1 个检查点 => 第二场评测复用第一场的检查点，跨场重跑。
    res = solve([3] * 8, evals, max_checkpoints=1, max_rerun_batches=10)
    assert res.feasible
    eval_segs = [s for s in res.segments if s.kind == "evaluation"]
    assert [e.start for e in eval_segs] == [12, 30]
    assert res.reruns_total >= 1
    # 两场评测引用同一个检查点序号。
    assert eval_segs[0].checkpoint_no == eval_segs[1].checkpoint_no == 1


def test_rerun_marking_matches_rollback():
    # 单检查点 + 两场评测 => 第二场复用检查点并产生带 rerun 标记的批次。
    evals = [
        EvaluationSpec(5, start=10, end=12, min_checkpoint_batches=1),
        EvaluationSpec(6, start=18, end=20, min_checkpoint_batches=1),
    ]
    res = solve([2] * 6, evals, max_checkpoints=1, max_rerun_batches=10)
    assert res.feasible
    second_eval = next(s for s in res.segments
                       if s.kind == "evaluation" and s.eval_id == 6)
    reruns = [s for s in res.segments
              if s.kind == "train" and s.rerun]
    cp_batches = second_eval.batches_trained
    pre_eval_trained = max(
        s.batch_index + 1 for s in res.segments
        if s.kind == "train" and s.end <= second_eval.start
    )
    assert len(reruns) == pre_eval_trained - cp_batches
    assert all(s.batch_index < pre_eval_trained for s in reruns)


def test_checkpoint_budget_respected():
    ev = EvaluationSpec(6, start=20, end=22, min_checkpoint_batches=2)
    res = solve([2] * 6, [ev], max_checkpoints=2, max_rerun_batches=10)
    assert res.feasible
    assert sum(1 for s in res.segments if s.kind == "checkpoint") <= 2


def test_max_scale_runs_and_saturates_gap():
    # 24 批、4 场评测的最大规模，求解必须快速完成且排程无重叠。
    evals = [
        EvaluationSpec(1, 20, 22, 2),
        EvaluationSpec(2, 45, 47, 4),
        EvaluationSpec(3, 75, 77, 6),
        EvaluationSpec(4, 110, 112, 8),
    ]
    res = solve([3] * 24, evals, max_checkpoints=4, max_rerun_batches=40)
    assert res.feasible
    timeline = sorted(res.segments, key=lambda s: s.start)
    for a, b in zip(timeline, timeline[1:]):
        assert a.end <= b.start
    # 训练覆盖全部 24 批（含重跑）。
    trained = {s.batch_index for s in timeline if s.kind == "train"}
    assert trained == set(range(24))
