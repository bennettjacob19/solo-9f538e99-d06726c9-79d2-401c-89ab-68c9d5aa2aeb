"""模拟时钟、检查点落盘、评测回滚与最终权重一致性测试。"""

import numpy as np
import pytest

from app import ml
from app.scheduler import EvaluationSpec, solve
from app.simulator import Session


def make_session(tmp_path, durations, evals, cp_limit, rerun_limit):
    res = solve(durations, evals, cp_limit, rerun_limit)
    assert res.feasible
    return Session(
        session_id="test",
        plan=res,
        data_dir=str(tmp_path),
        num_batches=len(durations),
    )


def test_partial_advance_does_not_split_segment(tmp_path):
    session = make_session(
        tmp_path, [4, 4], [], cp_limit=0, rerun_limit=0
    )
    out = session.advance(3)  # 第一批 4 秒，跑不完
    assert out["clock"] == 3
    assert out["advanced_events"] == []
    out = session.advance(1)  # 恰好在边界完成
    assert out["clock"] == 4
    assert len(out["advanced_events"]) == 1
    assert out["advanced_events"][0]["kind"] == "train"


def test_full_replay_finishes_and_matches_reference(tmp_path):
    evals = [
        EvaluationSpec(1, start=12, end=14, min_checkpoint_batches=2),
        EvaluationSpec(2, start=30, end=32, min_checkpoint_batches=3),
    ]
    # 只给 1 个检查点，强制第二场评测复用并产生跨场重跑。
    session = make_session(
        tmp_path, [3] * 8, evals, cp_limit=1, rerun_limit=20
    )
    # 一秒一秒推进，覆盖所有部分推进路径。
    while not session.finished:
        session.advance(1)

    res = session.results()
    assert res["finished"]
    assert res["consistent_with_uninterrupted"] is True
    assert res["max_abs_diff"] == 0.0
    assert res["reruns_executed"] == session.plan.reruns_total
    assert res["checkpoints_written"] == session.plan.checkpoints_used


def test_evaluation_uses_latest_checkpoint_and_rolls_back(tmp_path):
    ev = EvaluationSpec(1, start=10, end=12, min_checkpoint_batches=1)
    session = make_session(
        tmp_path, [2] * 5, [ev], cp_limit=1, rerun_limit=10
    )
    # 直接推进到评测结束之后。
    session.advance(13)
    status = session.status()
    eval_event = next(
        e for e in status["executed_segments"] if e["kind"] == "evaluation"
    )
    assert eval_event["inference"]["used_checkpoint_no"] == 1
    assert eval_event["inference"]["used_checkpoint_batches"] >= 1
    assert "mse" in eval_event["inference"]
    assert len(eval_event["inference"]["predictions"]) == 16
    # 回滚后训练进度等于检查点覆盖批次数。
    assert status["trained_cursor"] == eval_event["inference"][
        "used_checkpoint_batches"
    ]
    # 权重已回到检查点权重。
    ckpt = session.checkpoint_store[1]
    assert np.array_equal(session.current_weights, ckpt.weights)


def test_checkpoint_files_are_written_and_reloadable(tmp_path):
    ev = EvaluationSpec(1, start=10, end=12, min_checkpoint_batches=1)
    session = make_session(
        tmp_path, [2] * 4, [ev], cp_limit=2, rerun_limit=10
    )
    session.advance(10)
    for ckpt in session.checkpoint_store.values():
        w, batches = ml.load_checkpoint(ckpt.path)
        assert np.array_equal(w, ckpt.weights)
        assert batches == ckpt.batches_trained


def test_advance_over_makespan_boundary(tmp_path):
    # 两批各 4 秒，makespan=8；推进 6 秒（第一批完成、第二批跑不完），
    # 再推进 1 秒仍跑不完，最后推进 1 秒恰好完成。
    session = make_session(
        tmp_path, [4, 4], [], cp_limit=0, rerun_limit=0
    )
    out = session.advance(6)
    assert out["clock"] == 6 and len(out["advanced_events"]) == 1
    out = session.advance(1)
    assert out["clock"] == 7 and out["advanced_events"] == []
    out = session.advance(1)
    assert out["finished"] and out["clock"] == 8
    # 完成后再推进保持幂等。
    out = session.advance(10)
    assert out["clock"] == 8 and out["advanced_events"] == []


def test_rerun_batch_reproduces_identical_update(tmp_path):
    # 确定性：同一权重 + 同一批次数据，重跑更新与首跑完全一致。
    w0 = ml.initial_weights()
    w1 = ml.train_one_batch(w0, 0)
    w1_again = ml.train_one_batch(w0, 0)
    assert np.array_equal(w1, w1_again)
    # 参考权重与逐批串接结果一致。
    ref = ml.reference_uninterrupted(6)
    w = w0
    for b in range(6):
        w = ml.train_one_batch(w, b)
    assert np.array_equal(w, ref)


def test_cross_evaluation_reruns_are_counted(tmp_path):
    evals = [
        EvaluationSpec(1, 10, 12, 1),
        EvaluationSpec(2, 26, 28, 2),
    ]
    # 单检查点 => 第二场复用，两场之间训练的批次全部重跑。
    session = make_session(
        tmp_path, [2] * 7, evals, cp_limit=1, rerun_limit=20
    )
    assert session.plan.reruns_total >= 1
    while not session.finished:
        session.advance(2)
    executed_reruns = [
        e for e in session.events
        if e.segment.kind == "train" and e.segment.rerun
    ]
    assert len(executed_reruns) == session.reruns_executed
    assert session.reruns_executed == session.plan.reruns_total
    assert session.results()["consistent_with_uninterrupted"] is True
