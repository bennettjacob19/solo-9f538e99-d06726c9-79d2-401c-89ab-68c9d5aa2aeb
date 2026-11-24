"""用独立的暴力枚举器交叉验证记忆化 DFS 求解器的结论。

暴力枚举器不共享求解器代码：直接枚举每个空档的全部合法动作序列，
按评测点累计回滚重跑，检查所有预算。仅在小规模（<=6 批、<=2 场）运行。
"""

import itertools
import random

from app.scheduler import (
    CHECKPOINT_SECONDS,
    EvaluationSpec,
    solve,
)


def brute_force(durations, evals, max_cp, max_rerun):
    """返回 (feasible, min_reruns, min_checkpoints)。"""
    n = len(durations)
    evals = sorted(evals, key=lambda e: e.start)
    best = None  # (reruns, cps)

    def fill_gap(x, gap_end, cur, peak, last_cp, cp_used, reruns_used,
                 pending_evals, history):
        nonlocal best
        if not pending_evals:
            # 末段顺序训完，不再存档。
            t = x
            c = cur
            while c < n:
                t += durations[c]
                c += 1
            best = (reruns_used, cp_used) if best is None else min(
                best, (reruns_used, cp_used)
            )
            return

        spec = pending_evals[0]
        if x == gap_end:
            # 评测点：检查最低存档批次并回滚。
            if last_cp < spec.min_checkpoint_batches:
                return
            rolled = cur - last_cp
            if reruns_used + rolled > max_rerun:
                return
            rest = pending_evals[1:]
            next_end = rest[0].start if rest else None
            if rest:
                fill_gap(spec.end, next_end, last_cp, peak, last_cp,
                         cp_used, reruns_used + rolled, rest, history)
            else:
                fill_gap(spec.end, None, last_cp, peak, last_cp, cp_used,
                         reruns_used + rolled, [], history)
            return

        can_train = cur < n and x + durations[cur] <= gap_end
        can_cp = (x + CHECKPOINT_SECONDS <= gap_end
                  and cp_used < max_cp and last_cp < cur)

        if can_train:
            fill_gap(x + durations[cur], gap_end, cur + 1,
                     max(peak, cur + 1), last_cp, cp_used, reruns_used,
                     pending_evals, history)
            if can_cp:
                fill_gap(x + CHECKPOINT_SECONDS, gap_end, cur, peak, cur,
                         cp_used + 1, reruns_used, pending_evals, history)
            return
        if can_cp:
            fill_gap(x + CHECKPOINT_SECONDS, gap_end, cur, peak, cur,
                     cp_used + 1, reruns_used, pending_evals, history)
        # 空闲到空档结束（仅在没有可行动作时合法）。
        fill_gap(gap_end, gap_end, cur, peak, last_cp, cp_used,
                 reruns_used, pending_evals, history)

    fill_gap(0, evals[0].start, 0, 0, 0, 0, 0, evals, [])
    return (best is not None, *best) if best is not None else (False, None, None)


def test_brute_force_agreement_random_small_cases():
    rng = random.Random(7)
    checked = 0
    for _ in range(120):
        n = rng.randint(1, 6)
        durs = [rng.randint(1, 5) for _ in range(n)]
        ne = rng.randint(1, 2)
        t = rng.randint(2, 12)
        evs = []
        for k in range(ne):
            length = rng.randint(1, 3)
            evs.append(EvaluationSpec(
                k + 1, t, t + length, rng.randint(1, n)
            ))
            t += length + rng.randint(1, 12)
        cp_lim = rng.randint(0, 3)
        rr_lim = rng.randint(0, 8)

        bf = brute_force(durs, evs, cp_lim, rr_lim)
        res = solve(durs, evs, cp_lim, rr_lim)
        assert res.feasible == bf[0], (durs, evs, cp_lim, rr_lim, bf)
        if res.feasible:
            # DFS 解的预算必须不劣于暴力最优。
            assert res.reruns_total >= (bf[1] or 0)
            assert res.checkpoints_used <= cp_lim
            assert res.reruns_total <= rr_lim
        checked += 1
    assert checked == 120


def test_brute_force_agreement_handcrafted():
    cases = [
        ([3, 3, 3, 3], [EvaluationSpec(1, 11, 13, 1)], 2, 4),
        ([2] * 8, [
            EvaluationSpec(1, 10, 12, 4),
            EvaluationSpec(2, 16, 18, 4),
        ], 1, 1),
        ([2] * 8, [
            EvaluationSpec(1, 10, 12, 4),
            EvaluationSpec(2, 16, 18, 4),
        ], 1, 2),
        ([5, 5], [EvaluationSpec(1, 9, 11, 1)], 1, 2),
        ([4, 4, 4], [EvaluationSpec(1, 10, 13, 1)], 2, 5),
    ]
    for durs, evs, cp_lim, rr_lim in cases:
        bf = brute_force(durs, evs, cp_lim, rr_lim)
        res = solve(durs, evs, cp_lim, rr_lim)
        assert res.feasible == bf[0], (durs, evs, cp_lim, rr_lim)
