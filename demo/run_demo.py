#!/usr/bin/env python3
"""端到端演示脚本：提交跨两场评测、含重跑的排程并推进模拟时钟。

用法：
    python demo/run_demo.py            # 默认对接 http://127.0.0.1:8000
    BASE_URL=http://host:port python demo/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")


def call(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> int:
    plan = {
        # 8 批，每批 3 秒
        "durations": [3] * 8,
        "evaluations": [
            {"eval_id": 1, "start": 12, "end": 14,
             "min_checkpoint_batches": 2},
            {"eval_id": 2, "start": 30, "end": 32,
             "min_checkpoint_batches": 3},
        ],
        # 只允许 1 个检查点 => 第二场评测复用，跨场重跑 5 批
        "max_checkpoints": 1,
        "max_rerun_batches": 20,
    }
    print(f"==> 提交排程到 {BASE_URL}/plans")
    status, body = call("POST", "/plans", plan)
    if status != 201:
        print("    无解，演练未启动：")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 1

    sid = body["session_id"]
    print(f"    已受理 session={sid}，检查点 {body['checkpoints_used']} 个，"
          f"计划重跑 {body['reruns_total']} 批，总时长 {body['makespan']} 秒")

    # 每次推进 5 个模拟秒，直到完成。
    while True:
        status, body = call("POST", f"/sessions/{sid}/advance?seconds=5")
        assert status == 200, body
        for ev in body["advanced_events"]:
            line = f"    [{ev['start']:>3}-{ev['end']:>3}] {ev['kind']}"
            if ev["kind"] == "train":
                line += f" 批次{ev['batch_index']}" + ("（重跑）" if ev["rerun"] else "")
            elif ev["kind"] == "checkpoint":
                line += f" 覆盖{ev['batches_trained']}批 -> {ev.get('checkpoint_path','')}"
            elif ev["kind"] == "evaluation":
                inf = ev["inference"]
                line += (f" 用检查点{inf['used_checkpoint_no']}"
                         f"（{inf['used_checkpoint_batches']}批）推理 "
                         f"MSE={inf['mse']:.6f}")
            print(line)
        if body["finished"]:
            break

    status, results = call("GET", f"/sessions/{sid}/results")
    print("==> 最终结果：")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    assert results["consistent_with_uninterrupted"] is True
    print("==> 最终权重与不间断训练逐元素一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
