# 单卡夜班训练与定时评测演练（FastAPI + NumPy）

夜班训练与定时模型评测共用一张 GPU。本服务在本地用**确定性的逐批训练样例**
模拟这一资源争用问题：提交批次耗时、评测窗口与预算，服务先做约束求解，
**有解才受理并启动模拟，无解不启动**；随后通过推进模拟时钟，实跑训练、
检查点写入与评测推理，并可核对最终权重与“不间断训练”完全一致。

## 规则建模

| 规则 | 实现 |
| --- | --- |
| 训练批次不可拆、不可抢占 | 每个训练时段长度固定为该批耗时，不可跨评测窗口 |
| 写本地检查点独占卡 2 秒 | `checkpoint` 时段固定 2 秒，真实写入 `.npz` 文件 |
| 评测准点独占卡 | `evaluation` 时段严格占用请求的 `[start, end)` |
| 评测用最近写完的检查点实际推理 | 评测时从磁盘加载最近检查点权重，在固定评测集上算预测与 MSE |
| 评测后从同一检查点续训 | 权重与训练进度回滚到该检查点，之后的批次标记为重跑 |
| 空档放得下下一批时须用来训练或存档 | 求解器在仍可行动时禁止空闲；仅训练/有效存档都放不下时允许空闲 |
| 检查点写入次数 / 累计重跑批数上限 | 求解器将预算作为硬约束，超限即判无解 |
| 有排法才受理，无解不启动 | `POST /plans` 无解返回 422 且不创建任何会话或目录 |

确定性保证：每批数据由固定种子按批次索引独立生成，权重更新是纯 NumPy
闭式梯度计算，无随机性。因此同一点重跑同一批与首跑逐元素相同，最终权重
必然等于不间断训练的参考权重（接口返回 `max_abs_diff = 0.0`）。

## 目录结构

```
app/
  ml.py          确定性 NumPy 训练、推理、检查点读写
  scheduler.py   单卡排程约束求解器（记忆化 DFS + 支配剪枝）
  simulator.py   模拟时钟、训练/存档/评测的实际副作用与回滚
  schemas.py     Pydantic 入参校验（24 批 / 4 场评测等上限）
  main.py        FastAPI 路由
tests/           pytest 测试（21 个）
demo/run_demo.py 纯标准库端到端演示脚本
Dockerfile, docker-compose.yml
```

## 启动方式

### Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000/docs
```

检查点文件写入容器内 `/data`（命名卷 `gpu_drill_data`）。

### 本地 Python

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 检查点默认写入 ./sessions_data，可用环境变量 APP_DATA_DIR 覆盖
```

## API 一览（详见 /docs）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/plans` | 提交排程；有解 201 创建会话，无解 422 不启动 |
| GET | `/sessions` | 列出全部会话及当前时钟 |
| GET | `/sessions/{id}` | 查询已执行的训练/存档/评测时段、重跑批数 |
| POST | `/sessions/{id}/advance?seconds=N` | 推进模拟时钟 N 秒（时段不可拆，跑不完不产生效果） |
| GET | `/sessions/{id}/results` | 最终权重、参考权重、最大绝对差、一致性结论 |
| GET | `/health` | 健康检查 |

### 请求示例

```json
{
  "durations": [3, 3, 3, 3, 3, 3, 3, 3],
  "evaluations": [
    {"eval_id": 1, "start": 12, "end": 14, "min_checkpoint_batches": 2},
    {"eval_id": 2, "start": 30, "end": 32, "min_checkpoint_batches": 3}
  ],
  "max_checkpoints": 1,
  "max_rerun_batches": 20
}
```

约束：`durations` 1–24 个正整数；`evaluations` 0–4 场，两两不重叠，
`end > start`，`eval_id` 唯一，最低存档批次数不超过总批数。

## 演示与测试

```bash
python demo/run_demo.py           # 需先启动服务
python -m pytest tests/ -q        # 21 个测试
```

## 求解器说明

评测窗口把时间线切成有限空档。DFS 在每个空档枚举“训练 / 存档 / 空闲”
决策，评测点检查最近检查点的最低批次要求并计算回滚重跑量；同一空档内
相同训练进度只保留最早到达的路径（支配剪枝）。24 批 × 4 场评测规模下
求解为亚毫秒级。
