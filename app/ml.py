"""确定性的 NumPy 微型线性回归训练/推理逻辑。

所有数据由固定种子按批次生成，任何一次重跑同一批次得到的权重更新
与首次完全一致，从而保证“被评测打断后重跑”的最终权重与不间断训练
逐元素相同。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# 固定的全局随机种子，保证每次启动服务得到完全一致的批次数据。
SEED = 20261124
INPUT_DIM = 4
# 每个批次的样本数。
BATCH_SIZE = 8
# 线性回归的学习率。
LEARNING_RATE = 0.05

_WEIGHT_DTYPE = np.float64


def rng_for_batch(batch_index: int) -> np.random.Generator:
    """每个批次使用独立、确定的随机流，重跑与首跑完全一致。"""
    return np.random.default_rng(SEED + batch_index * 7919)


def make_batch(batch_index: int) -> tuple[np.ndarray, np.ndarray]:
    """生成第 ``batch_index`` 个训练批次（0 基），返回 (X, y)。"""
    rng = rng_for_batch(batch_index)
    x = rng.normal(size=(BATCH_SIZE, INPUT_DIM))
    true_w = np.linspace(1.0, 2.0, INPUT_DIM)
    noise = rng.normal(scale=0.01, size=BATCH_SIZE)
    y = x @ true_w + 0.5 + noise
    return x, y


def initial_weights() -> np.ndarray:
    """固定的初始权重（含偏置项）。"""
    rng = np.random.default_rng(SEED)
    return rng.normal(scale=0.1, size=INPUT_DIM + 1).astype(_WEIGHT_DTYPE).copy()


def _augment(x: np.ndarray) -> np.ndarray:
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.hstack([x, ones])


def train_one_batch(weights: np.ndarray, batch_index: int) -> np.ndarray:
    """对一批样本做一次全批梯度下降，返回更新后的权重（不修改入参）。

    损失为均方误差，梯度按闭式公式计算，纯 NumPy 运算且无随机性，
    因此相同权重 + 相同批次必然得到相同结果。
    """
    x, y = make_batch(batch_index)
    phi = _augment(x)
    pred = phi @ weights
    grad = (2.0 / x.shape[0]) * (phi.T @ (pred - y))
    return (weights - LEARNING_RATE * grad).astype(_WEIGHT_DTYPE, copy=True)


def predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _augment(x) @ weights


def evaluation_dataset() -> tuple[np.ndarray, np.ndarray]:
    """评测专用的固定数据集，所有场次使用同一数据集便于横向比较。"""
    rng = np.random.default_rng(SEED ^ 0x5EED)
    x = rng.normal(size=(16, INPUT_DIM))
    true_w = np.linspace(1.0, 2.0, INPUT_DIM)
    y = x @ true_w + 0.5 + rng.normal(scale=0.01, size=16)
    return x, y


def run_inference(weights: np.ndarray) -> dict:
    """用给定权重在固定评测集上推理，返回预测值与 MSE。"""
    x, y = evaluation_dataset()
    preds = predict(weights, x)
    mse = float(np.mean((preds - y) ** 2))
    return {
        "mse": mse,
        "predictions": [float(v) for v in preds],
        "targets": [float(v) for v in y],
    }


def reference_uninterrupted(num_batches: int) -> np.ndarray:
    """不间断训练全部批次得到的参考权重（用于一致性校验）。"""
    w = initial_weights()
    for b in range(num_batches):
        w = train_one_batch(w, b)
    return w


@dataclass
class Checkpoint:
    """内存中的检查点状态；同时落盘一份 .npz 作为本地检查点证据。"""

    checkpoint_id: int
    batches_trained: int
    weights: np.ndarray
    path: str

    def to_summary(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "batches_trained": self.batches_trained,
            "path": self.path,
        }


def save_checkpoint(
    directory: str, checkpoint_id: int, batches_trained: int, weights: np.ndarray
) -> Checkpoint:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"checkpoint_{checkpoint_id:03d}.npz")
    np.savez(
        path,
        weights=weights,
        batches_trained=np.int64(batches_trained),
        checkpoint_id=np.int64(checkpoint_id),
    )
    return Checkpoint(checkpoint_id, batches_trained, weights.copy(), path)


def load_checkpoint(path: str) -> tuple[np.ndarray, int]:
    with np.load(path) as data:
        return data["weights"].copy(), int(data["batches_trained"])
