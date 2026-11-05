"""Shared validation, batch-shape, and timing helpers for embedding methods."""

import contextlib
import warnings

import numpy as np


@contextlib.contextmanager
def stage(timer: object | None, name: str):
    """Record a timing stage when a timer is available."""
    if timer is None:
        yield
        return
    with timer.stage(name):
        yield


def synchronize_device(torch_module: object | None, device: str) -> None:
    """Wait for queued accelerator work; do nothing on CPU."""
    if torch_module is None or device == "cpu":
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if device == "cuda":
                torch_module.cuda.synchronize()
            elif device == "mps":
                torch_module.mps.synchronize()
        except Exception:
            pass


def validate_pair_arrays(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    if x0.ndim != 2 or x1.ndim != 2:
        raise ValueError(f"x0 and x1 must both be 2D arrays, got shapes {x0.shape} and {x1.shape}.")
    if x0.shape[0] == 0 or x1.shape[0] == 0:
        raise ValueError("x0 and x1 must both contain at least one sample.")
    if x0.shape[1] != x1.shape[1]:
        raise ValueError(
            "x0 and x1 must have the same number of features, "
            f"but got {x0.shape[1]} and {x1.shape[1]}."
        )
    return x0, x1


def validate_matrix(x: np.ndarray, name: str = "x") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {x.shape}.")
    if x.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if x.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    return x


def validate_batch_sizes(n0: int, n1: int) -> tuple[int, int]:
    n0 = int(n0)
    n1 = int(n1)
    if n0 <= 0 or n1 <= 0:
        raise ValueError(f"n0 and n1 must be positive, got n0={n0}, n1={n1}.")
    return n0, n1


def validate_positive_integer(value: int, name: str) -> int:
    if not float(value).is_integer():
        raise ValueError(f"{name} must be an integer, got {value}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def validate_embedding_dim(embd_dim: int, max_dim: int, method_name: str) -> int:
    embd_dim = validate_positive_integer(embd_dim, "embd_dim")
    if embd_dim > int(max_dim):
        raise ValueError(f"{method_name} embd_dim={embd_dim} exceeds max dimension {max_dim}.")
    return embd_dim


def joint_data_and_batches(
    x0: np.ndarray, x1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    x0, x1 = validate_pair_arrays(x0, x1)
    joint_data = np.concatenate([x0, x1], axis=0)
    batch_ids = np.concatenate(
        [
            np.zeros(x0.shape[0], dtype=int),
            np.ones(x1.shape[0], dtype=int),
        ]
    )
    return joint_data, batch_ids, x0.shape[0]


def split_joint_embedding(embedding: np.ndarray, n0: int) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.asarray(embedding, dtype=float)
    n0 = int(n0)
    return embedding[:n0], embedding[n0:]
