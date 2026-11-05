"""Optimal transport planning and EOT embeddings."""

import numpy as np

from ne_aligned_utils.config import DEFAULT_BACKEND
from ne_aligned_utils.methods.common import (
    stage as _stage,
    synchronize_device as _synchronize_device,
    validate_pair_arrays as _validate_pair_arrays,
    validate_positive_integer as _validate_positive_integer,
)
from ne_aligned_utils.methods.embedders import embed_2d, embedder_artifacts


#: Marginal constraints supported by ``compute_ot_plan``.
BALANCED_TRANSPORT = "balanced"
UNBALANCED_TRANSPORT = "unbalanced"
TRANSPORT_CHOICES = (BALANCED_TRANSPORT, UNBALANCED_TRANSPORT)

#: POT execution backends. Select a GPU with ``CUDA_VISIBLE_DEVICES``.
CPU_DEVICE = "cpu"
TORCH_CPU_DEVICE = "torch-cpu"
CUDA_DEVICE = "cuda"
OT_DEVICE_CHOICES = (CPU_DEVICE, TORCH_CPU_DEVICE, CUDA_DEVICE)


def _resolve_ot_device(device: str) -> tuple[str, object | None, str]:
    """Resolve the POT backend without silently falling back to another device.

    MPS is unsupported because the pipeline requires float64; float32 can underflow at small
    regularization values and change the resulting graph.
    """
    device = str(device)
    if device == CPU_DEVICE:
        return CPU_DEVICE, None, CPU_DEVICE
    if device == "mps":
        raise ValueError(
            "The OT solver does not support MPS. This pipeline is float64 and MPS cannot hold "
            "float64; float32 changes the plan completely at rel_reg 0.01 and below, where "
            "exp(-M / epsilon) underflows. Use device='cuda', or device='cpu'."
        )
    if device not in OT_DEVICE_CHOICES:
        raise ValueError(f"device must be one of {OT_DEVICE_CHOICES}, got {device!r}.")

    import torch

    if device == TORCH_CPU_DEVICE:
        return TORCH_CPU_DEVICE, torch, CPU_DEVICE
    if not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but torch reports no CUDA device. Refusing to fall back "
            "to the CPU, because the run would then be recorded as a GPU run. Use device='cpu', "
            "or device='torch-cpu' to exercise the torch backend without a GPU."
        )
    return CUDA_DEVICE, torch, CUDA_DEVICE


def _median_squared_distance(dist: object, torch_module: object | None) -> float:
    """Return NumPy's median definition on either backend.

    Unlike NumPy, ``torch.median`` returns the lower middle value for even-sized arrays.
    """
    if torch_module is None:
        return float(np.median(dist))
    flat = dist.reshape(-1)
    count = int(flat.numel())
    ordered = torch_module.sort(flat).values
    if count % 2:
        middle = ordered[count // 2]
    else:
        middle = 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])
    return float(middle)


def _validate_transport(transport: str) -> str:
    transport = str(transport)
    if transport not in TRANSPORT_CHOICES:
        raise ValueError(f"transport must be one of {TRANSPORT_CHOICES}, got {transport!r}.")
    return transport


def _resolve_reg_m(
    reg_m: float | None, rel_reg_m: float, median_squared_distance: float
) -> tuple[float, float]:
    """Resolve the absolute and relative marginal-relaxation weights."""
    if reg_m is None:
        rel_reg_m = float(rel_reg_m)
        if not (rel_reg_m > 0.0):
            raise ValueError(f"rel_reg_m must be positive, got {rel_reg_m}.")
        reg_m = median_squared_distance * rel_reg_m
    else:
        reg_m = float(reg_m)
        if not (reg_m > 0.0):
            raise ValueError(f"reg_m must be positive, got {reg_m}.")
        rel_reg_m = reg_m / median_squared_distance
    return reg_m, rel_reg_m


def _validate_unbalanced_plan(plan: np.ndarray, solver_method: str) -> None:
    """Reject unbalanced plans that lost transport structure to underflow."""
    if not np.all(np.isfinite(plan)):
        raise ValueError(
            "The unbalanced OT plan contains non-finite entries. This usually means the Sinkhorn "
            "scalings diverged; try solver_method='sinkhorn_stabilized' or a larger rel_reg."
        )
    total_mass = float(plan.sum())
    if not (total_mass > 0.0):
        raise ValueError(
            "The unbalanced OT plan has zero total mass, so the transport kernel underflowed. "
            f"Try solver_method='sinkhorn_stabilized' (the current one is {solver_method!r}) or a "
            "larger rel_reg."
        )
    # Exact zero rows or columns make their selected neighbors arbitrary.
    empty_rows = int((plan.sum(axis=1) <= 0.0).sum())
    empty_cols = int((plan.sum(axis=0) <= 0.0).sum())
    if empty_rows or empty_cols:
        raise ValueError(
            f"The unbalanced OT plan underflowed: {empty_rows} of {plan.shape[0]} rows and "
            f"{empty_cols} of {plan.shape[1]} columns are entirely zero, so those cells carry no "
            "transport information and their neighbours would be arbitrary. Try "
            f"solver_method='sinkhorn_stabilized' (the current one is {solver_method!r}) or a "
            "larger rel_reg."
        )


def compute_ot_plan(
    x0: np.ndarray,
    x1: np.ndarray,
    rel_reg: float = 1.0,
    epsilon: float | None = None,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    transport: str = BALANCED_TRANSPORT,
    rel_reg_m: float = 1.0,
    reg_m: float | None = None,
    solver_method: str = "sinkhorn",
    device: str = CPU_DEVICE,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Compute a balanced or unbalanced entropic OT plan.

    ``rel_reg`` scales epsilon by the median squared distance. For unbalanced transport,
    ``rel_reg_m`` scales the marginal penalty the same way; explicit ``epsilon`` and ``reg_m``
    values override those relative parameters. The returned plan is always a float64 NumPy array.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    import ot

    device, torch_module, torch_device = _resolve_ot_device(device)

    n0, n1 = x0.shape[0], x1.shape[0]
    a = np.ones(n0, dtype=float) / n0
    b = np.ones(n1, dtype=float) / n1
    if device != CPU_DEVICE:
        # POT selects its backend from the input array type.
        def _as_tensor(array: np.ndarray) -> object:
            return torch_module.as_tensor(array, dtype=torch_module.float64, device=torch_device)

        x0, x1, a, b = (_as_tensor(array) for array in (x0, x1, a, b))

    # Synchronization makes GPU stage timings measure execution rather than kernel launch.
    _synchronize_device(torch_module, torch_device)
    with _stage(timer, "time_ot_cost"):
        dist = ot.dist(x0, x1)
        median_squared_distance = _median_squared_distance(dist, torch_module)
        _synchronize_device(torch_module, torch_device)
    if epsilon is None:
        rel_reg = float(rel_reg)
        if rel_reg <= 0.0:
            raise ValueError(f"rel_reg must be positive, got {rel_reg}.")
        epsilon = median_squared_distance * rel_reg
    else:
        epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError(
            "The OT regularization epsilon is zero. Check that the two inputs are "
            "not degenerate, or use a larger rel_reg value."
        )

    transport = _validate_transport(transport)

    if transport == BALANCED_TRANSPORT:
        # Keep the balanced solver distinct from the unbalanced limiting case.
        with _stage(timer, "time_ot_solve"):
            plan = ot.sinkhorn(
                a,
                b,
                dist,
                reg=epsilon,
                numItermax=int(max_iter),
                stopThr=float(tol),
            )
            _synchronize_device(torch_module, torch_device)
        reg_m = None
        rel_reg_m = None
    else:
        reg_m, rel_reg_m = _resolve_reg_m(
            reg_m=reg_m, rel_reg_m=rel_reg_m, median_squared_distance=median_squared_distance
        )
        with _stage(timer, "time_ot_solve"):
            plan = ot.sinkhorn_unbalanced(
                a,
                b,
                dist,
                reg=epsilon,
                reg_m=reg_m,
                reg_type="kl",
                method=str(solver_method),
                numItermax=int(max_iter),
                stopThr=float(tol),
            )
            _synchronize_device(torch_module, torch_device)
    if torch_module is not None:
        plan = plan.detach().cpu().numpy()
    plan = np.asarray(plan, dtype=float)
    if transport == UNBALANCED_TRANSPORT:
        _validate_unbalanced_plan(plan, solver_method=str(solver_method))
    if not return_artifacts:
        return plan
    if torch_module is not None:
        dist = dist.detach().cpu().numpy()
    artifacts: dict[str, object] = {
        "cost_matrix": np.asarray(dist, dtype=float),
        "epsilon": float(epsilon),
        "median_squared_distance": float(median_squared_distance),
        "transport": transport,
        # Record the selected POT device, including torch-cpu.
        "ot_device": device,
    }
    if transport == UNBALANCED_TRANSPORT:
        artifacts.update(
            {
                "reg_m": float(reg_m),
                "rel_reg_m": float(rel_reg_m),
                "plan_total_mass": float(plan.sum()),
                "plan_row_mass": plan.sum(axis=1),
                "plan_col_mass": plan.sum(axis=0),
                # Distinguishes a uniformly small plan from partial underflow.
                "plan_nnz_fraction": float((plan > 0.0).mean()),
            }
        )
    return plan, artifacts


def compute_eot_embedding_from_plan(
    plan: np.ndarray,
    t: float = 0.0,
    embd_dim: int = 2,
    start: int = 1,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Stack batch-scaled singular coordinates, weighted by singular values to power ``t``.

    ``start`` skips leading modes (one by default); the output dimension is capped
    by the available modes. Rows follow batch 0, then batch 1.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")

    t = float(t)
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t}.")

    n0, n1 = plan.shape
    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    max_dim = min(u.shape[1], vt.shape[0]) - start
    if max_dim <= 0:
        raise ValueError(
            "EOT needs at least one non-trivial singular vector after dropping the "
            "leading transport singular mode."
        )

    embd_dim = _validate_positive_integer(embd_dim, "embd_dim")
    q = min(embd_dim, max_dim)
    stop = start + q
    weights = singular_values[start:stop] ** t
    left_embedding = u[:, start:stop] * weights * np.sqrt(n0)
    right_embedding = vt.T[:, start:stop] * weights * np.sqrt(n1)
    joint_embedding = np.concatenate([left_embedding, right_embedding], axis=0)

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "u": u,
        "vt": vt,
        "singular_values": singular_values,
        "t": float(t),
        "embd_dim": int(q),
        "start": int(start),
    }
    return joint_embedding, artifacts


def compute_eot_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    t: float = 0.0,
    rel_reg: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    embd_dim: int = 2,
    device: str = CPU_DEVICE,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Compute a balanced OT plan and return its stacked singular coordinates."""
    x0, x1 = _validate_pair_arrays(x0, x1)
    # EOT uses only plan solving and SVD, so no backend embedding stage is timed.
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
        with _stage(timer, "time_ot_postprocess"):
            joint_embedding, eot_artifacts = compute_eot_embedding_from_plan(
                plan,
                t=t,
                embd_dim=embd_dim,
                return_artifacts=True,
            )

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        **eot_artifacts,
        "ot_plan": plan,
        "epsilon": ot_artifacts["epsilon"],
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        # The task runner reads torch_device to keep runtime comparisons device-specific.
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
    }
    return joint_embedding, artifacts


def compute_eot_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    t: float = 0.0,
    rel_reg: float = 1.0,
    eot_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    device: str = CPU_DEVICE,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed EOT coordinates in 2D with the selected backend."""
    x0, x1 = _validate_pair_arrays(x0, x1)
    # This backend receives EOT coordinates and builds its own affinities.
    eot_embedding, eot_artifacts = compute_eot_embedding_from_arrays(
        x0,
        x1,
        t=t,
        rel_reg=rel_reg,
        tol=tol,
        max_iter=max_iter,
        embd_dim=eot_dim,
        device=device,
        return_artifacts=True,
        timer=timer,
    )
    if eot_embedding.shape[1] < 2:
        raise ValueError(
            "EOT+TSNE needs at least two non-trivial EOT dimensions before t-SNE, "
            f"but only {eot_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = embed_2d(
        eot_embedding,
        backend,
        random_state=random_state,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        return_artifacts=True,
        timer=timer,
    )

    if not return_artifacts:
        return tsne_embedding

    artifacts: dict[str, object] = {
        "left_embedding": tsne_embedding[: x0.shape[0]],
        "right_embedding": tsne_embedding[x0.shape[0] :],
        "eot_embedding": eot_embedding,
        "eot_left_embedding": eot_artifacts["left_embedding"],
        "eot_right_embedding": eot_artifacts["right_embedding"],
        "ot_plan": eot_artifacts["ot_plan"],
        "singular_values": eot_artifacts["singular_values"],
        "epsilon": eot_artifacts["epsilon"],
        "median_squared_distance": eot_artifacts["median_squared_distance"],
        "t": eot_artifacts["t"],
        "eot_dim": int(eot_dim),
        "eot_dim_used": int(eot_artifacts["embd_dim"]),
        **embedder_artifacts(backend, tsne_artifacts),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        # The task runner reads torch_device to record where the plan was solved.
        "ot_device": eot_artifacts["ot_device"],
        "torch_device": eot_artifacts["torch_device"],
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts
