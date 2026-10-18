"""Size constrained pairwise exchange optimizer for patient assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def target_counts(n: int, ratios: np.ndarray) -> np.ndarray:
    """Largest remainder rounding of ``n * ratios`` with no empty partition."""
    if n < len(ratios):
        raise ValueError(f"Need at least {len(ratios)} patients for {len(ratios)} partitions")
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts), kind="stable")[: n - int(counts.sum())]:
        counts[index] += 1
    while (counts == 0).any():
        counts[int(np.argmax(counts))] -= 1
        counts[int(np.argmin(counts))] += 1
    return counts


@dataclass
class ExchangeOptimizer:
    """Minimize the weighted squared standardized difference between each
    partition and the pooled cohort by exchanging patients between partitions.

    Every patient ``g`` has a standardized residual vector
    ``u_g = sqrt(w) * (num_g - pooled_mean * den_g) / pooled_sd``. For a
    partition with residual sum ``U_s`` and expected denominator ``c_s`` the
    loss is ``sum_s ||U_s / c_s||^2``. Partition sizes (in patients) are fixed,
    so the search uses exchanges only.
    """

    ratios: np.ndarray
    seed: int = 0
    max_swaps: int = 20000
    block: int = 384
    patience: int = 12
    restarts: int = 1

    def fit(
        self,
        num: np.ndarray,
        den: np.ndarray,
        pooled_mean: np.ndarray,
        pooled_sd: np.ndarray,
        weights: np.ndarray,
        fixed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        n = num.shape[0]
        k = len(self.ratios)
        counts = target_counts(n, np.asarray(self.ratios, dtype=float))
        scale = np.sqrt(weights) / pooled_sd
        u = (num - pooled_mean * den) * scale
        # expected denominators per partition and term
        c = np.outer(counts / n, den.sum(axis=0))
        c[c <= 0] = 1.0
        fixed = np.full(n, -1) if fixed is None else np.asarray(fixed)
        if (np.bincount(fixed[fixed >= 0], minlength=k) > counts).any():
            # honour fixed assignments even when they exceed the requested share
            counts = self._counts_with_fixed(n, counts, fixed, k)
            c = np.outer(counts / n, den.sum(axis=0))
            c[c <= 0] = 1.0
        best: tuple[float, np.ndarray, dict[str, float]] | None = None
        rng = np.random.default_rng(self.seed)
        for _ in range(max(self.restarts, 1)):
            assign = self._initial(rng, n, counts, fixed)
            assign, info = self._search(rng, u, c, assign, fixed, k)
            if best is None or info["loss"] < best[0]:
                best = (info["loss"], assign, info)
        assert best is not None
        return best[1], best[2]

    @staticmethod
    def _counts_with_fixed(n: int, counts: np.ndarray, fixed: np.ndarray, k: int) -> np.ndarray:
        have = np.bincount(fixed[fixed >= 0], minlength=k)
        counts = np.maximum(counts, have)
        while counts.sum() > n:
            slack = counts - have
            counts[int(np.argmax(slack))] -= 1
        return counts

    @staticmethod
    def _initial(
        rng: np.random.Generator, n: int, counts: np.ndarray, fixed: np.ndarray
    ) -> np.ndarray:
        assign = fixed.copy()
        free = np.flatnonzero(fixed < 0)
        remaining = counts - np.bincount(fixed[fixed >= 0], minlength=len(counts))
        labels = np.repeat(np.arange(len(counts)), remaining)
        assign[rng.permutation(free)] = labels
        return assign

    def _search(
        self,
        rng: np.random.Generator,
        u: np.ndarray,
        c: np.ndarray,
        assign: np.ndarray,
        fixed: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        inv = 1.0 / (c * c)  # (k, d)
        sums = np.vstack([u[assign == s].sum(axis=0) for s in range(k)])
        movable = fixed < 0
        members = [np.flatnonzero((assign == s) & movable) for s in range(k)]
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        initial = float((sums * sums * inv).sum())
        swaps = 0
        stale = 0
        turn = 0
        limit = self.patience * len(pairs)
        while swaps < self.max_swaps and stale < limit:
            a, b = pairs[turn % len(pairs)]
            turn += 1
            ia, ib = members[a], members[b]
            if len(ia) == 0 or len(ib) == 0:
                stale += 1
                continue
            exhaustive = len(ia) <= self.block and len(ib) <= self.block
            pa = np.arange(len(ia)) if len(ia) <= self.block else rng.choice(
                len(ia), self.block, replace=False
            )
            pb = np.arange(len(ib)) if len(ib) <= self.block else rng.choice(
                len(ib), self.block, replace=False
            )
            xa, xb = u[ia[pa]], u[ib[pb]]
            wa, wb = inv[a], inv[b]
            wsum = wa + wb
            # delta = xb[j] - xa[i]; change in loss for exchanging i and j
            lin = 2.0 * (sums[a] * wa - sums[b] * wb)
            la = xa @ lin
            lb = xb @ lin
            qa = (xa * xa) @ wsum
            qb = (xb * xb) @ wsum
            cross = (xa * wsum) @ xb.T
            change = (lb[None, :] - la[:, None]) + qa[:, None] + qb[None, :] - 2.0 * cross
            flat = int(np.argmin(change))
            i, j = divmod(flat, change.shape[1])
            if change[i, j] < -1e-12:
                gi, gj = ia[pa[i]], ib[pb[j]]
                delta = u[gj] - u[gi]
                sums[a] += delta
                sums[b] -= delta
                assign[gi], assign[gj] = b, a
                members[a][pa[i]] = gj
                members[b][pb[j]] = gi
                swaps += 1
                stale = 0
            else:
                stale += len(pairs) * self.patience if exhaustive and k == 2 else 1
        final = float((sums * sums * inv).sum())
        return assign, {"loss": final, "initial_loss": initial, "swaps": float(swaps)}
