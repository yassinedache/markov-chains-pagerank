"""Finite Markov chains: distributions, stationary distribution, simulation, convergence.

A chain on the states 0..n-1 is given by its transition matrix P (rows sum to 1):
P[i, j] is the probability of moving from i to j in one step. Distributions are row
vectors, so the distribution after one step is mu @ P.
"""

from math import gcd

import numpy as np

from .linalg import gauss_solve


def check_stochastic(P, tol=1e-12):
    """Return P as a float array after checking that it is a valid transition matrix."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if np.any(P < -tol):
        raise ValueError("P has negative entries")
    if np.max(np.abs(P.sum(axis=1) - 1)) > tol * P.shape[0] + tol:
        raise ValueError("the rows of P must sum to 1")
    return P


def tv_distance(mu, nu):
    """Total variation distance: max over events A of |mu(A) - nu(A)| = (1/2) sum |mu - nu|."""
    return 0.5 * float(np.sum(np.abs(np.asarray(mu) - np.asarray(nu))))


def distributions(mu0, P, n_steps):
    """Distributions mu_0, mu_1, ..., mu_n where mu_{k+1} = mu_k P (array of shape (n+1, states))."""
    P = check_stochastic(P)
    out = np.empty((n_steps + 1, P.shape[0]))
    out[0] = mu0
    for k in range(n_steps):
        out[k + 1] = out[k] @ P
    return out


def stationary_power(P, tol=1e-13, max_iter=1_000_000, mu0=None):
    """Stationary distribution by power iteration mu <- mu P, from the uniform distribution.

    Converges for an irreducible aperiodic chain. Returns (pi, number of iterations).
    Stops when the L1 change between two iterates is below tol.
    """
    P = check_stochastic(P)
    n = P.shape[0]
    mu = np.full(n, 1.0 / n) if mu0 is None else np.asarray(mu0, dtype=float)
    for k in range(1, max_iter + 1):
        new = mu @ P
        if np.sum(np.abs(new - mu)) < tol:
            return new / new.sum(), k
        mu = new
    raise RuntimeError("power iteration did not converge (is the chain periodic?)")


def stationary_linear(P):
    """Stationary distribution by solving pi (P - I) = 0 with sum(pi) = 1.

    The n equations pi (P - I) = 0 are dependent (they sum to 0), so the last one is replaced
    by the normalisation sum(pi) = 1. The system has a unique solution when the chain is
    irreducible. Solved by Gaussian elimination with partial pivoting (markov.linalg).
    """
    P = check_stochastic(P)
    n = P.shape[0]
    A = (P - np.eye(n)).T            # pi (P - I) = 0  <=>  (P - I)^T pi^T = 0
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return gauss_solve(A, b)


def simulate(P, n_steps, start=0, rng=None):
    """Simulate a trajectory X_0 = start, X_1, ..., X_n (array of length n+1)."""
    P = check_stochastic(P)
    rng = np.random.default_rng(rng)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0                 # guard against rounding
    u = rng.random(n_steps)
    path = np.empty(n_steps + 1, dtype=np.int64)
    x = int(start)
    path[0] = x
    rows = [row for row in cum]      # list of rows: faster indexing in the loop
    for k in range(n_steps):
        x = int(np.searchsorted(rows[x], u[k], side="right"))
        path[k + 1] = x
    return path


def occupation_many(P, n_steps, n_chains, checkpoints, start=0, rng=None):
    """Run n_chains independent copies of the chain in parallel (vectorised) from `start`.

    Returns an array of shape (len(checkpoints), n_chains, n_states): row c holds, for each copy,
    the fraction of time spent in each state among X_0, ..., X_{n-1} with n = checkpoints[c].
    """
    P = check_stochastic(P)
    rng = np.random.default_rng(rng)
    n_states = P.shape[0]
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0
    checkpoints = sorted(int(c) for c in checkpoints)
    if checkpoints[-1] > n_steps + 1:
        raise ValueError("a checkpoint is longer than the trajectories")
    rows = np.arange(n_chains)
    x = np.full(n_chains, int(start))
    counts = np.zeros((n_chains, n_states))
    counts[rows, x] += 1
    out, j = [], 0
    while j < len(checkpoints) and checkpoints[j] == 1:
        out.append(counts.copy())
        j += 1
    for k in range(1, checkpoints[-1]):
        u = rng.random(n_chains)
        x = (cum[x] <= u[:, None]).sum(axis=1)     # first state whose cumulative probability exceeds u
        counts[rows, x] += 1
        while j < len(checkpoints) and checkpoints[j] == k + 1:
            out.append(counts / (k + 1))
            j += 1
    return np.array(out)


def occupation_frequencies(path, n_states):
    """Fraction of time spent in each state along a trajectory (ergodic theorem: -> pi)."""
    return np.bincount(path, minlength=n_states) / len(path)


def is_irreducible(P):
    """True if every state can reach every other state (strongly connected transition graph)."""
    P = check_stochastic(P)
    n = P.shape[0]
    adj = P > 0

    def reach(adjacency):
        seen = np.zeros(n, dtype=bool)
        seen[0] = True
        frontier = [0]
        while frontier:
            nxt = np.nonzero(adjacency[frontier].any(axis=0) & ~seen)[0]
            seen[nxt] = True
            frontier = list(nxt)
        return seen.all()

    return reach(adj) and reach(adj.T)


def period(P):
    """Period of an irreducible chain: gcd of the lengths of all cycles through a state.

    Computed with a breadth-first search: if level(x) is the distance from state 0, the period
    is the gcd of level(i) + 1 - level(j) over all edges i -> j.
    """
    P = check_stochastic(P)
    if not is_irreducible(P):
        raise ValueError("the period is defined here for irreducible chains")
    n = P.shape[0]
    level = np.full(n, -1)
    level[0] = 0
    frontier = [0]
    while frontier:
        nxt = []
        for i in frontier:
            for j in np.nonzero(P[i] > 0)[0]:
                if level[j] < 0:
                    level[j] = level[i] + 1
                    nxt.append(j)
        frontier = nxt
    d = 0
    for i in range(n):
        for j in np.nonzero(P[i] > 0)[0]:
            d = gcd(d, int(abs(level[i] + 1 - level[j])))
    return d


def is_reversible(P, pi, tol=1e-12):
    """Detailed balance: pi_i P_ij = pi_j P_ji for all i, j."""
    F = np.asarray(pi)[:, None] * np.asarray(P)
    return bool(np.max(np.abs(F - F.T)) < tol)


def mixing_time(P, eps=0.25, start=0, pi=None, max_steps=10_000_000):
    """Smallest n with ||mu_n - pi||_TV <= eps, starting from the state `start`."""
    P = check_stochastic(P)
    pi = stationary_linear(P) if pi is None else pi
    mu = np.zeros(P.shape[0])
    mu[start] = 1.0
    for n in range(max_steps + 1):
        if tv_distance(mu, pi) <= eps:
            return n
        mu = mu @ P
    raise RuntimeError("did not mix within max_steps")


def tv_bound(pi_start, lam_star, n):
    """Spectral bound for a reversible chain started at x:
    ||P^n(x, .) - pi||_TV <= (1/2) sqrt((1 - pi(x)) / pi(x)) * lam_star^n."""
    return 0.5 * np.sqrt((1 - pi_start) / pi_start) * np.asarray(lam_star, dtype=float) ** np.asarray(n)


def observed_rate(values, start=None, stop=None):
    """Geometric rate r of a sequence decaying like C r^n (least-squares slope of log values)."""
    v = np.asarray(values, dtype=float)
    k = np.arange(len(v))
    sel = slice(start, stop)
    kk, vv = k[sel], v[sel]
    ok = vv > 0
    return float(np.exp(np.polyfit(kk[ok], np.log(vv[ok]), 1)[0]))
