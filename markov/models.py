"""Classic chains with known answers, used to check the methods."""

from math import comb, cos, pi as PI

import numpy as np


def two_state(p, q):
    """Two states: 0 -> 1 with probability p, 1 -> 0 with probability q.
    Stationary distribution (q, p) / (p + q); second eigenvalue 1 - p - q."""
    return np.array([[1 - p, p], [q, 1 - q]])


def lazy(P):
    """Lazy version (I + P) / 2: stay put with probability 1/2. Removes periodicity,
    keeps the stationary distribution, maps the eigenvalues lambda to (1 + lambda) / 2."""
    P = np.asarray(P, dtype=float)
    return 0.5 * (np.eye(P.shape[0]) + P)


def ehrenfest(N, lazy_chain=False):
    """Ehrenfest urn with N balls. State k = number of balls in urn A. At each step a ball chosen
    uniformly at random changes urn: k -> k - 1 with probability k/N, k -> k + 1 otherwise.
    Stationary distribution: Binomial(N, 1/2). Eigenvalues 1 - 2j/N, j = 0..N (period 2)."""
    P = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        if k > 0:
            P[k, k - 1] = k / N
        if k < N:
            P[k, k + 1] = (N - k) / N
    return lazy(P) if lazy_chain else P


def ehrenfest_stationary(N):
    return np.array([comb(N, k) for k in range(N + 1)], dtype=float) / 2.0 ** N


def ehrenfest_eigenvalues(N, lazy_chain=False):
    lam = np.array([1 - 2 * j / N for j in range(N + 1)])
    return (1 + lam) / 2 if lazy_chain else lam


def cycle_walk(m):
    """Lazy random walk on the cycle Z/mZ: stay with probability 1/2, move to each neighbour
    with probability 1/4. Uniform stationary distribution."""
    P = np.zeros((m, m))
    for x in range(m):
        P[x, x] += 0.5
        P[x, (x + 1) % m] += 0.25
        P[x, (x - 1) % m] += 0.25
    return P


def cycle_eigenvalues(m):
    """(1 + cos(2 pi k / m)) / 2 for k = 0..m-1 (P is circulant, its eigenvectors are Fourier modes)."""
    return np.array([(1 + cos(2 * PI * k / m)) / 2 for k in range(m)])


def cycle_lambda2(m):
    """Second largest eigenvalue of the lazy cycle walk: cos^2(pi / m) ~ 1 - pi^2 / m^2."""
    return cos(PI / m) ** 2
