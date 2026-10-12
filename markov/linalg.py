"""Linear algebra written by hand (NumPy is only used for array storage and arithmetic)."""

import numpy as np


def gauss_solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting.

    A is a square (n, n) matrix, b a vector of length n. Raises ValueError if A is singular
    (a pivot is zero up to rounding).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("A must be square and b must match its size")
    M = np.hstack([A, b[:, None]])
    scale = np.max(np.abs(A)) if n else 1.0
    for k in range(n):
        # partial pivoting: bring the largest remaining entry of column k to row k
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) <= 1e-14 * max(scale, 1e-300):
            raise ValueError("matrix is singular")
        if p != k:
            M[[k, p]] = M[[p, k]]
        # eliminate below the pivot (one row operation per row, vectorised over columns)
        factors = M[k + 1:, k] / M[k, k]
        M[k + 1:, k:] -= factors[:, None] * M[k, k:]
    # back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x
