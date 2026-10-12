"""Markov chains and PageRank, written from scratch with NumPy arrays.

Modules:
    linalg     Gaussian elimination with partial pivoting
    chain      distributions, stationary distribution (3 methods), simulation, period, mixing
    models     chains with known answers: two-state chain, Ehrenfest urn, lazy cycle walk
    pagerank   PageRank (power method on a sparse graph, Monte Carlo estimator)
    airports   the OpenFlights world airline network
"""

from . import airports, chain, linalg, models, pagerank  # noqa: F401

__all__ = ["airports", "chain", "linalg", "models", "pagerank"]
