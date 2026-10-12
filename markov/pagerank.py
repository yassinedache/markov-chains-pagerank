"""PageRank, written from scratch with sparse (edge list) operations.

Random surfer model on a directed weighted graph with n nodes: from node i the surfer
  - with probability alpha follows an out-link i -> j chosen with probability w_ij / sum_k w_ik,
  - with probability 1 - alpha jumps to a node drawn from the teleportation vector v (uniform by default).
A node without out-links ("dangling") always jumps according to v.
PageRank is the stationary distribution pi of this chain, whose transition matrix is the Google matrix
    G = alpha S' + (1 - alpha) 1 v^T,
where S' is the link matrix S with the rows of dangling nodes replaced by v.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Graph:
    """Directed weighted graph stored as an edge list (src[k] -> dst[k] with weight w[k])."""

    n: int
    src: np.ndarray
    dst: np.ndarray
    w: np.ndarray
    labels: list = field(default=None)

    def __post_init__(self):
        self.src = np.asarray(self.src, dtype=np.int64)
        self.dst = np.asarray(self.dst, dtype=np.int64)
        self.w = np.ones(len(self.src)) if self.w is None else np.asarray(self.w, dtype=float)
        if not (len(self.src) == len(self.dst) == len(self.w)):
            raise ValueError("src, dst and w must have the same length")
        if len(self.src) and (self.src.min() < 0 or self.dst.min() < 0 or max(self.src.max(), self.dst.max()) >= self.n):
            raise ValueError("node index out of range")
        if np.any(self.w <= 0):
            raise ValueError("edge weights must be positive")
        self.out_weight = np.bincount(self.src, weights=self.w, minlength=self.n)
        self.dangling = self.out_weight == 0

    @classmethod
    def from_edges(cls, edges, n=None, labels=None):
        edges = list(edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        w = [e[2] if len(e) > 2 else 1.0 for e in edges]
        n = n if n is not None else (max(src + dst) + 1 if edges else 0)
        return cls(n, np.array(src), np.array(dst), np.array(w, dtype=float), labels)

    def in_weight(self):
        return np.bincount(self.dst, weights=self.w, minlength=self.n)

    def transition_probs(self):
        """Probability of each edge for the surfer who follows a link: w_ij / sum_k w_ik."""
        return self.w / self.out_weight[self.src]

    def link_step(self, x):
        """Row vector x times the link matrix S (rows of dangling nodes are zero here; the
        dangling mass is redistributed in google_step).

        Computed in O(number of edges) with np.bincount: (x S)_j = sum over edges i -> j of x_i p_ij.
        """
        return np.bincount(self.dst, weights=x[self.src] * self.transition_probs(), minlength=self.n)

    def dense_google_matrix(self, alpha=0.85, v=None):
        """Dense Google matrix (only for small graphs, used in the tests)."""
        v = np.full(self.n, 1.0 / self.n) if v is None else np.asarray(v, dtype=float)
        S = np.zeros((self.n, self.n))
        np.add.at(S, (self.src, self.dst), self.transition_probs())
        S[self.dangling] = v
        return alpha * S + (1 - alpha) * np.outer(np.ones(self.n), v)


@dataclass
class PageRankResult:
    scores: np.ndarray
    iterations: int
    residuals: list          # ||x_{k+1} - x_k||_1 at each iteration
    error_bound: float       # guaranteed bound on ||x - pi||_1 at the end: residual / (1 - alpha)


def google_step(graph, x, alpha, v):
    """One step x -> x G of the power method, without ever forming G (O(edges))."""
    dangling_mass = x[graph.dangling].sum()
    return alpha * graph.link_step(x) + (alpha * dangling_mass + (1 - alpha) * x.sum()) * v


def pagerank(graph, alpha=0.85, tol=1e-12, max_iter=10_000, v=None, x0=None):
    """PageRank by the power method x_{k+1} = x_k G.

    Convergence is geometric: ||x_k - pi||_1 <= alpha^k ||x_0 - pi||_1 (see the report).
    The loop stops when ||x_{k+1} - x_k||_1 <= tol (1 - alpha), which guarantees ||x_{k+1} - pi||_1 <= tol.
    """
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    n = graph.n
    v = np.full(n, 1.0 / n) if v is None else np.asarray(v, dtype=float) / np.sum(v)
    x = v.copy() if x0 is None else np.asarray(x0, dtype=float) / np.sum(x0)
    residuals = []
    for k in range(1, max_iter + 1):
        new = google_step(graph, x, alpha, v)
        r = float(np.sum(np.abs(new - x)))
        residuals.append(r)
        x = new
        if r <= tol * (1 - alpha):
            break
    x = x / x.sum()
    return PageRankResult(x, k, residuals, residuals[-1] / (1 - alpha) if alpha < 1 else float("inf"))


def pagerank_monte_carlo(graph, alpha=0.85, n_walks=100_000, rng=None, v=None):
    """Monte Carlo estimate of PageRank by the end points of random walks.

    Each walk starts at a node drawn from v; at each step it stops with probability 1 - alpha,
    otherwise it moves along S' (a random out-link, or a jump drawn from v at a dangling node).
    Since pi = (1 - alpha) v (I - alpha S')^{-1} = sum_k (1 - alpha) alpha^k v S'^k, the end point of a
    walk is distributed exactly according to pi: the estimate is unbiased, with error ~ 1/sqrt(n_walks).
    All walks are simulated at once (vectorised).
    """
    rng = np.random.default_rng(rng)
    n = graph.n
    v = np.full(n, 1.0 / n) if v is None else np.asarray(v, dtype=float) / np.sum(v)
    cv = np.cumsum(v)
    cv[-1] = 1.0
    # edges sorted by source node, with cumulative probabilities inside each row
    order = np.argsort(graph.src, kind="stable")
    dst_sorted = graph.dst[order]
    prob_sorted = graph.transition_probs()[order]
    cum = np.cumsum(prob_sorted)
    start = np.concatenate([[0], np.cumsum(np.bincount(graph.src, minlength=n))])  # row pointers
    base = np.concatenate([[0.0], cum])[start[:-1]]                                # cum mass before each row

    pos = np.searchsorted(cv, rng.random(n_walks), side="right")
    ends = np.empty(n_walks, dtype=np.int64)
    alive = np.arange(n_walks)
    while alive.size:
        stop = rng.random(alive.size) >= alpha
        ends[alive[stop]] = pos[alive[stop]]
        alive = alive[~stop]
        cur = pos[alive]
        dang = graph.dangling[cur]
        nxt = np.empty(alive.size, dtype=np.int64)
        nxt[dang] = np.searchsorted(cv, rng.random(int(dang.sum())), side="right")
        live = ~dang
        u = rng.random(int(live.sum()))
        c = cur[live]
        idx = np.searchsorted(cum, base[c] + u * (cum[start[c + 1] - 1] - base[c]), side="right")
        idx = np.minimum(np.maximum(idx, start[c]), start[c + 1] - 1)   # stay inside the row (rounding)
        nxt[live] = dst_sorted[idx]
        pos[alive] = nxt
    return np.bincount(ends, minlength=n) / n_walks


def strongly_connected_components(graph):
    """Label of the strongly connected component of each node (Kosaraju's algorithm, iterative DFS)."""
    n = graph.n

    def csr(src, dst):
        order = np.argsort(src, kind="stable")
        ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
        return ptr, dst[order]

    ptr, nbr = csr(graph.src, graph.dst)
    rptr, rnbr = csr(graph.dst, graph.src)
    # pass 1: nodes in increasing order of DFS finishing time
    seen = np.zeros(n, dtype=bool)
    finish = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, ptr[s])]
        while stack:
            u, i = stack[-1]
            if i < ptr[u + 1]:
                stack[-1] = (u, i + 1)
                w = nbr[i]
                if not seen[w]:
                    seen[w] = True
                    stack.append((w, ptr[w]))
            else:
                stack.pop()
                finish.append(u)
    # pass 2: DFS on the reversed graph, by decreasing finishing time
    comp = np.full(n, -1, dtype=np.int64)
    c = 0
    for s in reversed(finish):
        if comp[s] >= 0:
            continue
        comp[s] = c
        stack = [s]
        while stack:
            u = stack.pop()
            for w in rnbr[rptr[u]:rptr[u + 1]]:
                if comp[w] < 0:
                    comp[w] = c
                    stack.append(w)
        c += 1
    return comp


def closed_classes(graph, comp=None):
    """Components that no link leaves (closed communicating classes of the link matrix S').

    A dangling node is not closed: in S' it jumps according to v. If there are at least two closed
    classes, the second eigenvalue of the Google matrix has modulus exactly alpha (Haveliwala & Kamvar).
    """
    comp = strongly_connected_components(graph) if comp is None else comp
    leaving = np.zeros(comp.max() + 1, dtype=bool)
    leaving[comp[graph.src[comp[graph.src] != comp[graph.dst]]]] = True
    has_links = np.zeros(comp.max() + 1, dtype=bool)
    has_links[comp[graph.src]] = True
    return [int(c) for c in range(comp.max() + 1) if has_links[c] and not leaving[c]]


def ranking(scores):
    """Rank of each node (1 = highest score)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def average_ranks(values):
    """Ranks 1..n in increasing order, tied values sharing the average of their ranks."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation: Pearson correlation of the (average) ranks."""
    ra, rb = average_ranks(a), average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))
