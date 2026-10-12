import numpy as np
import pytest

from markov import chain
from markov.pagerank import (Graph, average_ranks, closed_classes, pagerank, pagerank_monte_carlo, ranking, spearman,
                             strongly_connected_components)


def random_graph(n=40, m=150, seed=0, with_dangling=True):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    if with_dangling:
        keep = src >= 3                      # nodes 0, 1, 2 have no out-links
        src, dst = src[keep], dst[keep]
    w = rng.integers(1, 4, len(src)).astype(float)
    return Graph(n, src, dst, w)


def test_pagerank_is_a_probability_distribution():
    r = pagerank(random_graph())
    assert np.all(r.scores > 0)
    assert np.isclose(r.scores.sum(), 1.0)


def test_pagerank_is_stationary_for_dense_google_matrix():
    g = random_graph()
    G = g.dense_google_matrix(0.85)
    chain.check_stochastic(G)
    r = pagerank(g, 0.85, tol=1e-14)
    assert np.allclose(r.scores @ G, r.scores, atol=1e-14)
    # same answer as solving the linear system with Gaussian elimination
    assert np.allclose(chain.stationary_linear(G), r.scores, atol=1e-12)


def test_matches_networkx():
    nx = pytest.importorskip("networkx")
    g = random_graph(n=60, m=300, seed=3)
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for s, d, w in zip(g.src, g.dst, g.w):
        if G.has_edge(int(s), int(d)):
            G[int(s)][int(d)]["weight"] += w
        else:
            G.add_edge(int(s), int(d), weight=w)
    ref = nx.pagerank(G, alpha=0.85, weight="weight", tol=1e-15, max_iter=1000)
    r = pagerank(g, 0.85, tol=1e-14)
    assert np.max(np.abs(np.array([ref[i] for i in range(g.n)]) - r.scores)) < 1e-12


def test_geometric_convergence_bound():
    g = random_graph(seed=5)
    alpha = 0.9
    exact = pagerank(g, alpha, tol=1e-15).scores
    x = np.full(g.n, 1.0 / g.n)
    e0 = np.abs(x - exact).sum()
    for k in range(1, 60):
        x = pagerank(g, alpha, tol=0, max_iter=k).scores
        assert np.abs(x - exact).sum() <= alpha ** k * e0 + 1e-14


def test_second_eigenvalue_bounded_by_alpha():
    g = random_graph(seed=7)
    for alpha in (0.5, 0.85, 0.95):
        ev = np.sort(np.abs(np.linalg.eigvals(g.dense_google_matrix(alpha))))[::-1]
        assert np.isclose(ev[0], 1.0)
        assert ev[1] <= alpha + 1e-10


def test_stopping_rule_guarantees_the_error():
    g = random_graph(seed=11)
    exact = pagerank(g, 0.85, tol=1e-15).scores
    r = pagerank(g, 0.85, tol=1e-6)
    assert np.abs(r.scores - exact).sum() <= 1e-6
    assert r.error_bound <= 1e-6


def test_special_cases():
    # alpha = 0: pure teleportation
    g = random_graph()
    assert np.allclose(pagerank(g, 0.0).scores, 1.0 / g.n)
    # symmetric regular graph (directed cycle both ways): uniform PageRank
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)] + [((i + 1) % n, i) for i in range(n)]
    assert np.allclose(pagerank(Graph.from_edges(edges)).scores, 1.0 / n)
    # a node everyone links to gets the highest score
    star = Graph.from_edges([(i, 0) for i in range(1, 10)] + [(0, 1)])
    assert ranking(pagerank(star).scores)[0] == 1
    with pytest.raises(ValueError):
        pagerank(g, 1.0)


def test_personalization_vector():
    g = random_graph(seed=2)
    v = np.zeros(g.n)
    v[10] = 1.0
    r = pagerank(g, 0.85, v=v, tol=1e-14)
    G = g.dense_google_matrix(0.85, v=v)
    assert np.allclose(r.scores @ G, r.scores, atol=1e-13)
    assert r.scores[10] >= 0.15 - 1e-12          # at least the teleportation mass 1 - alpha


def test_relabelling_nodes_permutes_scores():
    g = random_graph(seed=4)
    perm = np.random.default_rng(0).permutation(g.n)
    h = Graph(g.n, perm[g.src], perm[g.dst], g.w)
    assert np.allclose(pagerank(h, tol=1e-14).scores[perm], pagerank(g, tol=1e-14).scores)


def test_monte_carlo_estimator_is_close():
    g = random_graph(seed=8)
    exact = pagerank(g, 0.85, tol=1e-14).scores
    est = pagerank_monte_carlo(g, 0.85, n_walks=400_000, rng=1)
    assert np.abs(est - exact).sum() < 0.03


def test_ranks_and_spearman():
    assert list(ranking([0.1, 0.5, 0.3])) == [3, 1, 2]
    assert list(average_ranks([10, 20, 20, 30])) == [1, 2.5, 2.5, 4]
    assert np.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert np.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


def test_strongly_connected_components_match_networkx():
    nx = pytest.importorskip("networkx")
    g = random_graph(n=80, m=120, seed=9)
    comp = strongly_connected_components(g)
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from(zip(g.src.tolist(), g.dst.tolist()))
    ours = sorted(sorted(np.nonzero(comp == c)[0].tolist()) for c in np.unique(comp))
    ref = sorted(sorted(s) for s in nx.strongly_connected_components(G))
    assert ours == ref


def test_two_closed_classes_give_second_eigenvalue_alpha():
    # two separate cycles {0,1,2} and {3,4} plus a node 5 pointing to both: two closed classes
    g = Graph.from_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (5, 0), (5, 3)])
    assert len(closed_classes(g)) == 2
    ev = np.sort(np.abs(np.linalg.eigvals(g.dense_google_matrix(0.8))))[::-1]
    assert np.isclose(ev[1], 0.8)
