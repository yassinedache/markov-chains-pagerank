import numpy as np
import pytest

from markov import chain, linalg, models


def test_gauss_solve_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 30))
    b = rng.normal(size=30)
    assert np.allclose(linalg.gauss_solve(A, b), np.linalg.solve(A, b), atol=1e-10)


def test_gauss_solve_needs_pivoting_and_detects_singular():
    # a zero in the first pivot position: fails without row exchanges
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    assert np.allclose(linalg.gauss_solve(A, [1.0, 2.0]), [1.0, 1.0])
    with pytest.raises(ValueError):
        linalg.gauss_solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_check_stochastic_rejects_invalid_matrices():
    with pytest.raises(ValueError):
        chain.check_stochastic([[0.5, 0.6], [0.5, 0.5]])
    with pytest.raises(ValueError):
        chain.check_stochastic([[1.2, -0.2], [0.5, 0.5]])


def test_two_state_chain_closed_form():
    p, q = 0.3, 0.1
    P = models.two_state(p, q)
    expected = np.array([q, p]) / (p + q)
    assert np.allclose(chain.stationary_linear(P), expected)
    assert np.allclose(chain.stationary_power(P)[0], expected)
    # the distance to equilibrium is multiplied by exactly |1 - p - q| at each step
    mus = chain.distributions(np.array([1.0, 0.0]), P, 30)
    d = [chain.tv_distance(m, expected) for m in mus]
    assert np.allclose(np.array(d[1:]) / np.array(d[:-1]), abs(1 - p - q))


def test_ehrenfest_stationary_is_binomial_and_reversible():
    N = 20
    pi = models.ehrenfest_stationary(N)
    for P in (models.ehrenfest(N), models.ehrenfest(N, lazy_chain=True)):
        assert np.allclose(chain.stationary_linear(P), pi, atol=1e-14)
        assert chain.is_reversible(P, pi)
    assert np.allclose(chain.stationary_power(models.ehrenfest(N, lazy_chain=True))[0], pi, atol=1e-12)


def test_known_eigenvalues():
    # NumPy is only used here, to check the closed-form spectra used in the experiments
    assert np.allclose(np.sort(np.linalg.eigvals(models.ehrenfest(12)).real), np.sort(models.ehrenfest_eigenvalues(12)))
    assert np.allclose(np.sort(np.linalg.eigvals(models.ehrenfest(12, True)).real), np.sort(models.ehrenfest_eigenvalues(12, True)))
    assert np.allclose(np.sort(np.linalg.eigvals(models.cycle_walk(15)).real), np.sort(models.cycle_eigenvalues(15)))
    assert np.isclose(sorted(models.cycle_eigenvalues(15))[-2], models.cycle_lambda2(15))


def test_period():
    assert chain.period(models.ehrenfest(10)) == 2
    assert chain.period(models.ehrenfest(10, lazy_chain=True)) == 1
    assert chain.period(np.roll(np.eye(5), 1, axis=1)) == 5          # deterministic 5-cycle
    assert chain.is_irreducible(models.cycle_walk(7))
    assert not chain.is_irreducible(np.array([[1.0, 0.0], [0.5, 0.5]]))


def test_periodic_chain_does_not_converge_but_lazy_one_does():
    N = 10
    pi = models.ehrenfest_stationary(N)
    start = np.zeros(N + 1)
    start[0] = 1.0
    plain = chain.distributions(start, models.ehrenfest(N), 2000)
    lazy = chain.distributions(start, models.ehrenfest(N, True), 400)
    assert chain.tv_distance(plain[-1], pi) > 0.49          # stuck on even states
    assert chain.tv_distance(lazy[-1], pi) < 1e-12
    # Cesaro averages of the periodic chain do converge (like 1/n)
    assert chain.tv_distance(plain.mean(axis=0), pi) < 0.005


def test_convergence_rate_is_second_eigenvalue():
    m = 20
    P = models.cycle_walk(m)
    pi = np.full(m, 1.0 / m)
    start = np.zeros(m)
    start[0] = 1.0
    d = [chain.tv_distance(mu, pi) for mu in chain.distributions(start, P, 600)]
    assert abs(chain.observed_rate(d, start=200, stop=600) - models.cycle_lambda2(m)) < 1e-4


def test_spectral_bound_holds():
    N = 15
    P = models.ehrenfest(N, True)
    pi = models.ehrenfest_stationary(N)
    lam_star = max(abs(x) for x in models.ehrenfest_eigenvalues(N, True) if not np.isclose(x, 1))
    start = np.zeros(N + 1)
    start[0] = 1.0
    mus = chain.distributions(start, P, 200)
    for n, mu in enumerate(mus):
        assert chain.tv_distance(mu, pi) <= chain.tv_bound(pi[0], lam_star, n) + 1e-15


def test_mixing_time_of_cycle_grows_like_m_squared():
    t = [chain.mixing_time(models.cycle_walk(m), pi=np.full(m, 1.0 / m)) for m in (16, 32, 64)]
    slope = np.polyfit(np.log([16, 32, 64]), np.log(t), 1)[0]
    assert 1.9 < slope < 2.1


def test_ergodic_theorem_simulation():
    N = 10
    P = models.ehrenfest(N)                       # periodic: frequencies still converge
    path = chain.simulate(P, 200_000, start=0, rng=1)
    freq = chain.occupation_frequencies(path, N + 1)
    assert chain.tv_distance(freq, models.ehrenfest_stationary(N)) < 0.01
    assert np.all(np.abs(np.diff(path)) == 1)    # only moves of +-1


def test_parallel_chains_match_the_ergodic_theorem():
    P = models.two_state(0.3, 0.1)
    F = chain.occupation_many(P, 20_000, 64, [1, 100, 20_000], rng=3)
    assert F.shape == (3, 64, 2)
    assert np.allclose(F[0][:, 0], 1.0)                      # X_0 = 0 for every copy
    assert np.allclose(F.sum(axis=2), 1.0)
    assert np.abs(F[-1].mean(axis=0) - [0.25, 0.75]).max() < 0.01
