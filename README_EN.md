# Markov chains and PageRank

![tests](https://github.com/yassinedache/markov-chains-pagerank/actions/workflows/tests.yml/badge.svg)

*Version française et détails : [README.md](README.md) · Full report (in French): [report/rapport.pdf](report/rapport.pdf)*

Personal project in Python (NumPy, Matplotlib). I implemented the basic tools for finite Markov chains and the **PageRank** algorithm, then **checked the theory experimentally**: convergence speed set by the second eigenvalue, mixing times, the $1/\sqrt n$ rate of Monte Carlo methods, and the $\alpha^k$ contraction of the power method. PageRank is applied to the **world airline network** (3,425 airports, 37,594 routes) and validated against NetworkX (used only for that comparison).

## Key results

- **Stationary distribution, three ways** (Ehrenfest urn, exact answer Binomial(20, 1/2)): linear system solved by hand-written Gaussian elimination (error $1.9 \times 10^{-16}$), power iteration on the lazy chain ($1.5 \times 10^{-14}$, 297 iterations), simulation via the ergodic theorem ($7.2 \times 10^{-4}$ after $10^6$ steps). The plain Ehrenfest chain has period 2 and does not converge.
- **Spectral gap**: for a reversible chain, $\lVert P^t(x,\cdot) - \pi\rVert_{TV} \le \frac12\sqrt{(1-\pi(x))/\pi(x)}\,\lambda_\star^t$ (proved in the report). Measured rates match $\lambda_\star$ to $10^{-6}$ (e.g. 0.993844 for the lazy walk on a 40-cycle, where $\lambda_\star = \cos^2(\pi/m)$), and the mixing time of the cycle walk grows like $m^{2.00}$.
- **PageRank on the airline network** (OpenFlights, routes as of June 2014): sparse power method, never forming the Google matrix, converges in 169 iterations, about three times faster than NetworkX, with a maximum difference of $6.6 \times 10^{-13}$. I prove $\lVert x_k - \pi\rVert_1 \le \alpha^k \lVert x_0 - \pi\rVert_1$, which gives a guaranteed stopping rule; since the graph has 7 closed classes, $|\lambda_2(G)| = \alpha$ exactly (Haveliwala & Kamvar) and the measured rate reaches $\alpha$.
- **Top airports**: Atlanta, Chicago O'Hare, Los Angeles, Dallas-Fort Worth, Paris CDG, London Heathrow, Singapore, Beijing, Denver, Frankfurt. PageRank is not just the number of incoming routes (Spearman 0.807): Alaskan airports climb hundreds of places because a random passenger who enters that dense regional network stays there a long time. Algiers (ALG) ranks 139th of 3,425.
- **Monte Carlo**: the ergodic theorem (400 parallel chains) and a random-walk estimator of PageRank (the end point of a walk that stops with probability $1-\alpha$ is exactly $\pi$-distributed) both converge like $1/\sqrt n$ (measured slopes $-0.51$ and $-0.48$); one million walks recover the exact top 10.

![World map](figures/5_world_map.png)

![Convergence](figures/4_convergence.png)

## Usage

```bash
pip install -r requirements.txt
python experiments/run_all.py   # figures, results.md and the report's tables (about a minute)
pytest                          # 26 tests
```

Data: OpenFlights airport and route databases (Open Database License, see [data/README.md](data/README.md)).
Author: **Yassine Abderrezak Dache**, BSc in Computer Science (USTHB). Code under the MIT licence.
