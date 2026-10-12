"""World airline network from OpenFlights (routes as of June 2014, Open Database License).

Nodes are airports, and there is an edge a -> b weighted by the number of airlines flying
the route a -> b. A random passenger who keeps taking a random flight out of the current
airport is a Markov chain on the airports, and PageRank measures where they spend their time.
"""

import csv
import os

import numpy as np

from .pagerank import Graph

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_airports(path=None):
    """IATA code -> dict(name, city, country, lat, lon)."""
    path = path or os.path.join(DATA, "airports.dat")
    info = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            iata = row[4]
            if len(iata) == 3 and iata != "\\N":
                info[iata] = {"name": row[1], "city": row[2], "country": row[3], "lat": float(row[6]), "lon": float(row[7])}
    return info


def load_network(path=None):
    """Return (graph, codes, info): graph nodes are numbered in the order of `codes` (sorted IATA codes)."""
    path = path or os.path.join(DATA, "routes.dat")
    info = load_airports()
    counts = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            a, b = row[2], row[4]
            if len(a) != 3 or len(b) != 3 or a == b:
                continue
            counts[(a, b)] = counts.get((a, b), 0) + 1
    codes = sorted({a for a, _ in counts} | {b for _, b in counts})
    index = {c: i for i, c in enumerate(codes)}
    src = np.array([index[a] for a, _ in counts])
    dst = np.array([index[b] for _, b in counts])
    w = np.array(list(counts.values()), dtype=float)
    return Graph(len(codes), src, dst, w, labels=codes), codes, info
