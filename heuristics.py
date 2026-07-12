import math
import random

import networkx as nx


# 1. EURISTICA NULLA
def make_h_zero():
    """Euristica zero: A* degenera in Dijkstra. Baseline di confronto."""
    def h(node, goal):
        return 0.0
    return h


# 2. DISTANZA IN LINEA D'ARIA
def _metri_per_grado_lat(lat_deg):
    """
    Metri per grado di latitudine ALLA LATITUDINE DATA (WGS84).
    """
    p = math.radians(lat_deg)
    return 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)


def _metri_per_grado_lon(lat_deg):
    """Metri per grado di longitudine alla latitudine data (i meridiani convergono)."""
    p = math.radians(lat_deg)
    return 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)


def make_h_euclidean(graph):
    """
    Distanza in linea d'aria su proiezione equirettangolare locale.
    Ammissibile: la strada tra due punti non e' mai piu' corta della linea d'aria.
    Consistente: la distanza euclidea soddisfa la disuguaglianza triangolare.
    """
    # cache delle coordinate: evita un lookup nel grafo a ogni chiamata di h
    pos = {n: (d["y"], d["x"]) for n, d in graph.nodes(data=True)}

    def h(node, goal):
        y1, x1 = pos[node]
        y2, x2 = pos[goal]
        lat_media = (y1 + y2) / 2
        dy = (y1 - y2) * _metri_per_grado_lat(lat_media)
        dx = (x1 - x2) * _metri_per_grado_lon(lat_media)
        return math.hypot(dx, dy)

    return h


# 3. EURISTICA PESATA  (Weighted A*)
def make_h_weighted(h_base, w):
    """
    f = g + w*h.  Per w > 1 l'euristica NON e' piu' ammissibile: A* perde
    l'ottimalita' ma espande molti meno nodi. Il costo trovato resta comunque
    garantito entro un fattore w dall'ottimo (w-ammissibilita').
    """
    def h(node, goal):
        return w * h_base(node, goal)
    return h


# 4. LANDMARK / ALT
def scegli_landmark_farthest(graph, k, seed=42):
    """
    Selezione 'farthest' (greedy maxmin): i landmark utili stanno ai BORDI
    del grafo.

    Procedura: parto da un nodo casuale, prendo il piu' lontano da lui, poi
    ogni volta il nodo che MASSIMIZZA la distanza minima dai landmark gia' scelti.
    """
    rng = random.Random(seed)
    seme = rng.choice(list(graph.nodes))

    d0 = nx.single_source_dijkstra_path_length(graph, seme, weight="length")
    landmarks = [max(d0, key=d0.get)]  # il nodo piu' lontano dal seme

    dist_min = dict(nx.single_source_dijkstra_path_length(
        graph, landmarks[0], weight="length"))

    while len(landmarks) < k:
        cand = max((n for n in dist_min if n not in landmarks),
                   key=lambda n: dist_min[n], default=None)
        if cand is None:
            break
        landmarks.append(cand)
        d_new = nx.single_source_dijkstra_path_length(graph, cand, weight="length")
        for n in dist_min:
            dist_min[n] = min(dist_min[n], d_new.get(n, float("inf")))

    return landmarks


def make_h_landmark(graph, landmarks=None, k=8, n_attivi=3, seed=42):
    """
    Euristica ALT (Goldberg & Harrelson, 2005).

    Il grafo stradale e' ORIENTATO (sensi unici: 'network_type=drive' crea un
    solo arco per le vie a senso unico), quindi in generale d(L,v) != d(v,L).
    Valgono percio' DUE disuguaglianze triangolari distinte:

        d(n,goal) >= d(L,goal) - d(L,n)      -> serve dist_from[L]
        d(n,goal) >= d(n,L)    - d(goal,L)   -> serve dist_to[L]

    Entrambe sono lower bound validi: h = max su tutti i landmark e su entrambi
    i bound e' ammissibile (il max di euristiche ammissibili lo e' a sua volta).

    n_attivi: per una data coppia (start, goal) solo pochi landmark danno bound
    utili. Selezionandoli una volta sola in preprocess() si riduce il costo per
    nodo senza perdere quasi nulla in informativita'.
    """
    if landmarks is None:
        landmarks = scegli_landmark_farthest(graph, k, seed=seed)

    # d(L, v): Dijkstra sul grafo normale (segue gli archi uscenti)
    dist_from = {L: nx.single_source_dijkstra_path_length(graph, L, weight="length")
                 for L in landmarks}

    # d(v, L): Dijkstra sul grafo TRASPOSTO (segue gli archi entranti).
    # reverse(copy=False) restituisce una vista: non duplica il grafo in memoria.
    G_rev = graph.reverse(copy=False)
    dist_to = {L: nx.single_source_dijkstra_path_length(G_rev, L, weight="length")
               for L in landmarks}

    INF = float("inf")
    stato = {"attivi": landmarks}  # di default: tutti

    def _bound(L, node, goal):
        df, dt = dist_from[L], dist_to[L]
        b = 0.0
        a1 = df.get(goal, INF) - df.get(node, INF)   # d(L,goal) - d(L,n)
        a2 = dt.get(node, INF) - dt.get(goal, INF)   # d(n,L)    - d(goal,L)
        if a1 != INF and a1 > b:
            b = a1
        if a2 != INF and a2 > b:
            b = a2
        return b

    def preprocess(start, goal):
        """
        Da chiamare UNA VOLTA prima di ogni ricerca: tiene solo gli n_attivi
        landmark che danno il bound migliore per QUESTA coppia (start, goal).
        """
        ordinati = sorted(landmarks, key=lambda L: _bound(L, start, goal), reverse=True)
        stato["attivi"] = ordinati[:n_attivi]

    def h(node, goal):
        best = 0.0
        for L in stato["attivi"]:
            b = _bound(L, node, goal)
            if b > best:
                best = b
        return best

    h.landmarks = landmarks
    h.preprocess = preprocess  
    return h