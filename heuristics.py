import math
import random

import networkx as nx


# ============================================================
# 1. EURISTICA NULLA
# ============================================================
def make_h_zero():
    """Restituisce l'euristica nulla: A* con questa euristica degenera in Dijkstra."""
    def h(node, goal):
        return 0.0
    return h


# ============================================================
# 2. DISTANZA IN LINEA D'ARIA
# ============================================================
def _metri_per_grado_lat(lat_deg):
    """Metri per grado di latitudine alla latitudine data (WGS84)."""
    p = math.radians(lat_deg)
    return 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)


def _metri_per_grado_lon(lat_deg):
    """Metri per grado di longitudine alla latitudine data."""
    p = math.radians(lat_deg)
    return 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)


def make_h_euclidean(graph):
    """Costruisce l'euristica basata sulla distanza in linea d'aria (proiezione locale)."""
    # Cache delle coordinate di tutti i nodi, calcolata una sola volta.
    pos = {n: (d["y"], d["x"]) for n, d in graph.nodes(data=True)}

    def h(node, goal):
        y1, x1 = pos[node]
        y2, x2 = pos[goal]

        # Proiezione equirettangolare locale: converte gradi in metri usando la
        # latitudine media dei due punti.
        lat_media = (y1 + y2) / 2
        dy = (y1 - y2) * _metri_per_grado_lat(lat_media)
        dx = (x1 - x2) * _metri_per_grado_lon(lat_media)

        return math.hypot(dx, dy)

    return h


# ============================================================
# 3. EURISTICA PESATA (Weighted A*)
# ============================================================
def make_h_weighted(h_base, w):
    """Restituisce h_base moltiplicata per il peso w (f = g + w*h)."""
    def h(node, goal):
        return w * h_base(node, goal)
    return h


# ============================================================
# 4. LANDMARK / ALT
# ============================================================
def scegli_landmark_farthest(graph, k, seed=42):
    """
    Seleziona k landmark con la strategia 'farthest' (greedy maxmin): parte da
    un nodo casuale, prende il piu' lontano da lui, poi ad ogni passo aggiunge
    il nodo che massimizza la distanza minima dai landmark gia' scelti.
    """
    rng = random.Random(seed)
    seme = rng.choice(list(graph.nodes))

    d0 = nx.single_source_dijkstra_path_length(graph, seme, weight="length")
    landmarks = [max(d0, key=d0.get)]

    # dist_min[n] = distanza di n dal landmark piu' vicino tra quelli scelti finora.
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
    Costruisce l'euristica ALT (Goldberg & Harrelson, 2005): usa distanze
    esatte precalcolate verso/da un insieme di landmark come lower bound sul
    costo residuo, tramite la disuguaglianza triangolare:

        d(n,goal) >= d(L,goal) - d(L,n)
        d(n,goal) >= d(n,L)    - d(goal,L)

    La funzione restituita ha due attributi: `preprocess(start, goal)`, da
    chiamare prima di ogni ricerca per selezionare i landmark piu' utili alla
    coppia, e `landmarks`, la lista completa dei landmark.
    """
    if landmarks is None:
        landmarks = scegli_landmark_farthest(graph, k, seed=seed)

    # Preprocessing: distanza esatta da e verso ogni landmark, su tutto il grafo.
    dist_from = {L: nx.single_source_dijkstra_path_length(graph, L, weight="length")
                 for L in landmarks}

    G_rev = graph.reverse(copy=False)
    dist_to = {L: nx.single_source_dijkstra_path_length(G_rev, L, weight="length")
               for L in landmarks}

    INF = float("inf")

    # Landmark correntemente attivi (aggiornati da preprocess()).
    stato = {"attivi": landmarks}

    def _bound(L, node, goal):
        """Miglior lower bound su d(node, goal) ricavabile dal landmark L."""
        df, dt = dist_from[L], dist_to[L]
        b = 0.0

        a1 = df.get(goal, INF) - df.get(node, INF)
        a2 = dt.get(node, INF) - dt.get(goal, INF)

        if a1 != INF and a1 > b:
            b = a1
        if a2 != INF and a2 > b:
            b = a2
        return b

    def preprocess(start, goal):
        """Seleziona gli n_attivi landmark con il bound migliore per questa coppia."""
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
