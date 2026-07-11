import networkx as nx


# ============================================================
# EURISTICHE (tutte ammissibili)
# ============================================================

def make_h_zero():
    """Euristica zero: A* degenera in Dijkstra. Baseline di confronto."""
    def h(node, goal):
        return 0
    return h


def make_h_euclidean(graph):
    """
    Distanza in linea d'aria approssimata su piano cartesiano,
    scalando i gradi in metri (come da bozza di progetto).
    Ammissibile: la linea d'aria non sovrastima mai la distanza stradale.
    """
    import math

    M_PER_DEG_LAT = 111_320  # metri per grado di latitudine (circa costante)

    def h(node, goal):
        y1, x1 = graph.nodes[node]["y"], graph.nodes[node]["x"]
        y2, x2 = graph.nodes[goal]["y"], graph.nodes[goal]["x"]
        # i gradi di longitudine "valgono" meno metri man mano che ci si
        # allontana dall'equatore: si corregge con il coseno della latitudine
        lat_media = math.radians((y1 + y2) / 2)
        dy = (y1 - y2) * M_PER_DEG_LAT
        dx = (x1 - x2) * M_PER_DEG_LAT * math.cos(lat_media)
        return math.hypot(dx, dy)
    return h


def make_h_landmark(graph, landmarks):
    """
    Landmark heuristic (ALT): precalcola con Dijkstra le distanze di ogni
    nodo da alcuni nodi di riferimento. Per la disuguaglianza triangolare:
        h(n) = max_L | d(L, goal) - d(L, n) |
    è ammissibile e in genere più informata dell'euclidea.
    """
    # Precalcolo: un Dijkstra completo per ogni landmark (costoso ma una tantum)
    dist_from = {
        L: nx.single_source_dijkstra_path_length(graph, L, weight="length")
        for L in landmarks
    }

    def h(node, goal):
        best = 0
        for L in landmarks:
            d = dist_from[L]
            if node in d and goal in d:
                est = abs(d[goal] - d[node])
                if est > best:
                    best = est
        return best
    return h
