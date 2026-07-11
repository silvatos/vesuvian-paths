import heapq
import time


# ============================================================
# ALGORITMO A*
# ============================================================
def a_star(graph, heuristic, start, goal, on_expand=None):
    """
    graph:     grafo osmnx/networkx (MultiDiGraph) con archi pesati su 'length' (metri)
    heuristic: FUNZIONE h(nodo, goal) -> stima del costo residuo (non un dizionario!)
    start:     id del nodo di partenza
    goal:      id del nodo di arrivo
    on_expand: callback opzionale chiamata con ogni nodo nell'ordine di espansione
               (utile per animare/tracciare l'esplorazione); non altera l'algoritmo

    Ritorna (path, costo_totale, metriche)
    """
    # --- Frontiera: coda a priorità (heapq) invece della lista del prof ---
    # ogni elemento è una tupla (f_score, nodo); heapq estrae sempre il minimo
    open_set = [(heuristic(start, goal), start)]

    came_from = {}            # traccia il nodo precedente per ogni nodo visitato
    g_score = {start: 0}      # costo migliore noto per raggiungere ciascun nodo
    closed_set = set()        # --- insieme dei nodi già espansi (assente nel codice del prof) ---

    # --- Metriche di confronto ---
    expanded_nodes = 0        # quanti nodi vengono effettivamente espansi
    peak_frontier = 1         # memoria di picco: max nodi contemporaneamente in coda
    t0 = time.perf_counter()

    while open_set:
        peak_frontier = max(peak_frontier, len(open_set))
        _, current = heapq.heappop(open_set)  # estrae il nodo con f_score minimo in O(log n)

        # Lo stesso nodo può finire più volte in coda (con f diversi):
        # se è già stato espanso, questa è una copia "vecchia" e la saltiamo.
        if current in closed_set:
            continue
        closed_set.add(current)
        expanded_nodes += 1
        if on_expand is not None:
            on_expand(current)  # registra l'ordine di espansione (per l'animazione)

        if current == goal:  # ricostruzione del percorso, come nel codice del prof
            path = []
            total_cost = g_score[goal]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            metrics = {
                "expanded_nodes": expanded_nodes,
                "time_s": time.perf_counter() - t0,
                "cost_m": total_cost,
                "path_len": len(path),
                "peak_frontier": peak_frontier,
            }
            return path[::-1], total_cost, metrics

        # Espansione dei vicini (sintassi osmnx: gli archi paralleli si gestiscono col min)
        for neighbor in graph.neighbors(current):
            if neighbor in closed_set:
                continue
            # costo dell'arco = lunghezza in metri; se ci sono archi paralleli prendo il più corto
            cost = min(d.get("length", 1) for d in graph.get_edge_data(current, neighbor).values())
            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)  # euristica calcolata al volo
                heapq.heappush(open_set, (f, neighbor))

    return None, None, None  # nessun percorso trovato


# ============================================================
# FATTORE DI RAMIFICAZIONE EFFETTIVO b*
# ============================================================
def effective_branching_factor(n_expanded, depth, tol=1e-6):
    """
    Risolve numericamente (bisezione): 1 + b* + b*^2 + ... + b*^d = N + 1
    dove N = nodi espansi e d = profondità della soluzione (len(path) - 1).
    """
    if depth == 0:
        return 0.0
    lo, hi = 1.0, float(n_expanded)
    target = n_expanded + 1
    while hi - lo > tol:
        mid = (lo + hi) / 2
        try:
            # con grafi molto grandi (depth e n_expanded elevati) mid**i può
            # eccedere il massimo rappresentabile: trattarlo come "somma troppo grande"
            total = sum(mid ** i for i in range(depth + 1))
        except OverflowError:
            total = float("inf")
        if total < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
