import heapq
import time
from collections import deque


# ============================================================
# UTILITY CONDIVISE
# ============================================================
def _costo_arco(graph, u, v):
    """Costo dell'arco u->v in metri (il minimo tra archi paralleli)."""
    # Il grafo e' un MultiDiGraph: get_edge_data(u, v) restituisce un
    # dizionario {chiave_arco: attributi} con TUTTI gli archi paralleli
    # u->v, non un singolo arco. Si prende il piu' corto tra questi.
    return min(d.get("length", 1) for d in graph.get_edge_data(u, v).values())


def _ricostruisci(came_from, start, goal):
    """Ricostruisce il percorso da start a goal risalendo i predecessori."""
    # came_from[nodo] = predecessore lungo il miglior cammino trovato fin li'.
    # Si parte dal goal e si risale a ritroso fino allo start, poi si inverte
    # la lista perche' e' stata costruita al contrario (goal -> ... -> start).
    path, current = [], goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]


def _costo_percorso(graph, path):
    """Somma le lunghezze degli archi del percorso."""
    return sum(_costo_arco(graph, u, v) for u, v in zip(path, path[1:]))


def _metriche(expanded, peak, t0, path, costo):
    """Impacchetta le metriche di una ricerca in un dizionario."""
    return {
        "expanded_nodes": expanded,
        "time_s": time.perf_counter() - t0,
        "cost_m": costo,
        "path_len": len(path),
        "peak_frontier": peak,
    }


# ============================================================
# BEST-FIRST SEARCH GENERICO
# ============================================================
def _best_first(graph, heuristic, start, goal, usa_g, on_expand=None):
    """
    Nucleo comune di A* e Greedy Best-First. La priorita' di ogni nodo e':
        usa_g = True   ->  f(n) = g(n) + h(n)    (A*)
        usa_g = False  ->  f(n) =        h(n)    (Greedy)

    Ritorna (path, costo_totale, metriche), oppure (None, None, None) se il
    goal non e' raggiungibile.
    """
    # FRONTIERA (open set): coda di priorita' implementata con heapq, un
    # min-heap binario su tuple (f, nodo). heapq confronta le tuple in ordine
    # lessicografico, quindi ordina implicitamente per f; heappush/heappop
    # costano O(log n) invece di O(n) per trovare il minimo in una lista.
    open_set = [(heuristic(start, goal), start)]

    came_from = {}            # nodo -> predecessore nel miglior cammino noto
    g_score = {start: 0}      # nodo -> costo REALE minimo noto per raggiungerlo
    closed_set = set()        # nodi gia' espansi definitivamente

    expanded_nodes = 0
    peak_frontier = 1
    t0 = time.perf_counter()

    while open_set:
        # Il picco di occupazione si misura PRIMA del pop, per catturare il
        # momento di massimo affollamento della coda.
        peak_frontier = max(peak_frontier, len(open_set))

        _, current = heapq.heappop(open_set)   # estrae il nodo con f minimo

        # LAZY DELETION. heapq non permette di aggiornare la priorita' di un
        # elemento gia' inserito: quando si trova un cammino migliore verso un
        # nodo gia' in coda, si inserisce una NUOVA copia con f piu' basso e si
        # lascia quella vecchia nell'heap. La copia migliore, avendo f minore,
        # viene estratta per prima. Quando (piu' tardi) esce anche la copia
        # vecchia, il nodo e' gia' in closed_set: questo controllo la scarta
        # senza contarla come una seconda espansione.
        if current in closed_set:
            continue

        closed_set.add(current)
        expanded_nodes += 1

        if on_expand is not None:
            on_expand(current)

        # GOAL TEST ALL'ESPANSIONE (non alla generazione). Estrarre il goal
        # dalla frontiera garantisce che nessun altro nodo in coda abbia f
        # minore: con un'euristica ammissibile e consistente, questo assicura
        # che il cammino appena trovato sia ottimo.
        if current == goal:
            path = _ricostruisci(came_from, start, goal)
            costo = _costo_percorso(graph, path)
            return path, costo, _metriche(expanded_nodes, peak_frontier, t0, path, costo)

        # ESPANSIONE: genera i successori del nodo corrente. Su un grafo
        # diretto, neighbors() segue solo gli archi USCENTI (i sensi unici
        # sono quindi rispettati automaticamente).
        for neighbor in graph.neighbors(current):
            # Un nodo gia' chiuso ha (con h consistente) il suo g_score gia'
            # ottimo: e' una GRAPH SEARCH, i nodi chiusi non si riaprono mai.
            if neighbor in closed_set:
                continue

            # Costo del cammino start -> ... -> current -> neighbor.
            tentative_g = g_score[current] + _costo_arco(graph, current, neighbor)

            # RILASSAMENTO: si aggiorna il vicino solo se questo e' il primo
            # cammino trovato verso di lui, oppure se e' strettamente
            # migliore del migliore noto finora.
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # Questa riga e' l'UNICA differenza tra A* e Greedy: con
                # usa_g=True la priorita' include il costo gia' pagato
                # (tentative_g), con usa_g=False conta solo la stima h.
                f = (tentative_g if usa_g else 0) + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    # Frontiera esaurita senza mai estrarre il goal: nessun cammino esiste.
    return None, None, None


# ============================================================
# A*
# ============================================================
def a_star(graph, heuristic, start, goal, on_expand=None):
    """
    Ricerca A*: f(n) = g(n) + h(n).

    on_expand: callback opzionale chiamata su ogni nodo nell'ordine di espansione.
    Ritorna (path, costo_totale, metriche).
    """
    return _best_first(graph, heuristic, start, goal, usa_g=True, on_expand=on_expand)


# ============================================================
# GREEDY BEST-FIRST SEARCH
# ============================================================
def greedy(graph, heuristic, start, goal, on_expand=None):
    """
    Ricerca Greedy Best-First: f(n) = h(n).

    Ignora il costo gia' pagato e sceglie sempre il nodo che l'euristica
    stima piu' vicino al goal: espande pochi nodi ma non garantisce
    l'ottimalita' del cammino trovato.

    Ritorna (path, costo_totale, metriche), stesso formato di a_star().
    """
    return _best_first(graph, heuristic, start, goal, usa_g=False, on_expand=on_expand)


# ============================================================
# BREADTH-FIRST SEARCH
# ============================================================
def bfs(graph, start, goal, on_expand=None):
    """
    Ricerca in ampiezza: minimizza il numero di archi del percorso, non il
    costo in metri.

    Ritorna (path, costo_totale, metriche), stesso formato di a_star().
    """
    t0 = time.perf_counter()

    # Caso degenere: start e goal coincidono. Va gestito a parte perche' il
    # goal test nel ciclo avviene sui VICINI del nodo estratto, quindi non
    # scatterebbe mai se il goal fosse gia' lo start.
    if start == goal:
        return [start], 0, _metriche(0, 1, t0, [start], 0)

    # FRONTIERA FIFO: una deque, non una coda di priorita'. popleft() costa
    # O(1) (contro l'O(n) di list.pop(0)), e l'assenza di priorita' fa si'
    # che i nodi escano nello stesso ordine in cui sono entrati: e' questo
    # che produce un'esplorazione per livelli, cioe' la BFS.
    frontiera = deque([start])
    came_from = {}
    raggiunti = {start}   # nodi gia' generati (in coda o gia' espansi): impedisce
                          # che lo stesso nodo venga accodato piu' volte

    expanded_nodes = 0
    peak_frontier = 1

    while frontiera:
        peak_frontier = max(peak_frontier, len(frontiera))

        current = frontiera.popleft()   # il nodo piu' vecchio in coda (FIFO)
        expanded_nodes += 1
        if on_expand is not None:
            on_expand(current)

        for neighbor in graph.neighbors(current):
            if neighbor in raggiunti:
                continue

            came_from[neighbor] = current

            # GOAL TEST ALLA GENERAZIONE (non all'espansione come in A*). E'
            # corretto perche' la BFS esplora per livelli con costi unitari:
            # il primo cammino che tocca il goal ha gia' il numero minimo di
            # archi possibile, quindi non serve aspettare che il goal esca
            # dalla coda.
            if neighbor == goal:
                path = _ricostruisci(came_from, start, goal)
                costo = _costo_percorso(graph, path)
                return path, costo, _metriche(expanded_nodes, peak_frontier, t0, path, costo)

            raggiunti.add(neighbor)
            frontiera.append(neighbor)

    return None, None, None


# ============================================================
# METRICHE DERIVATE
# ============================================================
def effective_branching_factor(n_expanded, depth, tol=1e-9):
    """
    Calcola il fattore di ramificazione effettivo b*, risolvendo per bisezione
    l'equazione 1 + b* + b*^2 + ... + b*^depth = n_expanded + 1.
    """
    if depth <= 0 or n_expanded <= 0:
        return 0.0

    # Intervallo di ricerca per la bisezione. lo = 1 perche' b* non puo' mai
    # essere minore di 1. hi e' un limite superiore ricavato osservando che la
    # somma geometrica e' sempre >= al suo ultimo termine b^depth: da
    # b^depth <= n_expanded+1 segue b <= (n_expanded+1)^(1/depth). Usare
    # questo bound (invece di un valore grande a piacere) evita overflow nel
    # calcolo di somma(), che eleva b alla potenza depth+1.
    lo = 1.0
    hi = (n_expanded + 1) ** (1.0 / depth)

    if hi <= lo:
        return 1.0

    def somma(b):
        """Somma geometrica 1 + b + ... + b^depth."""
        # Per b -> 1 la formula chiusa (b^(depth+1)-1)/(b-1) e' una forma
        # 0/0 numericamente instabile; il limite per b->1 e' depth+1 (sono
        # depth+1 termini tutti pari a 1).
        if b - 1.0 < 1e-12:
            return depth + 1.0
        return (b ** (depth + 1) - 1.0) / (b - 1.0)

    target = n_expanded + 1

    # BISEZIONE: somma(b) e' monotona crescente in b, quindi si puo' dimezzare
    # l'intervallo [lo, hi] a ogni iterazione confrontando somma(mid) con il
    # target, finche' la precisione richiesta (tol) non e' raggiunta.
    for _ in range(200):   # tetto di sicurezza sul numero di iterazioni
        if hi - lo <= tol:
            break
        mid = (lo + hi) / 2
        if somma(mid) < target:
            lo = mid   # servono piu' nodi: b* e' maggiore di mid
        else:
            hi = mid   # troppi nodi: b* e' minore o uguale a mid

    return (lo + hi) / 2
