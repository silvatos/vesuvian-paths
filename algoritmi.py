import heapq
import time
import math
from collections import deque


# UTILITY CONDIVISE
def _costo_arco(graph, u, v):
    """
    Costo dell'arco u->v in metri.
    In un MultiDiGraph tra due nodi possono esistere ARCHI PARALLELI (due strade
    distinte che collegano gli stessi incroci): prendiamo il piu' corto, che e'
    la scelta corretta in un problema di minimo costo.
    """
    return min(d.get("length", 1) for d in graph.get_edge_data(u, v).values())


def _ricostruisci(came_from, start, goal):
    """Risale i predecessori dal goal allo start e rovescia il risultato."""
    path, current = [], goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]


def _costo_percorso(graph, path):
    """
    Somma le lunghezze degli archi del percorso.

    Serve perche' Greedy e BFS NON minimizzano il costo: non possono usare
    g_score[goal] come fa A*, perche' quel valore non e' garantito ottimo.
    L'unico modo onesto di misurare il percorso che hanno trovato e' ripercorrerlo
    e sommare gli archi. Per A* con euristica ammissibile il risultato coincide
    con g_score[goal], quindi usiamo la stessa funzione per tutti: le metriche
    restano confrontabili.
    """
    return sum(_costo_arco(graph, u, v) for u, v in zip(path, path[1:]))


def _metriche(expanded, peak, t0, path, costo):
    return {
        "expanded_nodes": expanded,
        "time_s": time.perf_counter() - t0,
        "cost_m": costo,
        "path_len": len(path),
        "peak_frontier": peak,
    }


# BEST-FIRST SEARCH GENERICO
def _best_first(graph, heuristic, start, goal, usa_g, on_expand=None):
    """
    Nucleo comune di A* e Greedy. L'UNICA differenza tra i due e' la funzione
    di priorita' con cui si ordina la frontiera:

        usa_g = True   ->  f(n) = g(n) + h(n)    ->  A*
        usa_g = False  ->  f(n) =        h(n)    ->  Greedy Best-First

    e come caso particolare di A*, con h = 0, si ottiene Dijkstra / UCS.
    """
    open_set = [(heuristic(start, goal), start)]  # coda a priorita' (min-heap)
    came_from = {}
    g_score = {start: 0}      # costo REALE gia' pagato per arrivare al nodo
    closed_set = set()        # nodi gia' espansi

    expanded_nodes = 0
    peak_frontier = 1
    t0 = time.perf_counter()

    while open_set:
        peak_frontier = max(peak_frontier, len(open_set))
        _, current = heapq.heappop(open_set)

        # LAZY DELETION: heapq non permette di aggiornare la priorita' di un
        # elemento gia' in coda, quindi quando troviamo un cammino migliore ne
        # inseriamo una copia nuova con f piu' basso e lasciamo la vecchia. La
        # copia aggiornata esce per prima (l'heap estrae il minimo); quando poi
        # esce quella vecchia, il nodo e' gia' chiuso e la scartiamo qui.

        if current in closed_set:
            continue
        closed_set.add(current)
        expanded_nodes += 1
        if on_expand is not None:
            on_expand(current)

        # GOAL TEST ALL'ESPANSIONE, non alla generazione: e' qui che A* diventa
        # ottimo. Estrarre il goal dalla frontiera garantisce che nessun altro
        # nodo in coda abbia f minore, quindi (con h ammissibile e consistente)
        # nessun cammino migliore puo' esistere.
        
        if current == goal:
            path = _ricostruisci(came_from, start, goal)
            costo = _costo_percorso(graph, path)
            return path, costo, _metriche(expanded_nodes, peak_frontier, t0, path, costo)

        for neighbor in graph.neighbors(current):   # su MultiDiGraph: i SUCCESSORI
                                                    # -> i sensi unici sono rispettati
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + _costo_arco(graph, current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # <<< L'UNICA RIGA CHE DISTINGUE A* DA GREEDY >>>
                f = (tentative_g if usa_g else 0) + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None, None, None   # nessun percorso trovato


# A*
def a_star(graph, heuristic, start, goal, on_expand=None):
    """
    f(n) = g(n) + h(n).

    Ottimo se h e' AMMISSIBILE (non sovrastima mai il costo residuo). Poiche'
    usiamo un closed_set e non riapriamo i nodi gia' espansi (graph search),
    serve in realta' la proprieta' piu' forte di CONSISTENZA:
        h(n) <= c(n, n') + h(n')
    che le nostre euristiche (zero, euclidea, ALT) soddisfano tutte.

    Con h = 0 degenera in Dijkstra / Uniform Cost Search.
    Con h moltiplicata per w > 1 (euristica pesata) perde l'ottimalita' ma
    espande molti meno nodi: il costo trovato resta entro un fattore w dall'ottimo.

    on_expand: callback opzionale chiamata su ogni nodo nell'ordine di espansione
               (usata per animare l'esplorazione); non altera l'algoritmo.
    Ritorna (path, costo_totale, metriche).
    """
    return _best_first(graph, heuristic, start, goal, usa_g=True, on_expand=on_expand)


# GREEDY BEST-FIRST SEARCH
def greedy(graph, heuristic, start, goal, on_expand=None):
    """
    f(n) = h(n): ignora completamente il costo gia' pagato e sceglie sempre il
    nodo che SEMBRA piu' vicino al goal.

      - velocissimo: punta dritto verso il goal, espande pochissimi nodi
      - NON ottimo: puo' infilarsi in una strada che "va nella direzione giusta"
        ma e' molto piu' lunga, e non torna mai a riconsiderare la scelta
      - completo qui solo perche' il grafo e' finito e il closed_set impedisce i cicli

    Ritorna (path, costo_totale, metriche), stesso formato di a_star().
    """
    return _best_first(graph, heuristic, start, goal, usa_g=False, on_expand=on_expand)


# BREADTH-FIRST SEARCH
def bfs(graph, start, goal, on_expand=None):
    """
    Coda FIFO, nessuna priorita', nessuna euristica.

    E' ottimo nel NUMERO DI ARCHI, non nel costo: trova il percorso con meno
    incroci, che su una rete stradale puo' essere molto piu' lungo in metri
    (poche strade lunghissime invece di molte stradine corte). E' proprio questo
    il punto del confronto: "ottimo" ha senso solo rispetto a una funzione di
    costo, e sceglierne una sbagliata da' la risposta sbagliata.

    Il GOAL TEST e' fatto alla GENERAZIONE (non all'espansione come in A*):
    con costi unitari il primo cammino che raggiunge il goal e' gia' il piu'
    corto in numero di archi, quindi non serve aspettare che esca dalla coda.

    Curiosita' che emerge dai benchmark: BFS espande quasi quanto Dijkstra ma e'
    molto piu' veloce in millisecondi, perche' una deque costa O(1) contro
    l'O(log n) dell'heap e non chiama mai l'euristica. Nodi espansi e tempo NON
    sono la stessa metrica.
    """
    t0 = time.perf_counter()

    if start == goal:
        return [start], 0, _metriche(0, 1, t0, [start], 0)

    frontiera = deque([start]) 
    came_from = {}
    raggiunti = {start}        

    expanded_nodes = 0
    peak_frontier = 1

    while frontiera:
        peak_frontier = max(peak_frontier, len(frontiera))
        current = frontiera.popleft()
        expanded_nodes += 1
        if on_expand is not None:
            on_expand(current)

        for neighbor in graph.neighbors(current):
            if neighbor in raggiunti:
                continue
            came_from[neighbor] = current

            if neighbor == goal:                    # goal test alla generazione
                path = _ricostruisci(came_from, start, goal)
                costo = _costo_percorso(graph, path)
                return path, costo, _metriche(expanded_nodes, peak_frontier, t0, path, costo)

            raggiunti.add(neighbor)
            frontiera.append(neighbor)

    return None, None, None


# METRICHE DERIVATE
def effective_branching_factor(n_expanded, depth, tol=1e-9):
    """
    Risolve per bisezione  1 + b* + b*^2 + ... + b*^d = N + 1,
    dove N = nodi espansi e d = profondita' della soluzione (len(path) - 1).

    Limite superiore per b*: la somma geometrica e' sempre >= del suo ultimo
    termine, quindi  b*^d <= N + 1  ==>  b* <= (N+1)^(1/d).
    Usare questo bound (invece di hi = N) evita l'OverflowError: con N = 20.000
    e d = 300, (N+1)^(1/300) = 1.034, mentre N^301 non e' rappresentabile.
    """
    if depth <= 0 or n_expanded <= 0:
        return 0.0

    lo = 1.0
    hi = (n_expanded + 1) ** (1.0 / depth)   # bound stretto e sempre finito
    if hi <= lo:
        return 1.0

    def somma(b):
        # per b -> 1 la formula chiusa (b^(d+1)-1)/(b-1) e' instabile (0/0):
        # il limite vale d+1, e lo usiamo direttamente
        if b - 1.0 < 1e-12:
            return depth + 1.0
        return (b ** (depth + 1) - 1.0) / (b - 1.0)

    target = n_expanded + 1
    for _ in range(200):
        if hi - lo <= tol:
            break
        mid = (lo + hi) / 2
        if somma(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def penetranza(n_expanded, path_len):
    """
    nodi_espansi / lunghezza_del_percorso.
      = 1  -> euristica perfetta (espande solo i nodi della soluzione)
      >> 1 -> euristica debole
    E' la metrica che separa nettamente Zero / Euclidea / ALT.
    """
    return n_expanded / path_len if path_len else float("inf")