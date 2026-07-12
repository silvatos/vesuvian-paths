import heapq
import time
from collections import deque


# ============================================================
# UTILITY CONDIVISE
# ============================================================
def _costo_arco(graph, u, v):
    """
    Costo dell'arco u->v in metri.
    In un MultiDiGraph tra due nodi possono esistere ARCHI PARALLELI (due strade
    distinte che collegano gli stessi incroci): prendiamo il piu' corto, che e'
    la scelta corretta in un problema di minimo costo.
    """
    # get_edge_data(u, v) NON restituisce un singolo arco ma un dizionario
    # {chiave_arco: attributi} con TUTTI gli archi paralleli u->v.
    # .values() scorre gli attributi di ciascuno; d["length"] e' la lunghezza in
    # metri assegnata da OSMnx. Il default 1 e' una rete di sicurezza per archi
    # eventualmente privi del tag (non dovrebbe mai accadere su un grafo 'drive').
    return min(d.get("length", 1) for d in graph.get_edge_data(u, v).values())


def _ricostruisci(came_from, start, goal):
    """Risale i predecessori dal goal allo start e rovescia il risultato."""
    # Durante la ricerca non memorizziamo interi cammini (sarebbe uno spreco di
    # memoria): teniamo solo, per ogni nodo, il puntatore al suo predecessore.
    # Qui il cammino viene ricostruito all'indietro, dal goal verso lo start.
    path, current = [], goal

    # Ci si ferma sullo start, che e' l'unico nodo SENZA predecessore in came_from.
    # (Non si usa "while current in came_from" perche' su un grafo con cicli lo
    # start potrebbe essersi ritrovato una voce in came_from: il test esplicito
    # su start e' piu' sicuro.)
    while current != start:
        path.append(current)
        current = came_from[current]   # salto al predecessore
    path.append(start)                 # lo start non entra nel while: va aggiunto a mano

    return path[::-1]                  # il cammino e' goal->start: lo rovesciamo


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
    # zip(path, path[1:]) accoppia ogni nodo col successivo:
    #   path      = [A, B, C, D]
    #   path[1:]  = [B, C, D]
    #   zip       = (A,B), (B,C), (C,D)   -> esattamente gli archi del cammino
    return sum(_costo_arco(graph, u, v) for u, v in zip(path, path[1:]))


def _metriche(expanded, peak, t0, path, costo):
    """Impacchetta le metriche nello stesso formato per tutti gli algoritmi."""
    return {
        "expanded_nodes": expanded,                   # quanti nodi sono stati ESPANSI
                                                      # (non generati, non estratti)
        "time_s": time.perf_counter() - t0,           # perf_counter: orologio monotono ad
                                                      # alta risoluzione, immune ai cambi
                                                      # dell'ora di sistema (a differenza
                                                      # di time.time())
        "cost_m": costo,                              # costo reale del cammino, in metri
        "path_len": len(path),                        # numero di NODI (gli archi sono -1)
        "peak_frontier": peak,                        # max nodi contemporaneamente in coda
    }


# ============================================================
# BEST-FIRST SEARCH GENERICO
# ============================================================
def _best_first(graph, heuristic, start, goal, usa_g, on_expand=None):
    """
    Nucleo comune di A* e Greedy. L'UNICA differenza tra i due e' la funzione
    di priorita' con cui si ordina la frontiera:

        usa_g = True   ->  f(n) = g(n) + h(n)    ->  A*
        usa_g = False  ->  f(n) =        h(n)    ->  Greedy Best-First

    e come caso particolare di A*, con h = 0, si ottiene Dijkstra / UCS.
    """
    # FRONTIERA: lista gestita da heapq come min-heap binario.
    # Ogni elemento e' la tupla (f, nodo): Python confronta le tuple in ordine
    # lessicografico, quindi ordina per f. heappop estrae il minimo in O(log n),
    # contro l'O(n) che costerebbe cercare il minimo in una lista normale.
    open_set = [(heuristic(start, goal), start)]

    came_from = {}            # nodo -> predecessore nel miglior cammino noto finora
    g_score = {start: 0}      # nodo -> costo REALE minimo noto per raggiungerlo.
                              # Attenzione: g e' sempre il costo vero, anche in Greedy,
                              # dove pero' NON entra nella priorita'.
    closed_set = set()        # nodi gia' espansi. La sua presenza rende questa una
                              # GRAPH SEARCH (un nodo chiuso non si riapre mai) e non
                              # una tree search. E' lecito solo con h consistente.

    expanded_nodes = 0        # contatore delle espansioni VERE
    peak_frontier = 1         # picco di occupazione della frontiera (proxy della memoria)
    t0 = time.perf_counter()  # istante di partenza per la misura del tempo

    while open_set:
        # Il picco si misura PRIMA del pop, cosi' si cattura il momento di massimo
        # affollamento della coda.
        peak_frontier = max(peak_frontier, len(open_set))

        # Estrae il nodo con f minimo. La f non ci serve piu' (la ricalcoleremmo
        # comunque), quindi la scartiamo con "_".
        _, current = heapq.heappop(open_set)

        # LAZY DELETION: heapq non permette di aggiornare la priorita' di un
        # elemento gia' in coda, quindi quando troviamo un cammino migliore ne
        # inseriamo una copia nuova con f piu' basso e lasciamo la vecchia. La
        # copia aggiornata esce per prima (l'heap estrae il minimo); quando poi
        # esce quella vecchia, il nodo e' gia' chiuso e la scartiamo qui.
        if current in closed_set:
            continue    # copia "stale": non conta come espansione, si passa oltre

        closed_set.add(current)   # da qui in poi il nodo e' definitivamente chiuso
        expanded_nodes += 1       # incrementato DOPO il controllo sopra: contiamo le
                                  # espansioni reali, non i pop scartati

        if on_expand is not None:
            on_expand(current)    # callback esterno (usato per animare l'esplorazione).
                                  # L'algoritmo non sa cosa faccia: separazione delle
                                  # responsabilita'. Non altera la ricerca.

        # GOAL TEST ALL'ESPANSIONE, non alla generazione: e' qui che A* diventa
        # ottimo. Estrarre il goal dalla frontiera garantisce che nessun altro
        # nodo in coda abbia f minore, quindi (con h ammissibile e consistente)
        # nessun cammino migliore puo' esistere.
        if current == goal:
            path = _ricostruisci(came_from, start, goal)
            # Costo ricalcolato sugli archi e non letto da g_score[goal]: cosi' la
            # metrica e' prodotta allo stesso modo per A*, Greedy e BFS, e resta
            # confrontabile anche per gli algoritmi non ottimi.
            costo = _costo_percorso(graph, path)
            return path, costo, _metriche(expanded_nodes, peak_frontier, t0, path, costo)

        # ESPANSIONE: si generano i vicini del nodo corrente.
        # Su un MultiDiGraph .neighbors() restituisce i SUCCESSORI, cioe' segue solo
        # gli archi USCENTI: i sensi unici sono quindi rispettati automaticamente.
        for neighbor in graph.neighbors(current):

            # Un nodo gia' chiuso ha (con h consistente) il suo g gia' ottimo:
            # riconsiderarlo non potrebbe migliorarlo, quindi lo saltiamo.
            if neighbor in closed_set:
                continue

            # Costo del cammino start -> ... -> current -> neighbor passando da qui.
            # "tentative" perche' non e' detto che sia il migliore: lo verifichiamo sotto.
            tentative_g = g_score[current] + _costo_arco(graph, current, neighbor)

            # RILASSAMENTO: si aggiorna solo se questo cammino e' il primo trovato
            # verso 'neighbor', oppure se e' STRETTAMENTE migliore del migliore noto.
            # E' il meccanismo con cui la ricerca corregge progressivamente le stime.
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current    # il cammino migliore passa da 'current'
                g_score[neighbor] = tentative_g  # aggiorno il costo minimo noto

                # <<< L'UNICA RIGA CHE DISTINGUE A* DA GREEDY >>>
                # usa_g=True  -> f = g + h : A* bilancia costo pagato e costo stimato
                # usa_g=False -> f =     h : Greedy guarda SOLO avanti, e dimentica
                #                            quanto ha gia' speso
                f = (tentative_g if usa_g else 0) + heuristic(neighbor, goal)

                # Si inserisce (o si re-inserisce) il nodo in frontiera. Se c'era gia'
                # una copia con f piu' alto, resta li' e verra' scartata al pop
                # (vedi LAZY DELETION sopra).
                heapq.heappush(open_set, (f, neighbor))

    # Frontiera svuotata senza mai raggiungere il goal: non esiste alcun cammino.
    # (Non dovrebbe accadere se il grafo e' stato ridotto alla componente
    # fortemente connessa piu' grande.)
    return None, None, None


# ============================================================
# A*
# ============================================================
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
    # A* = best-first che USA g nella priorita'.
    return _best_first(graph, heuristic, start, goal, usa_g=True, on_expand=on_expand)


# ============================================================
# GREEDY BEST-FIRST SEARCH
# ============================================================
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
    # Greedy = best-first che IGNORA g nella priorita'. Stesso identico codice.
    return _best_first(graph, heuristic, start, goal, usa_g=False, on_expand=on_expand)


# ============================================================
# BREADTH-FIRST SEARCH
# ============================================================
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

    # Caso degenere: start e goal coincidono. Va gestito a parte perche' il goal
    # test qui sotto avviene sui VICINI, quindi non scatterebbe mai sullo start.
    if start == goal:
        return [start], 0, _metriche(0, 1, t0, [start], 0)

    # FRONTIERA FIFO. deque (double-ended queue) perche' popleft() costa O(1),
    # mentre list.pop(0) costerebbe O(n) (deve traslare tutti gli elementi).
    # Nessuna priorita': l'ordine di uscita e' l'ordine di ingresso -> esplorazione
    # per livelli, che e' esattamente cio' che definisce BFS.
    frontiera = deque([start])

    came_from = {}

    # 'raggiunti' contiene i nodi GIA' VISTI (sia quelli in coda sia quelli gia'
    # espansi). Non e' un closed_set: qui un nodo entra appena viene GENERATO, non
    # quando viene espanso. Serve a impedire che lo stesso nodo venga accodato piu'
    # volte, e con costi unitari la prima volta che lo si raggiunge e' gia' la
    # migliore -> non c'e' nulla da riconsiderare.
    raggiunti = {start}

    expanded_nodes = 0
    peak_frontier = 1

    while frontiera:
        peak_frontier = max(peak_frontier, len(frontiera))

        # popleft() estrae il piu' VECCHIO (FIFO). E' l'unica differenza sostanziale
        # rispetto a _best_first, dove heappop estrae il piu' PROMETTENTE.
        current = frontiera.popleft()

        expanded_nodes += 1
        if on_expand is not None:
            on_expand(current)

        for neighbor in graph.neighbors(current):
            if neighbor in raggiunti:
                continue    # gia' visto: e' gia' stato raggiunto con altrettanti
                            # o meno archi, quindi non ci interessa

            came_from[neighbor] = current

            # GOAL TEST ALLA GENERAZIONE. Lecito perche' BFS esplora per livelli:
            # il primo cammino che tocca il goal ha il minimo numero di archi
            # possibile. Non serve aspettare che il goal esca dalla coda (a
            # differenza di A*, dove il costo degli archi NON e' uniforme e un
            # cammino trovato dopo puo' essere piu' economico).
            if neighbor == goal:
                path = _ricostruisci(came_from, start, goal)
                # Il costo in metri viene calcolato SOLO ORA, a posteriori: BFS non
                # lo ha mai usato per decidere. Serve unicamente a confrontarlo con
                # gli altri algoritmi -> ed e' qui che si vede che il suo cammino,
                # pur avendo meno archi, e' piu' lungo in metri.
                costo = _costo_percorso(graph, path)
                return path, costo, _metriche(expanded_nodes, peak_frontier, t0, path, costo)

            raggiunti.add(neighbor)     # marcato come visto...
            frontiera.append(neighbor)  # ...e accodato in fondo (FIFO)

    return None, None, None


# ============================================================
# METRICHE DERIVATE
# ============================================================
def effective_branching_factor(n_expanded, depth, tol=1e-9):
    """
    Risolve per bisezione  1 + b* + b*^2 + ... + b*^d = N + 1,
    dove N = nodi espansi e d = profondita' della soluzione (len(path) - 1).

    Limite superiore per b*: la somma geometrica e' sempre >= del suo ultimo
    termine, quindi  b*^d <= N + 1  ==>  b* <= (N+1)^(1/d).
    Usare questo bound (invece di hi = N) evita l'OverflowError: con N = 20.000
    e d = 300, (N+1)^(1/300) = 1.034, mentre N^301 non e' rappresentabile.
    """
    # Casi degeneri: senza soluzione o senza espansioni b* non e' definito.
    if depth <= 0 or n_expanded <= 0:
        return 0.0

    # INTERVALLO DI RICERCA per la bisezione.
    # lo = 1: b* non puo' essere minore (un albero con ramificazione < 1 non esiste).
    lo = 1.0
    # hi: vedi il bound nel docstring. Stretto E finito -> niente overflow.
    hi = (n_expanded + 1) ** (1.0 / depth)

    # Se il bound superiore collassa su 1, l'euristica e' (numericamente) perfetta.
    if hi <= lo:
        return 1.0

    def somma(b):
        """Membro sinistro dell'equazione: la somma geometrica 1 + b + ... + b^d."""
        # Per b -> 1 la formula chiusa (b^(d+1)-1)/(b-1) e' una forma 0/0:
        # numericamente instabile. Il limite per b->1 vale d+1 (sono d+1 termini
        # tutti pari a 1), e lo restituiamo direttamente.
        if b - 1.0 < 1e-12:
            return depth + 1.0
        return (b ** (depth + 1) - 1.0) / (b - 1.0)

    target = n_expanded + 1   # membro destro dell'equazione

    # BISEZIONE. Funziona perche' somma(b) e' MONOTONA CRESCENTE in b: se la somma
    # calcolata e' troppo piccola, b* sta piu' in alto (alzo lo); altrimenti piu'
    # in basso (abbasso hi). L'intervallo si dimezza a ogni giro.
    for _ in range(200):          # tetto di sicurezza: evita loop infiniti
        if hi - lo <= tol:        # precisione raggiunta
            break
        mid = (lo + hi) / 2
        if somma(mid) < target:
            lo = mid              # servono piu' nodi -> b* e' maggiore di mid
        else:
            hi = mid              # troppi nodi -> b* e' minore di mid

    return (lo + hi) / 2          # punto medio dell'intervallo finale