import math
import random

import networkx as nx


# ============================================================
# NOTA SUL PATTERN: FACTORY + CLOSURE
# ============================================================
# Ogni euristica non e' una funzione, ma una FUNZIONE CHE RESTITUISCE UNA FUNZIONE.
#
# Perche' non un dizionario h[nodo]? Perche' h dipende ANCHE dal goal: servirebbe
# una tabella N x N, impossibile con decine di migliaia di nodi.
#
# Perche' non ricalcolare tutto a ogni chiamata? Perche' il lavoro costoso (il
# precalcolo dei landmark, la cache delle coordinate) e' INDIPENDENTE dal goal e
# va fatto una volta sola.
#
# La factory fa il lavoro pesante e lo CATTURA NELLA CLOSURE; la 'h' restituita
# calcola al volo, solo per i nodi che l'algoritmo incontra davvero.


# ============================================================
# 1. EURISTICA NULLA
# ============================================================
def make_h_zero():
    """Euristica zero: A* degenera in Dijkstra. Baseline di confronto."""
    def h(node, goal):
        # Banalmente AMMISSIBILE: 0 non sovrastima mai nulla, perche' i costi degli
        # archi sono non negativi. E banalmente CONSISTENTE: 0 <= c(n,n') + 0.
        # Massimamente disinformata: A* con h=0 ordina la frontiera per solo g,
        # cioe' espande "a cerchio" in tutte le direzioni -> e' Dijkstra.
        return 0.0
    return h


# ============================================================
# 2. DISTANZA IN LINEA D'ARIA
# ============================================================
def _metri_per_grado_lat(lat_deg):
    """
    Metri per grado di latitudine ALLA LATITUDINE DATA (WGS84).
    """
    # La Terra non e' una sfera ma un ellissoide (schiacciato ai poli), quindi il
    # valore varia con la latitudine. E' una FUNZIONE, non una costante: cambiando
    # citta' (Milano, Palermo...) si adatta da sola, senza toccare il codice.
    #
    # Varia poco: da 111.161 m a Bolzano a 110.950 a Lampedusa (~0,2%). Per questo
    # l'approssimazione "1 grado = 111 km" e' cosi' diffusa. Ma una costante media
    # come 111.320 SOVRASTIMA ovunque in Italia (+0,14% a Bolzano, +0,33% a
    # Lampedusa): piccola, ma sempre nella direzione sbagliata, e quindi una
    # violazione (tecnica) dell'ammissibilita'.
    p = math.radians(lat_deg)
    return 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)


def _metri_per_grado_lon(lat_deg):
    """Metri per grado di longitudine alla latitudine data (i meridiani convergono)."""
    # Qui la variazione e' ENORME: 1 grado di longitudine vale ~111 km all'equatore
    # e ZERO ai poli, perche' i meridiani convergono. In Italia si passa da 76.763 m
    # a Bolzano a 90.729 a Lampedusa: un 18% di differenza. E' questa la correzione
    # che conta davvero.
    p = math.radians(lat_deg)
    return 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)


def make_h_euclidean(graph):
    """
    Distanza in linea d'aria su proiezione equirettangolare locale.
    Ammissibile: la strada tra due punti non e' mai piu' corta della linea d'aria.
    Consistente: la distanza euclidea soddisfa la disuguaglianza triangolare.
    """
    # CACHE DELLE COORDINATE. Costruita una volta sola nella factory: senza, ogni
    # chiamata di h farebbe due lookup dentro il grafo, e h viene chiamata una volta
    # per ogni nodo GENERATO (decine di migliaia di volte per ricerca).
    # In OSMnx: 'y' = latitudine, 'x' = longitudine.
    pos = {n: (d["y"], d["x"]) for n, d in graph.nodes(data=True)}

    def h(node, goal):
        y1, x1 = pos[node]
        y2, x2 = pos[goal]

        # PROIEZIONE EQUIRETTANGOLARE LOCALE: su un'area piccola come una citta' la
        # Terra e' approssimabile con un piano, ma latitudine e longitudine NON
        # valgono lo stesso in metri. Si converte tutto in metri usando la latitudine
        # MEDIA dei due punti come riferimento per il fattore di conversione.
        lat_media = (y1 + y2) / 2
        dy = (y1 - y2) * _metri_per_grado_lat(lat_media)
        dx = (x1 - x2) * _metri_per_grado_lon(lat_media)

        # Pitagora sui due cateti (ora entrambi in METRI, quindi sommabili).
        # UNITA' DI MISURA: h restituisce metri, coerenti con 'length' degli archi.
        # Se g fosse in metri e h in gradi, f = g + h sommerebbe grandezze
        # incommensurabili e A* darebbe risultati privi di senso.
        return math.hypot(dx, dy)

        # AMMISSIBILITA': garantita SOLO perche' il costo degli archi e' la LUNGHEZZA
        # in metri. Se il costo diventasse il TEMPO di percorrenza, h andrebbe divisa
        # per la velocita' massima presente nella rete, altrimenti sovrastimerebbe.
        #
        # PERCHE' E' DEBOLE: in citta' il rapporto strada/linea d'aria e' 1,3-1,5
        # (le strade curvano, i sensi unici obbligano giri). h sottostima quindi
        # sistematicamente del 30-50% -> copre solo ~56% del costo reale, e A* e'
        # costretto a esplorare molto piu' del necessario.

    return h


# ============================================================
# 3. EURISTICA PESATA  (Weighted A*)
# ============================================================
def make_h_weighted(h_base, w):
    """
    f = g + w*h.  Per w > 1 l'euristica NON e' piu' ammissibile: A* perde
    l'ottimalita' ma espande molti meno nodi. Il costo trovato resta comunque
    garantito entro un fattore w dall'ottimo (w-ammissibilita').
    """
    # Non e' una nuova euristica: e' un DECORATORE che ne moltiplica un'altra.
    # h_base e' gia' una funzione (tipicamente h_euclidean), e la richiamiamo dentro.
    def h(node, goal):
        # Moltiplicare per w > 1 fa SOVRASTIMARE di proposito: l'euristica diventa
        # inammissibile PER COSTRUZIONE. Non e' un bug, e' il punto dell'esperimento.
        return w * h_base(node, goal)
    return h

    # UN SOLO PARAMETRO INTERPOLA L'INTERA FAMIGLIA DI ALGORITMI:
    #   w = 0    -> Dijkstra   (f = g)
    #   w = 1    -> A* classico
    #   w -> inf -> Greedy     (g diventa trascurabile rispetto a w*h)
    #
    # w-AMMISSIBILITA': il costo trovato e' <= w * ottimo. Con w=2 il bound teorico
    # e' il DOPPIO dell'ottimo, ma nella pratica si misura +3-5%: il bound e'
    # molto conservativo rispetto al comportamento reale.


# ============================================================
# 4. LANDMARK / ALT
# ============================================================
def scegli_landmark_farthest(graph, k, seed=42):
    """
    Selezione 'farthest' (greedy maxmin): i landmark utili stanno ai BORDI
    del grafo.

    Procedura: parto da un nodo casuale, prendo il piu' lontano da lui, poi
    ogni volta il nodo che MASSIMIZZA la distanza minima dai landmark gia' scelti.
    """
    # PERCHE' NON A CASO: un landmark al CENTRO della mappa da' bound quasi nulli,
    # perche' d(L,goal) - d(L,n) tende a zero quando L sta in mezzo tra i due nodi.
    # I landmark utili stanno ai bordi. Con questa selezione, a parita' di codice in
    # h, i nodi espansi calano di un fattore 2-4 rispetto a random.sample().

    rng = random.Random(seed)   # generatore LOCALE: non tocca lo stato globale di
                                # random, e il seed fisso rende la scelta riproducibile
    seme = rng.choice(list(graph.nodes))

    # Il seme casuale non e' il primo landmark: e' solo il punto di partenza. Il primo
    # landmark vero e' il nodo PIU' LONTANO dal seme, che sta gia' su un bordo.
    d0 = nx.single_source_dijkstra_path_length(graph, seme, weight="length")
    landmarks = [max(d0, key=d0.get)]   # max sulle CHIAVI, ordinate per VALORE (=distanza)

    # dist_min[n] = distanza di n dal landmark PIU' VICINO tra quelli gia' scelti.
    # Con un solo landmark coincide con la distanza da quello.
    dist_min = dict(nx.single_source_dijkstra_path_length(
        graph, landmarks[0], weight="length"))

    while len(landmarks) < k:
        # GREEDY MAXMIN: il prossimo landmark e' il nodo che massimizza la distanza
        # MINIMA dai landmark gia' scelti. Cosi' i landmark si "respingono" a vicenda
        # e finiscono distribuiti sui bordi, invece di ammassarsi in una zona.
        cand = max((n for n in dist_min if n not in landmarks),
                   key=lambda n: dist_min[n], default=None)
        if cand is None:
            break   # niente piu' candidati (grafo troppo piccolo per k landmark)

        landmarks.append(cand)

        # Aggiorno dist_min tenendo conto del nuovo landmark: per ogni nodo, la
        # distanza dal landmark piu' vicino puo' solo DIMINUIRE.
        d_new = nx.single_source_dijkstra_path_length(graph, cand, weight="length")
        for n in dist_min:
            # .get(n, inf): se n non e' raggiungibile dal nuovo landmark, il minimo
            # resta quello vecchio. (Modificare i VALORI mentre si itera sulle chiavi
            # e' lecito in Python; sarebbe illecito aggiungere o togliere chiavi.)
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
    # IDEA DI FONDO: si calcolano le distanze ESATTE tra pochi nodi speciali (i
    # landmark) e tutti gli altri; per disuguaglianza triangolare quelle distanze
    # esatte forniscono un LOWER BOUND sul costo residuo.
    #
    # Intuizione: se il landmark dista 5 km dal goal e il mio nodo dista 2 km dal
    # landmark, allora dal mio nodo al goal ci saranno ALMENO 3 km.
    #
    # E' molto piu' forte dell'euclidea perche' i suoi bound derivano da distanze
    # REALI SUL GRAFO, non dalla geometria: "sa" che di mezzo c'e' un vicolo cieco
    # o un tratto autostradale, cosa che la linea d'aria ignora del tutto.
    # Misurato: copre ~96% del costo reale, contro il ~56% dell'euclidea.

    if landmarks is None:
        landmarks = scegli_landmark_farthest(graph, k, seed=seed)

    # --- PREPROCESSING: 2 Dijkstra COMPLETI per ogni landmark ---

    # d(L, v): Dijkstra sul grafo normale (segue gli archi uscenti)
    dist_from = {L: nx.single_source_dijkstra_path_length(graph, L, weight="length")
                 for L in landmarks}

    # d(v, L): Dijkstra sul grafo TRASPOSTO (segue gli archi entranti).
    # reverse(copy=False) restituisce una vista: non duplica il grafo in memoria.
    #
    # SERVONO ENTRAMBI perche' il grafo e' DIRETTO. Su un grafo non orientato
    # d(L,v) = d(v,L) e basterebbe un Dijkstra: e' l'errore da NON fare, vedi _bound.
    G_rev = graph.reverse(copy=False)
    dist_to = {L: nx.single_source_dijkstra_path_length(G_rev, L, weight="length")
               for L in landmarks}

    INF = float("inf")

    # STATO MUTABILE DELLA CLOSURE. Perche' un dizionario e non una variabile?
    # Perche' preprocess() deve MODIFICARE questo valore, e in Python assegnare a
    # una variabile dentro una funzione annidata crea una variabile LOCALE nuova
    # invece di modificare quella esterna. Un dizionario (oggetto mutabile) aggira
    # il problema senza bisogno di 'nonlocal'.
    stato = {"attivi": landmarks}   # di default: tutti

    def _bound(L, node, goal):
        """Il miglior lower bound su d(node, goal) ricavabile dal landmark L."""
        df, dt = dist_from[L], dist_to[L]
        b = 0.0   # il bound non puo' mai essere negativo (i costi sono >= 0)

        a1 = df.get(goal, INF) - df.get(node, INF)   # d(L,goal) - d(L,n)
        a2 = dt.get(node, INF) - dt.get(goal, INF)   # d(n,L)    - d(goal,L)

        # I controlli != INF servono per i nodi IRRAGGIUNGIBILI da/verso L, che
        # Dijkstra non inserisce affatto nel dizionario (da cui il default INF).
        # Un bound INF sarebbe una sovrastima infinita -> lo scartiamo.
        # (Dopo largest_component(strongly=True) non dovrebbe accadere, ma la
        # protezione resta.)
        if a1 != INF and a1 > b:
            b = a1
        if a2 != INF and a2 > b:
            b = a2
        return b

        # >>> L'ERRORE DA NON FARE <<<
        #     est = abs(df[goal] - df[node])
        # Con il solo dist_from, abs() equivale a prendere il max tra
        # d(L,goal) - d(L,n)  e  d(L,n) - d(L,goal). Il SECONDO termine NON e' un
        # bound valido su grafo diretto: lo sarebbe solo se d(L,n) = d(n,L), cioe'
        # su grafo NON orientato. Con i sensi unici puo' SOVRASTIMARE -> euristica
        # inammissibile -> A* non piu' ottimo.
        # Verificato sperimentalmente: 15 violazioni su 250 coppie, sovrastima fino
        # al +25%, e 17 percorsi subottimi su 200 (errore fino al +34%).

    def preprocess(start, goal):
        """
        Da chiamare UNA VOLTA prima di ogni ricerca: tiene solo gli n_attivi
        landmark che danno il bound migliore per QUESTA coppia (start, goal).
        """
        # PERCHE': h viene chiamata per ogni nodo generato (decine di migliaia di
        # volte). Con k=8 landmark farebbe 8 lookup ogni volta. Ma per una data
        # coppia (start, goal) solo 2-3 landmark danno bound davvero stringenti:
        # tenere solo quelli taglia il costo per nodo senza perdere informativita'.
        #
        # Ordina i landmark per qualita' del bound MISURATA SULLO START (il bound in
        # start e' un buon proxy di quanto quel landmark sara' utile lungo tutta la
        # ricerca) e tiene i migliori n_attivi.
        ordinati = sorted(landmarks, key=lambda L: _bound(L, start, goal), reverse=True)
        stato["attivi"] = ordinati[:n_attivi]

        # ATTENZIONE: modifica uno stato CONDIVISO. Va richiamata prima di OGNI
        # ricerca con una coppia diversa. Se ci si dimentica, h usa i landmark della
        # coppia precedente: il risultato resta CORRETTO (sono comunque bound validi,
        # quindi ammissibili), solo meno efficiente. Non e' thread-safe: due ricerche
        # in parallelo si sovrascriverebbero lo stato a vicenda.

    def h(node, goal):
        best = 0.0
        for L in stato["attivi"]:
            b = _bound(L, node, goal)
            if b > best:
                best = b
        return best

        # IL MAX E' LECITO: il massimo di euristiche ammissibili e' a sua volta
        # ammissibile (e qui anche consistente). Ogni landmark e' un "punto di vista"
        # diverso, e si prende il vincolo piu' stringente.
        # COROLLARIO: piu' landmark non possono MAI peggiorare l'ammissibilita' —
        # possono solo rendere l'euristica piu' informata. Il prezzo e' tutto nel
        # preprocessing e nella memoria.

    # Si "appiccicano" alla funzione h due attributi. In Python le funzioni sono
    # oggetti di prima classe, quindi possono avere attributi: e' il modo piu'
    # semplice per esporre preprocess() al chiamante senza restituire una tupla o
    # creare una classe.
    h.landmarks = landmarks   # utile per plottare i landmark sulla mappa
    h.preprocess = preprocess # il driver la chiama con: if hasattr(h, "preprocess")
    return h

    # ============================================================
    # IL TRADE-OFF CENTRALE DEL PROGETTO
    # ============================================================
    # Costo del preprocessing : 2k Dijkstra completi (+ k per la selezione farthest)
    # Costo in memoria        : 2k dizionari da N voci ciascuno
    # Costo per nodo          : n_attivi lookup, contro 1 operazione dell'euclidea
    #
    # ALT scambia MEMORIA e PREPROCESSING per VELOCITA' DI QUERY. E' esattamente il
    # motivo per cui i navigatori reali lo usano: precalcolano offline, interrogano
    # online. Per questo il tempo di preprocessing va MISURATO e riportato in una
    # colonna separata: dire "ALT e' 6x piu' efficiente" senza dire "ma richiede 40
    # secondi di precalcolo" sarebbe disonesto.
    #
    # CONSEGUENZA DA ASPETTARSI: ALT espande molti meno nodi dell'euclidea, ma puo'
    # risultare PIU' LENTO in millisecondi su percorsi brevi, perche' il costo per
    # nodo e' piu' alto. Non e' un fallimento: E' IL RISULTATO, ed e' il motivo per
    # cui servono sia i nodi espansi sia il tempo.