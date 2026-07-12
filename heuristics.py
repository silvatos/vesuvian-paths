import os
import re
import csv
import math
import random
import statistics

import osmnx as ox
import matplotlib
matplotlib.use("Agg")          # backend senza finestre: salva su file invece di aprire
                               # una GUI. Va impostato PRIMA di importare pyplot.
import matplotlib.pyplot as plt

from algoritmi import a_star, effective_branching_factor, greedy, bfs
from heuristics import (make_h_zero, make_h_euclidean, make_h_landmark,
                        make_h_weighted, _metri_per_grado_lat, _metri_per_grado_lon)
from video import anima_esplorazione


# ============================================================
# CONFIGURAZIONE
# ============================================================
CARTELLA_MAPPE = "mappe"
CARTELLA_RISULTATI = "risultati"

K_LANDMARK = 8      # landmark precalcolati per l'euristica ALT.
                    # Piu' landmark = euristica piu' informata, MA preprocessing
                    # piu' lungo (2*K Dijkstra completi) e piu' memoria.
N_ATTIVI = 3        # quanti landmark usare davvero per ogni coppia (start, goal).
                    # Riduce il costo per nodo senza perdere quasi informativita'.
W = 2.0             # peso dell'euristica pesata. w > 1 -> inammissibile per
                    # costruzione: perde l'ottimalita', guadagna velocita'.

# Bin di distanza in linea d'aria (km). Vanno tarati sulla dimensione della citta':
# per Napoli (~12 km di estensione) servono bin fino a 12; per Torre del Greco
# basterebbe fermarsi a 6, altrimenti gli ultimi bin resterebbero vuoti.
BIN_KM = [(0.5, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 12)]
COPPIE_PER_BIN = 20   # piu' coppie = mediane piu' stabili, ma benchmark piu' lento
SEED = 42             # seed fisso: gli esperimenti devono essere RIPRODUCIBILI


# ============================================================
# CARICAMENTO MAPPA
# ============================================================
def _nome_file(luogo):
    """Trasforma il nome di un luogo in un nome di file sicuro (slug)."""
    # "Napoli, Italy" -> "napoli_italy". La regex sostituisce ogni sequenza di
    # caratteri NON alfanumerici con un underscore; strip("_") toglie quelli
    # eventualmente rimasti agli estremi. Serve perche' virgole e spazi sono
    # pessimi nomi di file.
    slug = re.sub(r"[^a-z0-9]+", "_", luogo.lower()).strip("_")
    return os.path.join(CARTELLA_MAPPE, f"{slug}.graphml")


def carica_grafo():
    """
    Carica il grafo dalla cartella 'mappe/' se gia' scaricato, altrimenti lo
    scarica da OSM e lo salva. Ripete finche' il luogo non viene trovato.
    """
    os.makedirs(CARTELLA_MAPPE, exist_ok=True)   # exist_ok: non fallisce se c'e' gia'

    while True:   # si ripete finche' l'utente non inserisce un luogo valido
        luogo = input("Città o area da caricare (es. 'Torre del Greco, Italy'): ").strip()
        if not luogo:
            print("Il campo non può essere vuoto.")
            continue

        percorso = _nome_file(luogo)

        # CACHE SU DISCO. Le API di OpenStreetMap hanno rate limiting e il download
        # di una citta' richiede decine di secondi: riscaricare a ogni avvio sarebbe
        # lento e scortese verso i loro server.
        if os.path.exists(percorso):
            print(f"Carico la mappa da '{percorso}' (nessun download necessario)...")
            graph = ox.load_graphml(percorso)
        else:
            print(f"Download della mappa in corso ({luogo})...")
            try:
                # network_type="drive": solo strade percorribili in AUTO (niente
                # sentieri, scale, piste ciclabili). E' anche cio' che rende il grafo
                # ORIENTATO: OSMnx legge il tag 'oneway' e per i sensi unici crea un
                # solo arco, nella direzione consentita.
                graph = ox.graph_from_place(luogo, network_type="drive")
            except Exception:
                # Nominatim non ha trovato il luogo: si riprova invece di crashare.
                print(f"  Luogo non trovato: '{luogo}'. Prova ad essere più preciso "
                      f"(es. aggiungi la nazione: 'Napoli, Italy').")
                continue
            ox.save_graphml(graph, percorso)
            print(f"Mappa salvata in locale come '{percorso}'.")

        # Componente FORTEMENTE connessa piu' grande: il grafo grezzo di OSM ha
        # frammenti isolati e nodi senza uscita, che renderebbero irraggiungibili
        # alcune coppie (start, goal) e farebbero fallire il benchmark.
        # "Fortemente" (e non semplicemente "connessa") perche' il grafo e' diretto:
        # serve che da ogni nodo si raggiunga ogni altro SEGUENDO IL VERSO degli archi.
        graph = ox.truncate.largest_component(graph, strongly=True)
        return graph, luogo


def chiedi_nodo(graph, etichetta):
    """Geocodifica un indirizzo con Nominatim e restituisce il nodo piu' vicino."""
    while True:
        query = input(f"{etichetta} (via, indirizzo o punto di interesse): ").strip()
        if not query:
            print("Il campo non può essere vuoto.")
            continue
        try:
            # ox.geocode restituisce la coppia (LATITUDINE, LONGITUDINE), in
            # quest'ordine. Attenzione: nearest_nodes vuole invece i parametri
            # nominati X=longitudine, Y=latitudine. Invertirli non da' errore:
            # restituisce silenziosamente un nodo a caso dall'altra parte del mondo.
            y, x = ox.geocode(query)
        except Exception:
            print(f"  Non trovato: '{query}'. Prova ad essere più preciso.")
            continue
        print(f"  Trovato: lat={y:.5f}, lon={x:.5f}")

        # Le coordinate geocodificate quasi mai coincidono con un nodo del grafo:
        # i nodi sono gli INCROCI, mentre un indirizzo sta a meta' di una strada.
        # nearest_nodes fa lo "snapping" all'incrocio piu' vicino.
        return ox.nearest_nodes(graph, X=x, Y=y)


# ============================================================
# REGISTRO DEGLI ALGORITMI
# ============================================================
def costruisci_algoritmi(G):
    """
    Costruisce le euristiche (una volta sola: il preprocessing ALT e' costoso)
    e restituisce un dizionario nome -> funzione(start, goal, on_expand).

    Le lambda uniformano le firme diverse (a_star e greedy vogliono l'euristica,
    bfs no), cosi' il ciclo di esecuzione resta uno solo.
    """
    print(f"\nPreprocessing euristica ALT ({K_LANDMARK} landmark, "
          f"{2 * K_LANDMARK} Dijkstra completi)...")

    # Le euristiche si costruiscono QUI, FUORI dal ciclo sulle coppie. Non e'
    # un'ottimizzazione: e' l'unica cosa che rende il benchmark eseguibile. Il
    # preprocessing di ALT (2*K Dijkstra completi) ripetuto per ognuna delle
    # centinaia di coppie richiederebbe ore.
    h_zero = make_h_zero()
    h_eucl = make_h_euclidean(G)
    h_alt = make_h_landmark(G, k=K_LANDMARK, n_attivi=N_ATTIVI, seed=SEED)
    h_pes = make_h_weighted(h_eucl, W)   # e' h_eucl moltiplicata per W
    print("Preprocessing completato.")

    # DIZIONARIO nome -> funzione. Le lambda sono CLOSURE: catturano G e l'euristica
    # dall'ambiente e se le portano dietro, cosi' dall'esterno tutte e sei accettano
    # la stessa firma (start, goal, on_expand) e il ciclo di esecuzione non ha bisogno
    # di alcun "if". oe=None ha un default perche' in benchmark il callback non serve.
    #
    # ORDINE IMPORTANTE: Dijkstra deve stare PER PRIMO, perche' esegui_tutti usa il
    # primo risultato come riferimento di ottimalita'.
    algoritmi = {
        "A* Zero (Dijkstra)": lambda s, g, oe=None: a_star(G, h_zero, s, g, on_expand=oe),
        "A* Euclidea":        lambda s, g, oe=None: a_star(G, h_eucl, s, g, on_expand=oe),
        "A* Landmark (ALT)":  lambda s, g, oe=None: a_star(G, h_alt, s, g, on_expand=oe),
        # f-string: {W:g} stampa "2" invece di "2.0" (il formato 'g' toglie gli zeri
        # inutili). Cosi' la chiave si adegua da sola se cambi la costante W.
        f"A* Pesata (w={W:g})": lambda s, g, oe=None: a_star(G, h_pes, s, g, on_expand=oe),
        "Greedy":             lambda s, g, oe=None: greedy(G, h_eucl, s, g, on_expand=oe),
        "BFS":                lambda s, g, oe=None: bfs(G, s, g, on_expand=oe),
    }
    return algoritmi, h_alt


def esegui_tutti(algoritmi, h_alt, start, goal, raccogli_espansi=False):
    """
    Esegue tutti gli algoritmi sulla STESSA coppia (confronto appaiato: elimina
    la variabilita' dovuta alla scelta di start/goal).

    Restituisce (lista_risultati, percorso_ottimo). Il riferimento di ottimalita'
    e' il costo di Dijkstra, che e' il primo della lista ed e' garantito ottimo.
    """
    # ALT sceglie i landmark "attivi" in base alla coppia: va fatto una volta
    # sola prima della ricerca, non a ogni nodo.
    h_alt.preprocess(start, goal)

    risultati = []
    costo_ottimo = None
    percorso_ottimo = None

    for nome, fn in algoritmi.items():
        # In benchmark NON raccogliamo l'ordine di espansione: farlo per centinaia
        # di run significherebbe accumulare liste da decine di migliaia di nodi
        # ciascuna, per poi buttarle. Serve solo in demo, per i video.
        espansi = [] if raccogli_espansi else None
        cb = espansi.append if raccogli_espansi else None

        path, cost, m = fn(start, goal, cb)

        if path is None:
            # Coppia irraggiungibile. Non dovrebbe capitare dopo largest_component,
            # ma se capita scartiamo l'INTERA coppia: tenere solo alcuni algoritmi
            # romperebbe il confronto appaiato.
            return None, None

        if costo_ottimo is None:
            # Prima iterazione = Dijkstra (vedi ordine del dizionario). Il suo costo
            # e' l'ottimo garantito, e diventa il riferimento per l'errore di tutti
            # gli altri.
            costo_ottimo = cost
            percorso_ottimo = path

        risultati.append({
            "algoritmo": nome,
            "costo_m": cost,
            # Errore relativo rispetto all'ottimo. Vale 0 per gli algoritmi ottimi
            # (Dijkstra, A* euclidea, A* ALT) e > 0 per gli altri.
            "errore_%": 100 * (cost - costo_ottimo) / costo_ottimo,
            # Numero di ARCHI (non di nodi). Questa colonna esiste per rendere
            # leggibile BFS: senza, il suo costo alto sembra solo un fallimento, e
            # non si vede che sul numero di archi e' invece OTTIMO.
            "archi_path": m["path_len"] - 1,
            "nodi_espansi": m["expanded_nodes"],
            "tempo_ms": m["time_s"] * 1000,
            "picco_frontiera": m["peak_frontier"],
            "b_star": effective_branching_factor(m["expanded_nodes"], m["path_len"] - 1),
            "path": path,        # serve solo alla demo (disegno mappa, video)
            "espansi": espansi,  # idem
        })

    return risultati, percorso_ottimo


# ============================================================
# MODALITA' 1 — DEMO SU UNA SINGOLA COPPIA
# ============================================================
def modalita_demo(G, algoritmi, h_alt):
    """Una coppia scelta dall'utente: tabella + video + immagini + grafici a barre."""
    start = chiedi_nodo(G, "Punto di partenza")
    goal = chiedi_nodo(G, "Punto di arrivo")

    # raccogli_espansi=True: qui SERVE l'ordine di espansione, per animare i video.
    risultati, percorso_ottimo = esegui_tutti(algoritmi, h_alt, start, goal,
                                              raccogli_espansi=True)
    if risultati is None:
        print("Nessun percorso tra i due punti.")
        return

    # --- tabella a terminale ---
    # b* si stampa con 6 decimali: su percorsi lunghi tutti i valori si schiacciano
    # tra 1.00 e 1.03, e con 2 decimali sembrerebbero identici.
    print(f"\n{'algoritmo':22}{'costo (m)':>11}{'err %':>8}{'archi':>7}"
          f"{'espansi':>10}{'ms':>9}{'picco':>8}{'b*':>11}")
    print("-" * 86)
    for r in risultati:
        print(f"{r['algoritmo']:22}{r['costo_m']:>11.0f}{r['errore_%']:>+8.2f}"
              f"{r['archi_path']:>7}{r['nodi_espansi']:>10}{r['tempo_ms']:>9.1f}"
              f"{r['picco_frontiera']:>8}{r['b_star']:>11.6f}")

    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)

    grafici_demo(risultati)

    # --- immagini della mappa ---
    mappa_file = os.path.join(CARTELLA_RISULTATI, "mappa.png")
    percorso_file = os.path.join(CARTELLA_RISULTATI, "percorso.png")

    print(f"\nGenerazione di '{mappa_file}'...")
    # node_size=0: su una citta' i nodi sarebbero decine di migliaia di puntini
    # illeggibili. edge_linewidth=0.3: strade sottili, cosi' il percorso risalta.
    ox.plot_graph(G, show=False, save=True, filepath=mappa_file,
                  node_size=0, edge_linewidth=0.3)

    print(f"Generazione di '{percorso_file}'...")
    # Si disegna il percorso OTTIMO (quello di Dijkstra), non l'ultimo calcolato:
    # Greedy, BFS e la pesata restituiscono percorsi subottimi.
    ox.plot_graph_route(G, percorso_ottimo, show=False, save=True,
                        filepath=percorso_file, node_size=0, edge_linewidth=0.3,
                        route_linewidth=1.5, orig_dest_size=20)

    # --- video dell'esplorazione, uno per algoritmo ---
    for r in risultati:
        slug = re.sub(r"[^a-z0-9]+", "_", r["algoritmo"].lower()).strip("_")
        video_file = os.path.join(CARTELLA_RISULTATI, f"esplorazione_{slug}.mp4")

        # CAMPIONAMENTO DEI FRAME. Su Napoli, Dijkstra espande decine di migliaia di
        # nodi: un frame per nodo renderebbe il video ingestibile (o esaurirebbe la
        # RAM). Prendiamo un nodo ogni PASSO, per un totale di ~300 frame,
        # indipendentemente dalla dimensione del grafo.
        PASSO = max(1, len(r["espansi"]) // 300)
        frames = r["espansi"][::PASSO]

        print(f"Generazione video '{video_file}' "
              f"({len(r['espansi'])} nodi -> {len(frames)} frame)...")
        anima_esplorazione(G, frames, start, goal, r["path"], video_file)

    print(f"\nTutto salvato in '{CARTELLA_RISULTATI}/'.")


def grafici_demo(risultati):
    """
    Barre affiancate: le metriche del confronto su una singola coppia.
    Scala logaritmica dove gli ordini di grandezza sono molto diversi (Dijkstra
    espande migliaia di nodi, ALT poche centinaia: in scala lineare le barre
    piccole sparirebbero).
    """
    nomi = [r["algoritmo"] for r in risultati]

    # (chiave_nel_dizionario, titolo_del_pannello, usare_scala_log?)
    metriche = [
        ("nodi_espansi",    "Nodi espansi",                   True),
        ("tempo_ms",        "Tempo di esecuzione (ms)",       True),
        ("costo_m",         "Costo del percorso (m)",         False),
        ("b_star",          "Effective branching factor b*",  False),
        ("picco_frontiera", "Memoria di picco (frontiera)",   True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))   # 6 riquadri, 5 metriche

    # zip si ferma alla lista piu' corta: il sesto 'ax' non viene mai toccato.
    for ax, (chiave, titolo, log) in zip(axes.flat, metriche):
        valori = [r[chiave] for r in risultati]
        ax.bar(range(len(nomi)), valori)

        # b* vale ~1.0x per tutti gli algoritmi: partendo da zero, le barre
        # sarebbero visivamente identiche. Zoomiamo sul range effettivo, cosi'
        # le differenze (1.019 vs 1.005) diventano leggibili.
        if chiave == "b_star":
            lo, hi = min(valori), max(valori)
            margine = (hi - lo) * 0.15 or 0.001
            ax.set_ylim(max(1.0, lo - margine), hi + margine)

        ax.set_title(titolo)
        ax.set_xticks(range(len(nomi)))
        ax.set_xticklabels(nomi, rotation=30, ha="right", fontsize=8)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    axes.flat[5].axis("off")   # nasconde il sesto riquadro, rimasto senza metrica

    fig.suptitle("Confronto su una singola coppia (start, goal)")
    fig.tight_layout()
    out = os.path.join(CARTELLA_RISULTATI, "confronto_singolo.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)             # libera la figura: senza, matplotlib accumula memoria
    print(f"Grafici salvati in '{out}'")


# ============================================================
# MODALITA' 2 — BENCHMARK SU DISTANZE CRESCENTI
# ============================================================
def distanza_aerea_km(G, u, v):
    """Distanza in linea d'aria tra due nodi, in km (per assegnare le coppie ai bin)."""
    y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
    y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]

    # Stessa proiezione equirettangolare locale usata dall'euristica euclidea:
    # i gradi vanno convertiti in metri, e il fattore di conversione della
    # longitudine dipende dalla latitudine (i meridiani convergono).
    lm = (y1 + y2) / 2
    dy = (y1 - y2) * _metri_per_grado_lat(lm)
    dx = (x1 - x2) * _metri_per_grado_lon(lm)
    return math.hypot(dx, dy) / 1000.0


def campiona_coppie(G):
    """
    Campiona coppie casuali e le assegna ai bin di distanza in linea d'aria.

    NOTA METODOLOGICA: la variabile indipendente e' la distanza AEREA, non il
    numero di nodi espansi. Quest'ultimo e' un RISULTATO dell'algoritmo: usarlo
    come ascissa sarebbe un ragionamento circolare (staresti misurando i nodi
    espansi in funzione dei nodi espansi).
    """
    rng = random.Random(SEED)   # generatore LOCALE con seed fisso: non tocca lo stato
                                # globale di random, e rende il campionamento riproducibile
    nodi = list(G.nodes)
    coppie = {b: [] for b in BIN_KM}

    # Si estraggono coppie a caso e si vede in quale bin cadono. I bin corti
    # (0.5-1 km) sono i piu' rari, perche' due nodi presi a caso in una citta'
    # grande tendono a essere lontani: per questo serve un tetto ampio di tentativi.
    tentativi, max_tentativi = 0, COPPIE_PER_BIN * len(BIN_KM) * 5000

    # Ci si ferma quando TUTTI i bin sono pieni, oppure quando si esauriscono
    # i tentativi (un bin puo' restare vuoto se la citta' e' troppo piccola
    # per contenere quelle distanze: vedi il commento su BIN_KM).
    while tentativi < max_tentativi and any(len(v) < COPPIE_PER_BIN for v in coppie.values()):
        tentativi += 1
        s, g = rng.sample(nodi, 2)   # due nodi DISTINTI
        d = distanza_aerea_km(G, s, g)
        for b in BIN_KM:
            if b[0] <= d < b[1] and len(coppie[b]) < COPPIE_PER_BIN:
                coppie[b].append((s, g, d))
                break   # un bin solo per coppia: i bin non si sovrappongono

    return coppie


def modalita_benchmark(G, algoritmi, h_alt):
    """Molte coppie su distanze crescenti: CSV + grafici a linee con mediana e quartili."""
    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)
    coppie = campiona_coppie(G)
    righe = []   # una riga per (coppia, algoritmo): formato "lungo", comodo per il CSV

    for b in BIN_KM:
        print(f"--- bin {b[0]}-{b[1]} km: {len(coppie[b])} coppie ---")
        for s, g, d_km in coppie[b]:
            # raccogli_espansi=False: qui i video non servono.
            risultati, _ = esegui_tutti(algoritmi, h_alt, s, g)
            if risultati is None:
                continue    # coppia irraggiungibile: saltata del tutto

            for r in risultati:
                righe.append({
                    "bin": f"{b[0]}-{b[1]}",
                    "bin_mid": (b[0] + b[1]) / 2,   # ascissa del punto nel grafico
                    "dist_aerea_km": round(d_km, 3),
                    "algoritmo": r["algoritmo"],
                    "costo_m": round(r["costo_m"], 1),
                    "errore_%": round(r["errore_%"], 2),
                    "archi_path": r["archi_path"],
                    "nodi_espansi": r["nodi_espansi"],
                    "tempo_ms": round(r["tempo_ms"], 3),
                    "picco_frontiera": r["picco_frontiera"],
                    "b_star": round(r["b_star"], 6),   # 6 decimali: vedi commento sopra
                })
                # NB: 'path' e 'espansi' NON finiscono nel CSV (sarebbero liste
                # da migliaia di elementi per riga).

    if not righe:
        print("Nessuna coppia valida campionata.")
        return

    # --- CSV con i dati GREZZI ---
    # Vanno allegati alla relazione: permettono a chiunque di rifare l'analisi
    # (medie, mediane, test statistici) senza rieseguire il benchmark.
    csv_path = os.path.join(CARTELLA_RISULTATI, "risultati.csv")
    with open(csv_path, "w", newline="") as f:   # newline="": evita righe vuote su Windows
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"\nDati grezzi salvati in '{csv_path}' ({len(righe)} righe)")

    grafici_benchmark(righe)


def grafici_benchmark(righe):
    """
    Un pannello per metrica, con la distanza aerea sull'asse x.

    Si riportano MEDIANA e BANDA INTERQUARTILE, non la singola run: su rete
    stradale la varianza tra coppie diverse e' enorme (dipende da dove capitano
    start e goal rispetto alle arterie principali) e un grafico a singola run
    sarebbe rumore puro.
    """
    # dict.fromkeys preserva l'ordine di inserimento e rimuove i duplicati:
    # cosi' la legenda segue l'ordine del dizionario degli algoritmi, non uno casuale.
    algos = list(dict.fromkeys(r["algoritmo"] for r in righe))
    bins = sorted({r["bin_mid"] for r in righe})

    metriche = [
        ("nodi_espansi",    "Nodi espansi",                   True),
        ("tempo_ms",        "Tempo di esecuzione (ms)",       True),
        ("costo_m",         "Costo del percorso (m)",         False),
        ("b_star",          "Effective branching factor b*",  False),
        ("picco_frontiera", "Memoria di picco (frontiera)",   True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    for ax, (chiave, titolo, log) in zip(axes.flat, metriche):
        for a in algos:
            xs, med, q1s, q3s = [], [], [], []

            for b in bins:
                # Tutti i valori di QUESTA metrica, per QUESTO algoritmo, in QUESTO bin.
                # Ordinati, perche' servono i quartili.
                vals = sorted(r[chiave] for r in righe
                              if r["algoritmo"] == a and r["bin_mid"] == b)
                if not vals:
                    continue   # bin vuoto (citta' troppo piccola per quelle distanze)

                xs.append(b)
                med.append(statistics.median(vals))
                q1s.append(vals[len(vals) // 4])         # primo quartile (25%)
                q3s.append(vals[(3 * len(vals)) // 4])   # terzo quartile (75%)

            ax.plot(xs, med, marker="o", label=a)        # la mediana: la linea
            ax.fill_between(xs, q1s, q3s, alpha=0.12)    # la dispersione: la banda

        ax.set_title(titolo)
        ax.set_xlabel("Distanza in linea d'aria (km)")
        if log:
            # gli ordini di grandezza tra Dijkstra e ALT sono tali che in scala
            # lineare le curve degli algoritmi informati sarebbero schiacciate a zero
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    axes.flat[0].legend(fontsize=8)   # una sola legenda, sul primo pannello
    axes.flat[5].axis("off")          # nasconde il sesto riquadro (5 metriche, 6 posti)

    fig.suptitle("Confronto su distanze crescenti (mediana e banda interquartile)")
    fig.tight_layout()
    out = os.path.join(CARTELLA_RISULTATI, "confronto_distanze.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Grafici salvati in '{out}'")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":   # eseguito solo se lanciato direttamente, non se importato
    G, luogo = carica_grafo()
    print(f"Mappa pronta: {len(G.nodes)} nodi, {len(G.edges)} archi.")

    # Le euristiche si costruiscono UNA VOLTA, prima di scegliere la modalita':
    # il preprocessing ALT e' il pezzo piu' costoso dell'intero programma.
    algoritmi, h_alt = costruisci_algoritmi(G)

    print("\nModalità:")
    print("  1) Demo   — una coppia scelta da te: tabella, grafici, immagini, video")
    print("  2) Benchmark — coppie casuali su distanze crescenti: CSV e grafici")
    scelta = input("Scelta [1/2]: ").strip()

    if scelta == "2":
        modalita_benchmark(G, algoritmi, h_alt)
    else:
        modalita_demo(G, algoritmi, h_alt)