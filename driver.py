import os
import re
import csv
import random
import statistics

import osmnx as ox
import matplotlib
matplotlib.use("Agg")          # backend senza finestre: salva su file
import matplotlib.pyplot as plt

from algoritmi import a_star, effective_branching_factor, greedy, bfs
from heuristics import (make_h_zero, make_h_euclidean, make_h_landmark,
                        make_h_weighted)
from video import anima_esplorazione

# CONFIGURAZIONE
CARTELLA_MAPPE = "mappe"
CARTELLA_RISULTATI = "risultati"

K_LANDMARK = 8      # landmark precalcolati per l'euristica ALT
N_ATTIVI = 3        # quanti usarne davvero per ogni coppia (start, goal)
W = 2.0             # peso dell'euristica pesata

# Benchmark: bin di distanza in linea d'aria (km) e coppie campionate per bin
BIN_KM = [(0.5, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 12)]
COPPIE_PER_BIN = 20
SEED = 42


# CARICAMENTO MAPPA
def _nome_file(luogo):
    """Trasforma il nome di un luogo in un nome di file sicuro (slug)."""
    slug = re.sub(r"[^a-z0-9]+", "_", luogo.lower()).strip("_")
    return os.path.join(CARTELLA_MAPPE, f"{slug}.graphml")


def carica_grafo():
    """
    Carica il grafo dalla cartella 'mappe/' se gia' scaricato, altrimenti lo
    scarica da OSM e lo salva. Ripete finche' il luogo non viene trovato.
    """
    os.makedirs(CARTELLA_MAPPE, exist_ok=True)
    while True:
        luogo = input("Città o area da caricare (es. 'Torre del Greco, Italy'): ").strip()
        if not luogo:
            print("Il campo non può essere vuoto.")
            continue

        percorso = _nome_file(luogo)
        if os.path.exists(percorso):
            print(f"Carico la mappa da '{percorso}' (nessun download necessario)...")
            graph = ox.load_graphml(percorso)
        else:
            print(f"Download della mappa in corso ({luogo})...")
            try:
                graph = ox.graph_from_place(luogo, network_type="drive")
            except Exception:
                print(f"  Luogo non trovato: '{luogo}'. Prova ad essere più preciso "
                      f"(es. aggiungi la nazione: 'Napoli, Italy').")
                continue
            ox.save_graphml(graph, percorso)
            print(f"Mappa salvata in locale come '{percorso}'.")

        # Componente FORTEMENTE connessa piu' grande: il grafo grezzo di OSM ha
        # frammenti isolati e nodi senza uscita, che renderebbero irraggiungibili
        # alcune coppie (start, goal) e farebbero fallire il benchmark.
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
            y, x = ox.geocode(query)     # y = latitudine, x = longitudine
        except Exception:
            print(f"  Non trovato: '{query}'. Prova ad essere più preciso.")
            continue
        print(f"  Trovato: lat={y:.5f}, lon={x:.5f}")
        return ox.nearest_nodes(graph, X=x, Y=y)


# REGISTRO DEGLI ALGORITMI
def costruisci_algoritmi(G):
    """
    Costruisce le euristiche (una volta sola: il preprocessing ALT e' costoso)
    e restituisce un dizionario nome -> funzione(start, goal, on_expand).

    Le lambda uniformano le firme diverse (a_star e greedy vogliono l'euristica,
    bfs no), cosi' il ciclo di esecuzione resta uno solo.
    """
    print(f"\nPreprocessing euristica ALT ({K_LANDMARK} landmark, "
          f"{2 * K_LANDMARK} Dijkstra completi)...")
    h_zero = make_h_zero()
    h_eucl = make_h_euclidean(G)
    h_alt = make_h_landmark(G, k=K_LANDMARK, n_attivi=N_ATTIVI, seed=SEED)
    h_pes = make_h_weighted(h_eucl, W)
    print("Preprocessing completato.")

    algoritmi = {
        "A* Zero (Dijkstra)": lambda s, g, oe=None: a_star(G, h_zero, s, g, on_expand=oe),
        "A* Euclidea":        lambda s, g, oe=None: a_star(G, h_eucl, s, g, on_expand=oe),
        "A* Landmark (ALT)":  lambda s, g, oe=None: a_star(G, h_alt, s, g, on_expand=oe),
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
        espansi = [] if raccogli_espansi else None
        cb = espansi.append if raccogli_espansi else None

        path, cost, m = fn(start, goal, cb)
        if path is None:
            return None, None          # coppia irraggiungibile: la scarto

        if costo_ottimo is None:       # il primo e' Dijkstra
            costo_ottimo = cost
            percorso_ottimo = path

        risultati.append({
            "algoritmo": nome,
            "costo_m": cost,
            "errore_%": 100 * (cost - costo_ottimo) / costo_ottimo,
            "archi_path": m["path_len"] - 1,
            "nodi_espansi": m["expanded_nodes"],
            "tempo_ms": m["time_s"] * 1000,
            "picco_frontiera": m["peak_frontier"],
            "penetranza": m["expanded_nodes"] / m["path_len"],
            "b_star": effective_branching_factor(m["expanded_nodes"], m["path_len"] - 1),
            "path": path,
            "espansi": espansi,
        })

    return risultati, percorso_ottimo


# MODALITA' 1 — DEMO SU UNA SINGOLA COPPIA
def modalita_demo(G, algoritmi, h_alt):
    """Una coppia scelta dall'utente: tabella + video + immagini + grafici a barre."""
    start = chiedi_nodo(G, "Punto di partenza")
    goal = chiedi_nodo(G, "Punto di arrivo")

    risultati, percorso_ottimo = esegui_tutti(algoritmi, h_alt, start, goal,
                                              raccogli_espansi=True)
    if risultati is None:
        print("Nessun percorso tra i due punti.")
        return

    # --- tabella a terminale ---
    print(f"\n{'algoritmo':22}{'costo (m)':>11}{'err %':>8}{'archi':>7}"
          f"{'espansi':>10}{'ms':>9}{'picco':>8}{'penetr.':>9}")
    print("-" * 84)
    for r in risultati:
        print(f"{r['algoritmo']:22}{r['costo_m']:>11.0f}{r['errore_%']:>+8.2f}"
              f"{r['archi_path']:>7}{r['nodi_espansi']:>10}{r['tempo_ms']:>9.1f}"
              f"{r['picco_frontiera']:>8}{r['penetranza']:>9.2f}")

    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)

    # --- grafici a barre: confronto diretto sulle metriche ---
    grafici_demo(risultati)

    # --- immagini della mappa ---
    mappa_file = os.path.join(CARTELLA_RISULTATI, "mappa.png")
    percorso_file = os.path.join(CARTELLA_RISULTATI, "percorso.png")
    print(f"\nGenerazione di '{mappa_file}'...")
    ox.plot_graph(G, show=False, save=True, filepath=mappa_file,
                  node_size=0, edge_linewidth=0.3)
    print(f"Generazione di '{percorso_file}'...")
    ox.plot_graph_route(G, percorso_ottimo, show=False, save=True,
                        filepath=percorso_file, node_size=0, edge_linewidth=0.3,
                        route_linewidth=1.5, orig_dest_size=20)

    # --- video dell'esplorazione, uno per algoritmo ---
    for r in risultati:
        slug = re.sub(r"[^a-z0-9]+", "_", r["algoritmo"].lower()).strip("_")
        video_file = os.path.join(CARTELLA_RISULTATI, f"esplorazione_{slug}.mp4")
        print(f"Generazione video '{video_file}' ({len(r['espansi'])} frame)...")
        anima_esplorazione(G, r["espansi"], start, goal, r["path"], video_file)

    print(f"\nTutto salvato in '{CARTELLA_RISULTATI}/'.")


def grafici_demo(risultati):
    """
    Quattro barre affiancate: le metriche del confronto su una singola coppia.
    Scala logaritmica dove gli ordini di grandezza sono molto diversi (Dijkstra
    espande migliaia di nodi, ALT poche centinaia: in scala lineare le barre
    piccole sparirebbero).
    """
    nomi = [r["algoritmo"] for r in risultati]
    metriche = [
        ("nodi_espansi",    "Nodi espansi",              True),   # log
        ("tempo_ms",        "Tempo di esecuzione (ms)",  True),   # log
        ("costo_m",         "Costo del percorso (m)",    False),
        ("b_star",          "Effective branching factor b*", False),
        ("picco_frontiera", "Memoria di picco (frontiera)",  True),  # log
    ]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for ax, (chiave, titolo, log) in zip(axes.flat, metriche):
        valori = [r[chiave] for r in risultati]
        barre = ax.bar(range(len(nomi)), valori)
        # l'ottimo (errore 0) va evidenziato: distingue a colpo d'occhio gli
        # algoritmi ottimi da quelli che sbagliano
        if chiave == "errore_%":
            for b, v in zip(barre, valori):
                b.set_color("tab:green" if abs(v) < 1e-6 else "tab:red")
        ax.set_title(titolo)
        ax.set_xticks(range(len(nomi)))
        ax.set_xticklabels(nomi, rotation=30, ha="right", fontsize=8)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Confronto su una singola coppia (start, goal)")
    fig.tight_layout()
    out = os.path.join(CARTELLA_RISULTATI, "confronto_singolo.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Grafici salvati in '{out}'")


# MODALITA' 2 — BENCHMARK SU DISTANZE CRESCENTI
def distanza_aerea_km(G, u, v):
    """Distanza in linea d'aria tra due nodi, in km (per assegnare le coppie ai bin)."""
    from heuristics import _metri_per_grado_lat, _metri_per_grado_lon
    import math
    y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
    y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
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
    rng = random.Random(SEED)
    nodi = list(G.nodes)
    coppie = {b: [] for b in BIN_KM}

    tentativi, max_tentativi = 0, COPPIE_PER_BIN * len(BIN_KM) * 5000
    while tentativi < max_tentativi and any(len(v) < COPPIE_PER_BIN for v in coppie.values()):
        tentativi += 1
        s, g = rng.sample(nodi, 2)
        d = distanza_aerea_km(G, s, g)
        for b in BIN_KM:
            if b[0] <= d < b[1] and len(coppie[b]) < COPPIE_PER_BIN:
                coppie[b].append((s, g, d))
                break
    return coppie


def modalita_benchmark(G, algoritmi, h_alt):
    """Molte coppie su distanze crescenti: CSV + grafici a linee con mediana e quartili."""
    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)
    coppie = campiona_coppie(G)
    righe = []

    for b in BIN_KM:
        print(f"--- bin {b[0]}-{b[1]} km: {len(coppie[b])} coppie ---")
        for s, g, d_km in coppie[b]:
            risultati, _ = esegui_tutti(algoritmi, h_alt, s, g)
            if risultati is None:
                continue
            for r in risultati:
                righe.append({
                    "bin": f"{b[0]}-{b[1]}",
                    "bin_mid": (b[0] + b[1]) / 2,
                    "dist_aerea_km": round(d_km, 3),
                    "algoritmo": r["algoritmo"],
                    "costo_m": round(r["costo_m"], 1),
                    "errore_%": round(r["errore_%"], 2),
                    "archi_path": r["archi_path"],
                    "nodi_espansi": r["nodi_espansi"],
                    "tempo_ms": round(r["tempo_ms"], 3),
                    "picco_frontiera": r["picco_frontiera"],
                    "penetranza": round(r["penetranza"], 2),
                    "b_star": round(r["b_star"], 4),
                })

    if not righe:
        print("Nessuna coppia valida campionata.")
        return

    csv_path = os.path.join(CARTELLA_RISULTATI, "risultati.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"\nDati grezzi salvati in '{csv_path}' ({len(righe)} righe)")

    grafici_benchmark(righe)


def grafici_benchmark(righe):
    """
    Sei pannelli, uno per metrica, con la distanza aerea sull'asse x.

    Si riportano MEDIANA e BANDA INTERQUARTILE, non la singola run: su rete
    stradale la varianza tra coppie diverse e' enorme (dipende da dove capitano
    start e goal rispetto alle arterie principali) e un grafico a singola run
    sarebbe rumore puro.
    """
    algos = list(dict.fromkeys(r["algoritmo"] for r in righe))
    bins = sorted({r["bin_mid"] for r in righe})

    metriche = [
        ("nodi_espansi",    "Nodi espansi",                 True),
        ("tempo_ms",        "Tempo (ms)",                   True),
        ("picco_frontiera", "Picco frontiera (memoria)",    True),
        ("costo_m",         "Costo del percorso (m)",       False),
        ("errore_%",        "Errore vs ottimo (%)",         False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for ax, (chiave, titolo, log) in zip(axes.flat, metriche):
        for a in algos:
            xs, med, q1s, q3s = [], [], [], []
            for b in bins:
                vals = sorted(r[chiave] for r in righe
                              if r["algoritmo"] == a and r["bin_mid"] == b)
                if not vals:
                    continue
                xs.append(b)
                med.append(statistics.median(vals))
                q1s.append(vals[len(vals) // 4])
                q3s.append(vals[(3 * len(vals)) // 4])
            ax.plot(xs, med, marker="o", label=a)
            ax.fill_between(xs, q1s, q3s, alpha=0.12)   # banda interquartile
        ax.set_title(titolo)
        ax.set_xlabel("Distanza in linea d'aria (km)")
        if log:
            # gli ordini di grandezza tra Dijkstra e ALT sono tali che in scala
            # lineare le curve degli algoritmi informati sarebbero schiacciate a zero
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    axes.flat[5].axis("off")

    fig.suptitle("Confronto su distanze crescenti (mediana e banda interquartile)")
    fig.tight_layout()
    out = os.path.join(CARTELLA_RISULTATI, "confronto_distanze.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Grafici salvati in '{out}'")


# ENTRY POINT
if __name__ == "__main__":
    G, luogo = carica_grafo()
    print(f"Mappa pronta: {len(G.nodes)} nodi, {len(G.edges)} archi.")

    algoritmi, h_alt = costruisci_algoritmi(G)

    print("\nModalità:")
    print("  1) Demo   — una coppia scelta da te: tabella, grafici, immagini, video")
    print("  2) Benchmark — coppie casuali su distanze crescenti: CSV e grafici")
    scelta = input("Scelta [1/2]: ").strip()

    if scelta == "2":
        modalita_benchmark(G, algoritmi, h_alt)
    else:
        modalita_demo(G, algoritmi, h_alt)