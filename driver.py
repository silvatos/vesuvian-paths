import os
import re
import csv
import math
import random
import statistics

import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algoritmi import a_star, effective_branching_factor, greedy, bfs
from heuristics import (make_h_zero, make_h_euclidean, make_h_landmark,
                        make_h_weighted, _metri_per_grado_lat, _metri_per_grado_lon)


def sottografo_bbox(graph, bbox):
    """Ritaglia il grafo alla bounding box (left, bottom, right, top)."""
    left, bottom, right, top = bbox
    nodi = [n for n, d in graph.nodes(data=True)
            if left <= d["x"] <= right and bottom <= d["y"] <= top]
    return graph.subgraph(nodi)


# ============================================================
# CONFIGURAZIONE
# ============================================================
LUOGO = "Città Metropolitana di Napoli, Italy"

CARTELLA_MAPPE = "mappe"
CARTELLA_RISULTATI = "risultati"

# --- euristiche ---
K_LANDMARK = 8      # numero di landmark precalcolati per ALT
N_ATTIVI = 3        # landmark usati per ogni coppia (start, goal)
W = 2.0             # peso dell'euristica pesata

# --- benchmark ---
BIN_KM = [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 25), (25, 35), (35, 50)]
COPPIE_PER_BIN = 20   # numero di coppie campionate per ogni bin di distanza

# --- immagini dei percorsi ---
MARGINE_ZOOM = 0.15   # margine attorno al percorso, per lato, nelle foto di esempio

SEED = 42


# ============================================================
# CARICAMENTO MAPPA
# ============================================================
def carica_grafo():
    """
    Carica il grafo stradale dalla cartella 'mappe/' se gia' scaricato,
    altrimenti lo scarica da OpenStreetMap e lo salva in locale. Restituisce
    la componente fortemente connessa piu' grande.
    """
    os.makedirs(CARTELLA_MAPPE, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "_", LUOGO.lower()).strip("_")
    percorso = os.path.join(CARTELLA_MAPPE, f"{slug}.graphml")

    if os.path.exists(percorso):
        print(f"Carico la mappa da '{percorso}' (nessun download necessario)...")
        graph = ox.load_graphml(percorso)
    else:
        print(f"Download della mappa in corso ({LUOGO})...")
        print("  Su un'area provinciale può richiedere diversi minuti.")
        graph = ox.graph_from_place(LUOGO, network_type="drive")
        ox.save_graphml(graph, percorso)
        print(f"Mappa salvata in locale come '{percorso}'.")

    graph = ox.truncate.largest_component(graph, strongly=True)
    return graph


# ============================================================
# REGISTRO DEGLI ALGORITMI
# ============================================================
def costruisci_algoritmi(G):
    """
    Costruisce le euristiche e restituisce un dizionario
    nome -> funzione(start, goal, on_expand), oltre all'euristica ALT (che va
    ripreprocessata a ogni coppia).
    """
    print(f"\nPreprocessing euristica ALT ({K_LANDMARK} landmark, "
          f"{2 * K_LANDMARK} Dijkstra completi)...")
    print("  Su un grafo provinciale può richiedere diversi minuti.")

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
    Esegue tutti gli algoritmi sulla stessa coppia (start, goal) e restituisce
    la lista dei risultati, oppure None se la coppia non e' raggiungibile.

    raccogli_espansi: se True, registra anche l'ordine di espansione di ogni
    algoritmo (usato per generare le immagini di esempio).
    """
    h_alt.preprocess(start, goal)

    risultati = []
    costo_ottimo = None

    for nome, fn in algoritmi.items():
        espansi = [] if raccogli_espansi else None
        cb = espansi.append if raccogli_espansi else None

        path, cost, m = fn(start, goal, cb)

        if path is None:
            return None

        if costo_ottimo is None:
            costo_ottimo = cost   # il primo algoritmo eseguito e' Dijkstra: ottimo

        risultati.append({
            "algoritmo": nome,
            "costo_m": cost,
            "errore_%": 100 * (cost - costo_ottimo) / costo_ottimo,
            "archi_path": m["path_len"] - 1,
            "nodi_espansi": m["expanded_nodes"],
            "tempo_ms": m["time_s"] * 1000,
            "picco_frontiera": m["peak_frontier"],
            "b_star": effective_branching_factor(m["expanded_nodes"], m["path_len"] - 1),
            "path": path,
            "espansi": espansi,
        })

    return risultati


# ============================================================
# CAMPIONAMENTO DELLE COPPIE
# ============================================================
def distanza_aerea_km(G, u, v):
    """Distanza in linea d'aria tra due nodi, in km."""
    y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
    y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
    lm = (y1 + y2) / 2
    dy = (y1 - y2) * _metri_per_grado_lat(lm)
    dx = (x1 - x2) * _metri_per_grado_lon(lm)
    return math.hypot(dx, dy) / 1000.0


def campiona_coppie(G):
    """
    Campiona coppie casuali di nodi e le assegna ai bin di distanza in linea
    d'aria definiti in BIN_KM, fino a COPPIE_PER_BIN coppie per bin.
    """
    rng = random.Random(SEED)
    nodi = list(G.nodes)
    coppie = {b: [] for b in BIN_KM}

    tentativi, max_tentativi = 0, COPPIE_PER_BIN * len(BIN_KM) * 8000

    while tentativi < max_tentativi and any(len(v) < COPPIE_PER_BIN for v in coppie.values()):
        tentativi += 1
        s, g = rng.sample(nodi, 2)
        d = distanza_aerea_km(G, s, g)
        for b in BIN_KM:
            if b[0] <= d < b[1] and len(coppie[b]) < COPPIE_PER_BIN:
                coppie[b].append((s, g, d))
                break

    return coppie


# ============================================================
# BENCHMARK
# ============================================================
def benchmark(G, algoritmi, h_alt):
    """
    Esegue tutti gli algoritmi su tutte le coppie campionate, salva il CSV con
    i dati grezzi, genera i grafici di confronto e le immagini dei percorsi
    di esempio per ciascun bin.
    """
    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)

    print("\nCampionamento delle coppie...")
    coppie = campiona_coppie(G)

    righe = []   # una riga per ogni coppia (coppia, algoritmo), per il CSV

    for b in BIN_KM:
        n = len(coppie[b])
        print(f"--- bin {b[0]}-{b[1]} km: {n} coppie ---")
        if n == 0:
            continue

        for s, g, d_km in coppie[b]:
            risultati = esegui_tutti(algoritmi, h_alt, s, g)
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
                    "b_star": round(r["b_star"], 6),
                })

    if not righe:
        print("Nessuna coppia valida campionata.")
        return

    # --- CSV con i dati grezzi ---
    csv_path = os.path.join(CARTELLA_RISULTATI, "risultati.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"\nDati grezzi salvati in '{csv_path}' ({len(righe)} righe)")

    grafici_benchmark(righe)

    genera_immagini_bin(G, algoritmi, coppie)


# ============================================================
# IMMAGINI DI ESEMPIO (una per bin)
# ============================================================
def genera_immagini_bin(G, algoritmi, coppie):
    """
    Per ogni bin di distanza, disegna il percorso ottimo di una coppia
    rappresentativa (la prima campionata) e lo salva come immagine PNG.
    """
    dijkstra = algoritmi["A* Zero (Dijkstra)"]

    for b in BIN_KM:
        if not coppie[b]:
            continue

        start, goal, d_km = coppie[b][0]
        path, _, _ = dijkstra(start, goal)
        if path is None:
            continue

        # Bounding box del percorso, con margine.
        lats = [G.nodes[n]["y"] for n in path]
        lons = [G.nodes[n]["x"] for n in path]
        dlat = (max(lats) - min(lats)) * MARGINE_ZOOM or 0.01
        dlon = (max(lons) - min(lons)) * MARGINE_ZOOM or 0.01
        bbox = (min(lons) - dlon, min(lats) - dlat,
                max(lons) + dlon, max(lats) + dlat)

        sfondo = sottografo_bbox(G, bbox)
        grafo_percorso = sfondo if all(n in sfondo.nodes for n in path) else G

        nome_file = f"percorso_bin_{b[0]}_{b[1]}km.png"
        percorso_file = os.path.join(CARTELLA_RISULTATI, nome_file)
        print(f"Generazione di '{percorso_file}' (coppia da {d_km:.1f} km)...")
        ox.plot_graph_route(grafo_percorso, path, bbox=bbox, show=False, save=True,
                            filepath=percorso_file, node_size=0, edge_linewidth=0.4,
                            route_linewidth=2, orig_dest_size=40)

    print(f"\nImmagini dei percorsi salvate in '{CARTELLA_RISULTATI}/'.")


# ============================================================
# GRAFICI
# ============================================================
def grafici_benchmark(righe):
    """
    Genera un pannello di grafici (uno per metrica) con mediana e banda
    interquartile per ciascun algoritmo, in funzione del bin di distanza.
    """
    algos = list(dict.fromkeys(r["algoritmo"] for r in righe))
    bins = sorted({r["bin_mid"] for r in righe})

    # Posizioni categoriche sull'asse x (0, 1, 2, ...), una per bin.
    pos = {b: i for i, b in enumerate(bins)}

    # Etichette dei bin: intervallo, numero di coppie e distanza mediana effettiva.
    primo = algos[0]
    etichette = {}
    for b in bins:
        dist = sorted(r["dist_aerea_km"] for r in righe
                      if r["bin_mid"] == b and r["algoritmo"] == primo)
        label_bin = next(r["bin"] for r in righe if r["bin_mid"] == b)
        etichette[b] = (f"{label_bin} km\n"
                        f"n = {len(dist)}\n"
                        f"med = {statistics.median(dist):.1f} km")

    metriche = [
        ("nodi_espansi",    "Nodi espansi",                   True),
        ("tempo_ms",        "Tempo di esecuzione (ms)",       True),
        ("costo_m",         "Costo del percorso (m)",         False),
        ("b_star",          "Effective branching factor b*",  False),
        ("picco_frontiera", "Memoria di picco (frontiera)",   True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    for ax, (chiave, titolo, log) in zip(axes.flat, metriche):
        for a in algos:
            xs, med, q1s, q3s = [], [], [], []

            for b in bins:
                vals = sorted(r[chiave] for r in righe
                              if r["algoritmo"] == a and r["bin_mid"] == b)
                if not vals:
                    continue
                xs.append(pos[b])
                med.append(statistics.median(vals))
                q1s.append(vals[len(vals) // 4])
                q3s.append(vals[(3 * len(vals)) // 4])

            ax.plot(xs, med, marker="o", label=a)
            ax.fill_between(xs, q1s, q3s, alpha=0.12)

        ax.set_title(titolo)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([etichette[b] for b in bins], fontsize=7)
        ax.set_xlabel("Distanza in linea d'aria")

        if chiave == "b_star":
            # Zoom sul range effettivo di b* (vicino a 1.0 per tutti gli algoritmi).
            tutti = [r[chiave] for r in righe]
            lo, hi = min(tutti), max(tutti)
            margine = (hi - lo) * 0.1 or 0.001
            ax.set_ylim(max(1.0, lo - margine), hi + margine)

        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    axes.flat[0].legend(fontsize=8)
    axes.flat[5].axis("off")

    fig.suptitle(f"Confronto su distanze crescenti — {LUOGO}\n"
                 f"(mediana e banda interquartile)")
    fig.tight_layout()
    out = os.path.join(CARTELLA_RISULTATI, "confronto_distanze.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Grafici salvati in '{out}'")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    G = carica_grafo()
    print(f"Mappa pronta: {len(G.nodes)} nodi, {len(G.edges)} archi.")

    algoritmi, h_alt = costruisci_algoritmi(G)

    benchmark(G, algoritmi, h_alt)
