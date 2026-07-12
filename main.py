import os
import random
import re

import osmnx as ox

from a_star import a_star, effective_branching_factor
from heuristics import make_h_zero, make_h_euclidean, make_h_landmark
from video import anima_esplorazione

CARTELLA_MAPPE = "mappe"
CARTELLA_RISULTATI = "risultati"  # immagini e video generati


def _nome_file(luogo):
    """Trasforma il nome di un luogo in un nome di file sicuro (slug)."""
    slug = re.sub(r"[^a-z0-9]+", "_", luogo.lower()).strip("_")
    return os.path.join(CARTELLA_MAPPE, f"{slug}.graphml")


def carica_grafo():
    """
    Chiede all'utente la città/luogo su cui lavorare, poi carica il grafo dalla
    cartella 'mappe/' se già scaricato in precedenza, altrimenti lo scarica da OSM
    e lo salva in locale. Ripete la richiesta finché il luogo non viene trovato.
    """
    os.makedirs(CARTELLA_MAPPE, exist_ok=True)
    while True:
        luogo = input("Città o area da caricare (es. 'Napoli, Italy'): ").strip()
        if not luogo:
            print("Il campo non può essere vuoto.")
            continue

        percorso = _nome_file(luogo)
        if os.path.exists(percorso):
            print(f"Carico la mappa da '{percorso}' (nessun download necessario)...")
            return ox.load_graphml(percorso)

        print(f"Download della mappa in corso ({luogo})...")
        try:
            graph = ox.graph_from_place(luogo, network_type="drive")
        except Exception:
            print(f"  Luogo non trovato: '{luogo}'. Prova ad essere più preciso "
                  f"(es. aggiungi la nazione: 'Napoli, Italy').")
            continue
        ox.save_graphml(graph, percorso)
        print(f"Mappa salvata in locale come '{percorso}' per le prossime esecuzioni.")
        return graph


def chiedi_nodo(graph, etichetta):
    """
    Chiede all'utente una via, un indirizzo o un punto di interesse (via
    terminale), lo geocodifica con Nominatim e restituisce il nodo del
    grafo più vicino. Ripete la richiesta finché la geocodifica non riesce.
    """
    while True:
        query = input(f"{etichetta} (via, indirizzo o punto di interesse): ").strip()
        if not query:
            print("Il campo non può essere vuoto.")
            continue
        try:
            y, x = ox.geocode(query)
        except Exception:
            print(f"  Non trovato: '{query}'. Prova ad essere più preciso "
                  f"(es. aggiungi via/numero civico e città).")
            continue
        print(f"  Trovato: lat={y:.5f}, lon={x:.5f}")
        return ox.nearest_nodes(graph, X=x, Y=y)


# ============================================================
# ESEMPIO D'USO: confronto euristiche su una mappa scelta dall'utente
# ============================================================
if __name__ == "__main__":
    G = carica_grafo()
    print(f"Mappa pronta: {len(G.nodes)} nodi, {len(G.edges)} archi.\n")

    # L'utente inserisce i due punti da terminale (indirizzo o POI)
    start = chiedi_nodo(G, "Punto di partenza")
    goal = chiedi_nodo(G, "Punto di arrivo")

    # Landmark: qui scelti a caso, meglio sceglierli "ai bordi" del grafo
    landmarks = random(list(G.nodes), 4)

    heuristics = {
        "Zero (Dijkstra)": make_h_zero(),
        "Euclidea":        make_h_euclidean(G),
        "Landmark (ALT)":  make_h_landmark(G, landmarks),
    }

    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)
    best_path = None
    for name, h in heuristics.items():
        ordine_espansi = []  # raccoglie i nodi nell'ordine in cui A* li espande
        path, cost, m = a_star(G, h, start, goal, on_expand=ordine_espansi.append)
        b_star = effective_branching_factor(m["expanded_nodes"], m["path_len"] - 1)
        print(f"\n--- {name} ---")
        print(f"Costo percorso:   {cost:.0f} m")
        print(f"Nodi espansi:     {m['expanded_nodes']}")
        print(f"Tempo:            {m['time_s']*1000:.2f} ms")
        print(f"Picco frontiera:  {m['peak_frontier']}")
        print(f"b* effettivo:     {b_star:.3f}")
        best_path = path  # tutte le euristiche sono ammissibili: costo (quasi) identico

        # video dell'esplorazione per questa euristica
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        video_file = os.path.join(CARTELLA_RISULTATI, f"esplorazione_{slug}.mp4")
        print(f"Generazione video '{video_file}' in corso...")
        anima_esplorazione(G, ordine_espansi, start, goal, path, video_file)

    # --- Immagini: mappa semplice e mappa col percorso ---
    mappa_file = os.path.join(CARTELLA_RISULTATI, "mappa.png")
    percorso_file = os.path.join(CARTELLA_RISULTATI, "percorso.png")
    print(f"\nGenerazione di '{mappa_file}' in corso...")
    ox.plot_graph(G, show=False, save=True, filepath=mappa_file, node_size=0, edge_linewidth=0.3)
    print(f"Generazione di '{percorso_file}' in corso...")
    ox.plot_graph_route(G, best_path, show=False, save=True, filepath=percorso_file, node_size=0,
                         edge_linewidth=0.3, route_linewidth=1.5, orig_dest_size=20)
    print(f"Immagini salvate in '{CARTELLA_RISULTATI}/'.")
