import math

import imageio_ffmpeg
import matplotlib.animation as animation
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import osmnx as ox

# collega matplotlib al binario ffmpeg fornito da imageio-ffmpeg (per salvare in mp4)
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

COLORE_SFONDO = "#111111"   # sfondo scuro di mappa e figura (nessun bordo bianco)
COLORE_ESPANSI = "#00d5ff"  # colore unico delle strade esplorate (azzurro acceso)
SPESSORE_STRADE = 0.3       # spessore degli archi del grafo (sfondo)
SPESSORE_ESPLORATI = 0.5    # spessore delle strade esplorate (leggermente maggiore)


def _figsize_mappa(graph, lato=10):
    """
    Calcola una dimensione di figura con le stesse proporzioni della mappa,
    così il grafo riempie tutto il fotogramma senza margini né distorsioni.
    """
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    lat_media = math.radians((max(ys) + min(ys)) / 2)
    dx = (max(xs) - min(xs)) * math.cos(lat_media)  # correzione longitudine
    dy = max(ys) - min(ys)
    aspect = dx / dy if dy else 1.0
    return (lato, lato / aspect) if aspect >= 1 else (lato * aspect, lato)


# ============================================================
# ANIMAZIONE DELL'ESPLORAZIONE DI A*
# ============================================================
def anima_esplorazione(graph, ordine_espansi, start, goal, path, filepath,
                       n_frame=120, fps=20, hold_finale=20, dpi=200, bitrate=5000):
    """
    Genera un video MP4 che mostra l'ordine in cui A* esplora la rete stradale.
    Le strade esplorate vengono disegnate come linee (leggermente più spesse
    delle strade di sfondo), non come punti.

    graph:          grafo osmnx/networkx
    ordine_espansi: lista dei nodi nell'ordine di espansione (dalla callback on_expand)
    start, goal:    nodi di partenza e arrivo
    path:           percorso finale trovato (lista di nodi)
    filepath:       dove salvare il video (.mp4)
    n_frame:        numero di fotogrammi per la fase di esplorazione
    fps:            fotogrammi al secondo del video
    hold_finale:    fotogrammi finali che mostrano il percorso completo
    dpi:            risoluzione (200 => circa 1600x1600 px, alta definizione)
    bitrate:        bitrate del video in kbps (qualità)
    """
    # Segmenti stradali esplorati, in ordine di scoperta: quando un nodo viene
    # espanso, gli archi verso i nodi già espansi diventano "strade esplorate".
    espansi = set()
    segmenti = []
    for u in ordine_espansi:
        espansi.add(u)
        xu, yu = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vicini = set(graph.successors(u)) | set(graph.predecessors(u))
        for v in vicini:
            if v in espansi:
                segmenti.append([(xu, yu), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
    M = len(segmenti)

    # base: rete stradale (sfondo scuro, archi sottili) disegnata una sola volta.
    # figsize con le proporzioni della mappa + assi a tutto fotogramma => niente bordi
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0,
                            edge_linewidth=SPESSORE_STRADE, bgcolor=COLORE_SFONDO,
                            figsize=_figsize_mappa(graph))
    fig.patch.set_facecolor(COLORE_SFONDO)          # anche lo sfondo esterno agli assi
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # gli assi riempiono la figura
    ax.margins(0)

    # strade esplorate: collezione di segmenti dello stesso colore, che cresce
    lc = mcoll.LineCollection([], colors=COLORE_ESPANSI,
                              linewidths=SPESSORE_ESPLORATI, zorder=2)
    ax.add_collection(lc)

    # percorso finale (inizialmente nascosto, appare alla fine)
    px = [graph.nodes[n]["x"] for n in path]
    py = [graph.nodes[n]["y"] for n in path]
    (linea_path,) = ax.plot(px, py, color="red", linewidth=1.5, zorder=3, visible=False)

    # marcatori di partenza (verde) e arrivo (rosso)
    ax.scatter([graph.nodes[start]["x"]], [graph.nodes[start]["y"]],
               s=45, c="lime", edgecolor="white", linewidth=0.5, zorder=4)
    ax.scatter([graph.nodes[goal]["x"]], [graph.nodes[goal]["y"]],
               s=45, c="red", edgecolor="white", linewidth=0.5, zorder=4)

    # quanti segmenti rivelare per fotogramma (arrotondamento per eccesso)
    step = max(1, -(-M // n_frame))
    frame_espansione = max(1, -(-M // step))

    def update(frame):
        k = min(M, (frame + 1) * step)
        lc.set_segments(segmenti[:k])
        if frame >= frame_espansione - 1:
            linea_path.set_visible(True)  # esplorazione completa: mostra il percorso
        return lc, linea_path

    anim = animation.FuncAnimation(
        fig, update, frames=frame_espansione + hold_finale,
        interval=1000 / fps, blit=False,
    )
    writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
    anim.save(filepath, writer=writer, dpi=dpi,
              savefig_kwargs={"facecolor": COLORE_SFONDO})
    plt.close(fig)
