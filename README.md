# A* su mappe reali — Ricerca di percorsi stradali

Progetto per il corso di **Elementi di Intelligenza Artificiale**.

Implementazione dell'algoritmo **A\*** per il calcolo del percorso minimo su reti
stradali reali scaricate da [OpenStreetMap](https://www.openstreetmap.org/)
tramite [OSMnx](https://osmnx.readthedocs.io/). Il programma confronta tre
euristiche ammissibili e ne visualizza le differenze con metriche, immagini e
video dell'esplorazione.

## Funzionalità

- Calcolo del percorso minimo con **A\*** su un grafo stradale reale.
- Confronto di tre euristiche:
  - **Zero** (A\* degenera in Dijkstra) — baseline;
  - **Euclidea** — distanza in linea d'aria;
  - **Landmark (ALT)** — basata sulla disuguaglianza triangolare da nodi di riferimento.
- **Metriche** per ciascuna euristica: costo del percorso, nodi espansi, tempo,
  picco della frontiera e fattore di ramificazione effettivo *b\**.
- Scelta interattiva da terminale di **città**, **punto di partenza** e **arrivo**
  (indirizzo, via o punto di interesse), con geocodifica automatica.
- Salvataggio locale delle mappe scaricate (nessun ri-download alle esecuzioni successive).
- Generazione di **immagini** (mappa e percorso) e **video MP4** che mostrano
  l'ordine di esplorazione dei nodi, uno per euristica.

## Struttura del progetto

| File | Descrizione |
|------|-------------|
| `a_star.py` | Algoritmo A\* e calcolo del fattore di ramificazione effettivo *b\**. |
| `heuristics.py` | Le tre euristiche (zero, euclidea, landmark). |
| `video.py` | Generazione dei video MP4 dell'esplorazione. |
| `main.py` | Programma principale: input utente, confronto euristiche, output. |
| `requirements.txt` | Dipendenze Python. |

Le cartelle `mappe/` (grafi scaricati) e `risultati/` (immagini e video) vengono
create automaticamente all'esecuzione e non sono incluse nel repository.

## Installazione

Richiede **Python 3.10+**.

```bash
# 1. Clona il repository
git clone <url-del-repository>
cd <cartella-del-progetto>

# 2. Crea e attiva un ambiente virtuale
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Installa le dipendenze
pip install -r requirements.txt
```

Il pacchetto `imageio-ffmpeg` include già il binario di **ffmpeg** necessario a
generare i video MP4: non serve installare ffmpeg separatamente.

## Utilizzo

```bash
python main.py
```

Il programma chiede in sequenza:

1. la **città/area** su cui lavorare (es. `Napoli, Italy`) — scaricata da OSM e
   salvata in `mappe/` (i lanci successivi la ricaricano dal file locale);
2. il **punto di partenza** (es. `Piazza del Gesù, Napoli`);
3. il **punto di arrivo** (es. `Stazione Centrale, Napoli`).

Al termine stampa le metriche di confronto e salva in `risultati/`:

- `mappa.png` — la rete stradale;
- `percorso.png` — la rete con il percorso ottimale evidenziato;
- `esplorazione_<euristica>.mp4` — un video per euristica con l'ordine di esplorazione.

## Note

- I dati stradali provengono da OpenStreetMap (© contributori OpenStreetMap, licenza ODbL).
- Su aree molto estese (es. intere città metropolitane) il primo download e la
  generazione dei video possono richiedere alcuni minuti.
