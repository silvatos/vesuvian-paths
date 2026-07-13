# A* su mappe reali — Ricerca di percorsi stradali

Progetto per il corso di **Elementi di Intelligenza Artificiale**.

Implementazione dell'algoritmo **A\*** per il calcolo del percorso minimo su reti
stradali reali scaricate da [OpenStreetMap](https://www.openstreetmap.org/)
tramite [OSMnx](https://osmnx.readthedocs.io/). Il programma confronta tre
euristiche ammissibili e ne visualizza le differenze con metriche e immagini
dei percorsi trovati.

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
- Generazione di un'**immagine del percorso ottimo** per ciascun bin di distanza,
  utile come esempio concreto in documentazione.

## Struttura del progetto

| File | Descrizione |
|------|-------------|
| `algoritmi.py` | Algoritmo A\*, Greedy e BFS e calcolo del fattore di ramificazione effettivo *b\**. |
| `heuristics.py` | Le quattro euristiche (zero, euclidea, euclidea pesata w=2,landmark ALT). |
| `driver.py` | Programma principale: input utente, confronto euristiche, output, immagini. |
| `requirements.txt` | Dipendenze Python. |

Le cartelle `mappe/` (grafi scaricati) e `risultati/` (CSV, grafici e immagini)
vengono create automaticamente all'esecuzione e non sono incluse nel repository.

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

## Utilizzo

```bash
python driver.py
```

Il programma chiede in sequenza:

1. la **città/area** su cui lavorare (es. `Napoli, Italy`) — scaricata da OSM e
   salvata in `mappe/` (i lanci successivi la ricaricano dal file locale);
2. il **punto di partenza** (es. `Piazza del Gesù, Napoli`);
3. il **punto di arrivo** (es. `Stazione Centrale, Napoli`).

Al termine stampa le metriche di confronto e salva in `risultati/`:

- `risultati.csv` — i dati grezzi di tutte le coppie e gli algoritmi;
- `confronto_distanze.png` — i grafici di confronto tra euristiche;
- `percorso_bin_<min>_<max>km.png` — il percorso ottimo di una coppia di esempio
  per ciascun bin di distanza (vedi `BIN_KM` in `driver.py`).

## Note

- I dati stradali provengono da OpenStreetMap (© contributori OpenStreetMap, licenza ODbL).
- Su aree molto estese (es. intere città metropolitane) il primo download può
  richiedere alcuni minuti.
