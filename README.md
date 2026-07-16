# Algoritmi di ricerca su mappe reali

Progetto per il corso di **Elementi di Intelligenza Artificiale**.

Confronto sperimentale di diversi algoritmi di ricerca del cammino minimo su una
rete stradale reale scaricata da [OpenStreetMap](https://www.openstreetmap.org/)
tramite [OSMnx](https://osmnx.readthedocs.io/). Il programma esegue un
**benchmark** automatico: campiona molte coppie di punti a distanze crescenti,
lancia tutti gli algoritmi su ciascuna coppia e produce dati, grafici di
confronto e immagini di percorsi di esempio.

## Algoritmi ed euristiche a confronto

Sei configurazioni di ricerca (in `driver.py`):

| Algoritmo | Priorità | Note |
|-----------|----------|------|
| A\* Zero (Dijkstra) | `f = g` | euristica nulla: A\* degenera in Dijkstra, ottimo — usato come riferimento |
| A\* Euclidea | `f = g + h` | h = distanza in linea d'aria (ammissibile), ottimo |
| A\* Landmark (ALT) | `f = g + h` | h da landmark + disuguaglianza triangolare (ammissibile), ottimo |
| A\* Pesata (w=2) | `f = g + w·h` | euristica euclidea pesata (`w=2`): più veloce ma **non** ottima |
| Greedy Best-First | `f = h` | ignora il costo già pagato: veloce ma **non** ottimo |
| BFS | coda FIFO | minimizza il **numero di archi**, non il costo in metri |

Le quattro euristiche (in `heuristics.py`): nulla, distanza in linea d'aria
(proiezione equirettangolare locale), euclidea pesata, e Landmark/ALT
(Goldberg & Harrelson) con selezione dei landmark tramite strategia *farthest*.

## Metriche calcolate

Per ogni coppia e ogni algoritmo: costo del percorso (m), errore % rispetto
all'ottimo (Dijkstra), numero di archi del percorso, nodi espansi, tempo di
esecuzione (ms), picco della frontiera (memoria) e fattore di ramificazione
effettivo *b\**.

## Struttura del progetto

| File | Descrizione |
|------|-------------|
| `algoritmi.py` | A\*, Greedy Best-First, BFS e calcolo del fattore di ramificazione effettivo *b\**. |
| `heuristics.py` | Le quattro euristiche (nulla, euclidea, euclidea pesata, Landmark/ALT). |
| `driver.py` | Programma principale: caricamento mappa, campionamento coppie, benchmark, output. |
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

Il programma esegue l'intero benchmark in modo automatico, senza input interattivo:

1. **carica la mappa** del luogo configurato (di default la Città Metropolitana
   di Napoli): la scarica da OpenStreetMap al primo avvio e la salva in `mappe/`,
   riusandola dai file locali alle esecuzioni successive;
2. **campiona** coppie casuali di nodi, suddivise in bin di distanza in linea d'aria;
3. **esegue** tutti gli algoritmi su ogni coppia e salva i risultati.

Al termine, in `risultati/` trovi:

- `risultati.csv` — i dati grezzi di tutte le coppie e di tutti gli algoritmi;
- `confronto_distanze.png` — un pannello di grafici (mediana e banda
  interquartile per metrica) che confronta gli algoritmi al crescere della distanza;
- `percorso_bin_<min>_<max>km.png` — il percorso ottimo di una coppia di esempio
  per ciascun bin di distanza.

### Configurazione

I parametri si modificano dalle costanti in cima a `driver.py`, tra cui:

- `LUOGO` — la città/area da analizzare (qualsiasi luogo riconosciuto da OpenStreetMap);
- `BIN_KM` e `COPPIE_PER_BIN` — i bin di distanza e quante coppie campionare per bin;
- `K_LANDMARK`, `N_ATTIVI` — landmark precalcolati e attivi per l'euristica ALT;
- `W` — il peso dell'euristica pesata;
- `SEED` — seme casuale, per risultati riproducibili.

## Note

- I dati stradali provengono da OpenStreetMap (© contributori OpenStreetMap, licenza ODbL).
- Su aree molto estese (es. intere città metropolitane) il primo download e il
  preprocessing dell'euristica ALT possono richiedere diversi minuti.
