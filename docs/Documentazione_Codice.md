# Documentazione tecnica del codice

Spiegazione del funzionamento di ogni funzione dei tre moduli del progetto:
[`algoritmi.py`](../algoritmi.py), [`heuristics.py`](../heuristics.py) e
[`driver.py`](../driver.py). Per ogni funzione si descrive *cosa fa* e *come*, a
livello di logica, senza entrare nel dettaglio di ogni singola istruzione.

---

## 1. `algoritmi.py` — algoritmi di ricerca

Contiene i tre algoritmi (A\*, Greedy, BFS), alcune utility condivise e il
calcolo del fattore di ramificazione effettivo.

### Utility condivise

**`_costo_arco(graph, u, v)`**
Restituisce il costo dell'arco `u → v` in metri. Il grafo è un `MultiDiGraph`,
quindi tra due nodi possono esistere più archi paralleli: la funzione prende la
lunghezza (`length`) minima tra tutti quelli disponibili.

**`_ricostruisci(came_from, start, goal)`**
Ricostruisce il cammino trovato. Il dizionario `came_from` associa a ogni nodo
il suo predecessore lungo il miglior cammino noto; partendo dal `goal` si risale
di predecessore in predecessore fino allo `start`, poi la lista viene invertita
(era stata costruita al contrario).

**`_costo_percorso(graph, path)`**
Somma le lunghezze degli archi che compongono il percorso, scorrendo le coppie
di nodi consecutivi. Fornisce il costo reale in metri di un cammino.

**`_metriche(expanded, peak, t0, path, costo)`**
Impacchetta in un dizionario le metriche di una ricerca: nodi espansi, tempo
trascorso (da `t0`), costo in metri, lunghezza del percorso e picco della
frontiera. Serve a uniformare il valore di ritorno di tutti gli algoritmi.

### Nucleo comune: best-first search

**`_best_first(graph, heuristic, start, goal, usa_g, on_expand=None)`**
È il cuore condiviso da A\* e Greedy. Usa una **coda di priorità** (`heapq`) che
ordina i nodi per valore `f`. Il flag `usa_g` è l'unica differenza tra i due
algoritmi:
- `usa_g=True` → `f = g + h` (A\*);
- `usa_g=False` → `f = h` (Greedy).

Meccanismi principali:
- **`closed_set`**: insieme dei nodi già espansi definitivamente; è una *graph
  search*, i nodi chiusi non vengono mai riaperti.
- **Lazy deletion**: `heapq` non consente di aggiornare la priorità di un
  elemento già inserito. Quando si scopre un cammino migliore verso un nodo, si
  inserisce una nuova copia con `f` più basso e si lascia la vecchia nell'heap;
  quando quest'ultima viene estratta, il nodo è già in `closed_set` e viene
  scartata senza essere ricontata.
- **Goal test all'espansione**: il goal è riconosciuto quando viene *estratto*
  dalla coda (non quando viene generato). Con euristica ammissibile e
  consistente questo garantisce che il cammino trovato sia ottimo.
- **Rilassamento**: un vicino viene aggiornato solo se lo si raggiunge per la
  prima volta o con un costo `g` strettamente minore di quello noto.

Il parametro `on_expand`, se fornito, è una callback chiamata su ogni nodo
nell'ordine di espansione (utile per tracciare/visualizzare la ricerca).
Ritorna `(path, costo, metriche)` oppure `(None, None, None)` se il goal è
irraggiungibile.

**`a_star(graph, heuristic, start, goal, on_expand=None)`**
Semplice wrapper: chiama `_best_first` con `usa_g=True`. Ricerca ottima con
`f = g + h`.

**`greedy(graph, heuristic, start, goal, on_expand=None)`**
Wrapper con `usa_g=False`: la priorità è solo `h`. Sceglie sempre il nodo che
l'euristica stima più vicino al goal, ignorando il costo già pagato: espande
pochi nodi ma **non** garantisce l'ottimalità.

### Breadth-First Search

**`bfs(graph, start, goal, on_expand=None)`**
Ricerca in ampiezza che minimizza il **numero di archi** del percorso, non il
costo in metri. Differenze rispetto al best-first:
- usa una **coda FIFO** (`deque`), non una coda di priorità: i nodi escono
  nell'ordine di inserimento, producendo l'esplorazione per livelli tipica della
  BFS;
- l'insieme `raggiunti` impedisce di accodare più volte lo stesso nodo;
- il **goal test avviene alla generazione** del vicino (non all'espansione):
  è corretto perché con costi unitari il primo cammino che tocca il goal ha già
  il numero minimo di archi.
Gestisce a parte il caso degenere `start == goal`.

### Metrica derivata

**`effective_branching_factor(n_expanded, depth, tol=1e-9)`**
Calcola il fattore di ramificazione effettivo *b\**, cioè il numero medio di
figli che un albero uniforme di profondità `depth` dovrebbe avere per contenere
`n_expanded` nodi. Risolve numericamente per **bisezione** l'equazione
`1 + b + b² + … + b^depth = n_expanded + 1`. La somma geometrica è monotona
crescente in `b`, quindi la bisezione converge; l'estremo superiore iniziale è
scelto per evitare overflow, e il caso `b → 1` è gestito col suo limite
(`depth + 1`) per evitare la forma indeterminata 0/0.

---

## 2. `heuristics.py` — euristiche

Costruisce le funzioni euristiche `h(node, goal)` passate agli algoritmi. Sono
tutte *factory*: restituiscono la funzione `h` con i dati necessari già
precalcolati (chiusura).

**`make_h_zero()`**
Restituisce l'euristica nulla `h = 0`. Con essa A\* perde ogni informazione
sulla direzione e degenera in **Dijkstra**: è la baseline di confronto.

**`_metri_per_grado_lat(lat_deg)`** e **`_metri_per_grado_lon(lat_deg)`**
Funzioni di supporto che convertono i gradi geografici in metri alla latitudine
data, usando le formule WGS84. La longitudine "vale" meno metri man mano che ci
si allontana dall'equatore, perciò dipende dalla latitudine.

**`make_h_euclidean(graph)`**
Costruisce l'euristica basata sulla **distanza in linea d'aria**. Precalcola e
mette in cache le coordinate di tutti i nodi, poi la `h` restituita converte la
differenza di latitudine/longitudine in metri (proiezione equirettangolare
locale attorno alla latitudine media dei due punti) e ne calcola la distanza
euclidea. È ammissibile: la linea d'aria non sovrastima mai la distanza reale su
strada.

**`make_h_weighted(h_base, w)`**
Restituisce un'euristica che moltiplica una euristica di base per un peso `w`:
`h = w · h_base`. Con `w > 1` si ottiene il **Weighted A\*** (`f = g + w·h`),
che espande meno nodi ma perde l'ottimalità (l'euristica non è più ammissibile).

**`scegli_landmark_farthest(graph, k, seed=42)`**
Seleziona `k` **landmark** (nodi di riferimento) con la strategia *farthest*
(greedy maxmin): parte da un nodo casuale, sceglie come primo landmark il più
lontano da esso, poi a ogni passo aggiunge il nodo che massimizza la distanza
minima dai landmark già scelti. Così i landmark risultano ben distribuiti "ai
bordi" del grafo, dove sono più informativi. Le distanze si calcolano con
Dijkstra.

**`make_h_landmark(graph, landmarks=None, k=8, n_attivi=3, seed=42)`**
Costruisce l'euristica **ALT** (A\*, Landmarks, Triangle inequality; Goldberg &
Harrelson, 2005). In fase di preprocessing calcola, con Dijkstra, le distanze
esatte **da** ogni landmark (sul grafo) e **verso** ogni landmark (sul grafo
invertito). Per ogni landmark `L`, la disuguaglianza triangolare fornisce un
lower bound ammissibile sul costo residuo `d(node, goal)`.

La funzione restituita espone due attributi:
- **`preprocess(start, goal)`**: da chiamare prima di ogni ricerca; seleziona i
  `n_attivi` landmark che danno il bound migliore per quella specifica coppia
  (tra i `k` disponibili), riducendo il costo di valutare `h`;
- **`landmarks`**: la lista completa dei landmark scelti.

La `h` finale restituisce, per un nodo, il massimo dei bound calcolati sui
landmark attivi (il massimo di più lower bound ammissibili è ancora un lower
bound ammissibile, e più stretto). In genere è più informata dell'euclidea.

---

## 3. `driver.py` — orchestrazione ed esperimenti

Non contiene logica di ricerca: coordina i moduli precedenti, esegue il
benchmark e produce gli output. I parametri sono le costanti in cima al file
(`LUOGO`, `BIN_KM`, `COPPIE_PER_BIN`, `K_LANDMARK`, `N_ATTIVI`, `W`, `SEED`).

**`sottografo_bbox(graph, bbox)`**
Restituisce il sottografo dei soli nodi che cadono dentro una *bounding box*
`(left, bottom, right, top)`. Usato per ritagliare l'area attorno a un percorso
quando si generano le immagini di esempio.

**`carica_grafo()`**
Carica il grafo stradale del luogo configurato. Se il file `.graphml` è già
presente in `mappe/` lo carica da lì; altrimenti lo scarica da OpenStreetMap con
OSMnx e lo salva per i riutilizzi futuri. Infine restituisce la **componente
fortemente connessa più grande**, così che ogni coppia di nodi sia
mutuamente raggiungibile (importante su un grafo diretto con sensi unici).

**`costruisci_algoritmi(G)`**
Istanzia le quattro euristiche (inclusi il preprocessing di ALT, il passo più
costoso) e costruisce un dizionario `nome → funzione(start, goal, on_expand)`
con le sei configurazioni di ricerca (A\* con le varie euristiche, Greedy, BFS).
Restituisce il dizionario e, separatamente, l'euristica ALT, che va
ri-preprocessata a ogni coppia.

**`esegui_tutti(algoritmi, h_alt, start, goal, raccogli_espansi=False)`**
Esegue tutti gli algoritmi sulla **stessa** coppia `(start, goal)`. Prima
richiama `h_alt.preprocess` per adattare i landmark alla coppia. Il primo
algoritmo eseguito è Dijkstra (ottimo), il cui costo diventa il riferimento per
calcolare l'**errore percentuale** degli altri. Restituisce una lista di
dizionari (uno per algoritmo) con tutte le metriche, oppure `None` se la coppia
non è raggiungibile. Con `raccogli_espansi=True` registra anche l'ordine di
espansione di ciascun algoritmo.

**`distanza_aerea_km(G, u, v)`**
Distanza in linea d'aria tra due nodi, in km, usando le stesse formule
metri-per-grado del modulo delle euristiche. Serve a classificare le coppie nei
bin di distanza.

**`campiona_coppie(G)`**
Campiona coppie casuali di nodi e le assegna ai **bin di distanza** definiti in
`BIN_KM` (in base alla distanza aerea), fino a raccoglierne `COPPIE_PER_BIN` per
bin. Un tetto massimo di tentativi evita cicli infiniti se qualche bin è
difficile da riempire. Il generatore casuale è inizializzato con `SEED` per la
riproducibilità.

**`benchmark(G, algoritmi, h_alt)`**
Funzione principale dell'esperimento. Campiona le coppie, esegue tutti gli
algoritmi su ognuna, raccoglie i risultati in una tabella e:
1. salva i **dati grezzi** in `risultati/risultati.csv` (una riga per coppia ×
   algoritmo);
2. genera i **grafici di confronto** (`grafici_benchmark`);
3. genera le **immagini dei percorsi** di esempio (`genera_immagini_bin`).

**`genera_immagini_bin(G, algoritmi, coppie)`**
Per ogni bin di distanza prende una coppia rappresentativa, ne calcola il
percorso ottimo con Dijkstra, ritaglia il grafo attorno al percorso (bounding
box con un margine) e salva un'immagine PNG con il tracciato evidenziato. Dà
un'idea visiva concreta di come sono fatti i percorsi a ciascuna scala di
distanza.

**`grafici_benchmark(righe)`**
Costruisce un pannello di grafici, uno per metrica (nodi espansi, tempo, costo,
*b\**, memoria di picco). Per ogni algoritmo e ogni bin traccia la **mediana** e
una banda che copre l'**intervallo interquartile**, in funzione della distanza.
Le metriche molto variabili usano scala logaritmica; il grafico di *b\** viene
"zoomato" sul suo range effettivo (valori prossimi a 1). Salva il tutto in
`risultati/confronto_distanze.png`.

**Entry point (`if __name__ == "__main__"`)**
Sequenza completa del programma: carica la mappa, costruisce algoritmi ed
euristiche, esegue il benchmark. È la guardia standard che fa partire il codice
solo quando `driver.py` è eseguito direttamente.
