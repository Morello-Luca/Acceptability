# Consensus Ranking con Bradley-Terry e MLE

Questo progetto implementa un sistema di **consensus ranking** utilizzando il modello di Bradley-Terry e la stima tramite **Maximum Likelihood Estimation (MLE)**. Il sistema combina due approcci:
- **Ottimizzazione MLE** per ottenere un insieme di pesi che rappresentano la preferenza relativa per ciascuna feature.
- **Metodi di consenso** che, a partire dai ranking osservati e dai pesi ottenuti, producono un ordinamento finale delle feature.

## Indice
- [Introduzione](#introduzione)
- [Background Teorico](#background-teorico)
  - [Il Modello di Bradley-Terry](#il-modello-di-bradley-terry)
  - [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
  - [Metodi di Consenso](#metodi-di-consenso)
- [Struttura del Progetto](#struttura-del-progetto)
- [Requisiti e Dipendenze](#requisiti-e-dipendenze)
- [Istruzioni per l'Installazione e l'Uso](#istruzioni-per-linstallazione-e-luso)
- [Dettagli Tecnici e Spiegazioni](#dettagli-tecnici-e-spiegazioni)
- [Licenza](#licenza)

## Introduzione

Il progetto è volto a risolvere problemi di *ranking* in cui si raccolgono preferenze espresse da diversi osservatori. Utilizzando il **modello di Bradley-Terry**, è possibile stimare dei pesi che rappresentano la probabilità relativa di preferenza tra le feature. Successivamente, si applicano due metodi di consenso per ottenere un ordinamento finale:
1. **Metodo 2:** Basato su una matrice di confronto a coppie ponderata.
2. **Metodo 3:** Un approccio regolarizzato che combina la distanza Kemeny-Young e la distanza Kendall Tau mediante una ricerca locale.

## Background Teorico

### Il Modello di Bradley-Terry

Il modello di Bradley-Terry è utilizzato per modellare le probabilità di preferenza tra coppie di elementi. Se abbiamo due feature *i* e *j* con pesi \( w_i \) e \( w_j \), la probabilità che *i* sia preferita a *j* è data da:
\[
P(i \succ j) = \frac{w_i}{w_i + w_j}
\]
Questo modello è ampiamente usato in situazioni in cui le preferenze sono esprimibili a coppie.

### Maximum Likelihood Estimation (MLE)

La **stima per massima verosimiglianza (MLE)** consiste nel trovare i pesi \( w \) che massimizzano la probabilità (o log-verosimiglianza) di osservare i ranking raccolti. Nel nostro caso, per ciascun ranking osservato si confrontano tutte le coppie di feature e si costruisce la log-verosimiglianza totale, che viene poi massimizzata (o, equivalentemente, il negativo viene minimizzato).

### Metodi di Consenso

Una volta ottenuti i pesi tramite MLE, si procede a ricavare un ordinamento finale (consenso) delle feature. Il progetto implementa due metodi:
- **Metodo 2:** Costruisce una matrice di confronti a coppie ponderata dai ranking osservati e ordina le feature in base alle "vittorie nette".
- **Metodo 3:** Utilizza una ricerca locale per minimizzare un costo composto dalla distanza Kemeny-Young pesata e dalla distanza Kendall Tau rispetto all'ordinamento MLE. Questo approccio regolarizzato cerca di mantenere il consenso quanto più simile all'ordine MLE, ma riducendo al contempo il numero totale di disaccordi con i ranking osservati.

## Struttura del Progetto

Il progetto è organizzato come segue:
- **main.py:** Contiene il codice completo con le funzioni per il caricamento dei dati, la stima MLE, la conversione dei ranking, il calcolo dei pesi e l'implementazione dei metodi di consenso.
- **list.csv:** Esempio di file CSV contenente i ranking.  
  > **Nota:** Il file CSV deve avere la prima colonna come identificativo e le colonne successive con i ranking numerici (valori minori indicano preferenze più alte).

## Requisiti e Dipendenze

Il progetto richiede:
- Python 3.x
- Librerie:
  - **numpy**
  - **pandas**
  - **scipy**

Puoi installare le dipendenze tramite `pip`:

```bash
pip install numpy pandas scipy
