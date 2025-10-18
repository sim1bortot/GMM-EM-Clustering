'''
---------------------------------------------------------------- 
            MODELLO GMM (GAUSSIAN MIXTURE MODEL)
----------------------------------------------------------------

I dati vengono tutti dal file .csv ma secondo l'ipotesi di GMM non provengono tutti da un'unica fonte
ma si possono intendere come una mistura di diverse distribuzioni di probabilità a forma di campana (gaussiane).
Ogni campana corrisponde ad un cluster. 
Raggruppare i dati quindi corrisponde a trovare i parametri di queste campane:
        - media mu      : il centro di ogni campana
        - covarianza    : forma/orientamento di ogni campana
        - peso          : dimensione di ogni campana (cluster) rispetto alle altre
        
---------------------------------------------------------------- 
            ALGORITMO EM (EXPECTATION MAXIMIZATION)
----------------------------------------------------------------

Poichè dobbiamo trovare i parametri delle campane, utilizziamo questo algoritmo per trovarli, 
tramite un processo iterativo ipotesi-aggiornamento.

Inizia creando ipotesi casuali per i parametri delle m campane (cluster). In pratica disegna 
m ellissi casuali sopra i tuoi dati.

E-STEP : 
    in questa fase, l'algoritmo guarda ogni singolo punto del dataset e, basandosi sulle ellissi attuali, 
    calcola la probabilità che quel punto appartenga a ciascun cluster. 
    Questo non è un assegnamento netto, ma uno "soft" detto responsabilità.
M-STEP :
    Una volta che tutti i punti hanno le loro "responsabilità", l'algoritmo aggiorna i parametri di ogni ellisse 
    per adattarsi meglio ai punti di cui è "responsabile".
    es. 
    Il centro del Cluster 1 viene spostato verso la posizione media di tutti i punti che molto probabilmente gli appartengono
    La sua forma viene modificata per riflettere la dispersione di quegli stessi punti.
    Il suo peso aumenta se molti punti gli sono stati assegnati.
Convergenza:
    Questo ciclo E-M si ripete. Ad ogni ripetizione, le ellissi si adattano sempre meglio alla struttura reale dei dati. 
    L'algoritmo si ferma quando gli aggiornamenti diventano così piccoli da essere trascurabili. 
    A questo punto si dice che ha raggiunto la convergenza.

'''

from util import *
import pandas as pd
import sys

# -------------------------------------------------------------------
# Impostazioni
# -------------------------------------------------------------------

draw_plots = True 

nome_file_dati = 'temperature.csv'

np.random.seed(17)

# -------------------------------------------------------------------
# Caricamento dei Dati
# -------------------------------------------------------------------
try:
    df = pd.read_csv(nome_file_dati)
    x = df.values
    print(f"Dati caricati con successo dal file '{nome_file_dati}'. Trovati {x.shape[0]} campioni con {x.shape[1]} feature.")
except FileNotFoundError:
    print(f"ERRORE: File non trovato! Esegui prima lo script 'crea_dati_temperatura.py' per generare il file.")
    sys.exit()

# -------------------------------------------------------------------
# Impostazione del Numero di Cluster
# -------------------------------------------------------------------
m_clusters = 4
print(f"Il programma cercherà {m_clusters} cluster per rappresentare le stagioni.")

# -------------------------------------------------------------------
# Esecuzione dell'Algoritmo EM
# -------------------------------------------------------------------
gmm = GMM(x.shape[0], x.shape[1], m_clusters)
num_iters = 30
q = []
threshold = 1e-3

print("\n--- Inizio Esecuzione Algoritmo EM ---")
for e in range(num_iters):
    gmm.e_step(x)
    gmm.m_step(x)
    q.append(gmm.q(x))
    print(f"Iterazione: {e + 1}, q: {q[-1]:.4f}")
    if (len(q) > 2) and (abs(q[-1] - q[-2]) < threshold): # Usiamo abs per sicurezza
        print("\nConvergenza raggiunta. Arresto anticipato!")
        break
print("--- Esecuzione Terminata ---")

# -------------------------------------------------------------------
# Riepilogo e Generazione Grafici Finali
# -------------------------------------------------------------------
print("\n## Riepilogo dei Risultati ##")
print("Medie dei cluster trovati (temperature medie stimate per ogni stagione):")
medie_trovate = gmm.mu.flatten()
medie_trovate.sort()
for i, media in enumerate(medie_trovate):
    print(f"Stagione {i+1} (stimata): {media:.2f}°C")

# Chiamiamo le funzioni per creare entrambi i grafici
if draw_plots:
    # 1. Chiama la funzione per l'istogramma con le curve
    plot_1d_gmm(x, gmm, 'Distribuzione Dati e Cluster Stagionali')
    
    # 2. Chiama la funzione per il grafico a punti colorati
    plot_1d_scatter_clusters(x, gmm, 'Assegnazione Punti ai Cluster Stagionali')