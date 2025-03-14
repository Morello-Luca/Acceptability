import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

#########################################
# 1. Data Loading Function
#########################################

def load_rankings_from_csv(file_path):
    """
    Carica i ranking da un file CSV.
    
    Assunzioni:
      - La prima colonna è un identificativo (ad es. Respondent) e viene ignorata.
      - Le colonne successive contengono i ranking numerici delle feature.
        Un valore numerico più basso indica una maggiore preferenza.
    
    Parametri:
      file_path: percorso del file CSV.
    
    Restituisce:
      rankings: array NumPy dei ranking.
      feature_types: lista dei nomi delle feature.
    """
    df = pd.read_csv(file_path)
    feature_types = list(df.columns[1:])  # Escludo la prima colonna
    rankings = df[feature_types].to_numpy()
    return rankings, feature_types

#########################################
# 2. Bradley-Terry Model Functions
#########################################

def bradley_terry_probability(w, i, j):
    """
    Calcola la probabilità, secondo il modello Bradley-Terry, che la feature i
    sia preferita a j.
    
    Parametri:
      w: vettore dei pesi per ogni feature.
      i, j: indici delle feature da confrontare.
    
    Restituisce:
      Probabilità di preferire la feature i a j.
    """
    return w[i] / (w[i] + w[j])

def likelihood(w, rankings):
    """
    Calcola il negativo della log-likelihood totale su tutti i ranking.
    
    Per ciascun ranking (cioè per ciascun osservatore) si valuta il contributo
    alla log-likelihood per ogni coppia di feature in base all'ordine.
    
    Parametri:
      w: vettore dei pesi.
      rankings: array NumPy dei ranking numerici.
    
    Restituisce:
      Il negativo della log-likelihood totale (da minimizzare).
    """
    total_log_likelihood = 0.0
    for ranking in rankings:
        respondent_log_likelihood = 0.0
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if ranking[i] < ranking[j]:
                    respondent_log_likelihood += np.log(bradley_terry_probability(w, i, j))
                else:
                    respondent_log_likelihood += np.log(bradley_terry_probability(w, j, i))
        total_log_likelihood += respondent_log_likelihood
    return -total_log_likelihood

#########################################
# 3. MLE Weight Optimization (Multi-start)
#########################################

def optimize_weights_multistart(rankings, n_starts=5, local_methods=["L-BFGS-B", "SLSQP"]):
    """
    Ottimizza i pesi del modello Bradley-Terry con una strategia ibrida:
      1. Ottimizzazione globale tramite Differential Evolution.
      2. Ottimizzazioni locali con più punti di partenza.
      3. Raffinamento finale con il metodo L-BFGS-B.
    
    I pesi finali vengono normalizzati in modo che il massimo sia 1.
    
    Parametri:
      rankings: array NumPy dei ranking numerici.
      n_starts: numero di partenze casuali per l'ottimizzazione locale.
      local_methods: lista di metodi locali da utilizzare.
    
    Restituisce:
      normalized_weights: vettore dei pesi normalizzati.
      optimized_weights: vettore dei pesi ottimizzati (raw).
      solutions: lista dei risultati di ciascuna ottimizzazione.
    """
    num_weights = len(rankings[0])
    bounds = [(0, 10)] * num_weights

    best_solution = None
    best_obj = np.inf
    solutions = []

    # Strategia globale: Differential Evolution
    result_global = differential_evolution(lambda w: likelihood(w, rankings), bounds)
    solutions.append(("Differential Evolution", result_global))
    if result_global.fun < best_obj:
        best_obj = result_global.fun
        best_solution = result_global

    # Ottimizzazioni locali multi-start
    for method in local_methods:
        for i in range(n_starts):
            initial_weights = np.random.uniform(0, 10, num_weights)
            result_local = minimize(likelihood, initial_weights, args=(rankings,), method=method, bounds=bounds)
            solutions.append((f"Local {method} (start {i})", result_local))
            if result_local.fun < best_obj:
                best_obj = result_local.fun
                best_solution = result_local

    # Raffinamento finale con L-BFGS-B
    refined = minimize(likelihood, best_solution.x, args=(rankings,), method="L-BFGS-B", bounds=bounds)
    solutions.append(("Refined L-BFGS-B", refined))
    if refined.fun < best_obj:
        best_obj = refined.fun
        best_solution = refined

    optimized_weights = best_solution.x
    normalized_weights = optimized_weights / np.max(optimized_weights)
    
    return normalized_weights, optimized_weights, solutions

#########################################
# 4. Ranking Conversion and Weight Calculation
#########################################

def convert_numeric_to_ordering(rankings, feature_types):
    """
    Converte i ranking numerici in un ordinamento di feature.
    
    Ogni vettore di ranking numerico viene convertito in una lista di nomi di feature,
    ordinati dal più preferito (valore minore) al meno preferito.
    
    Parametri:
      rankings: array NumPy dei ranking numerici.
      feature_types: lista dei nomi delle feature.
    
    Restituisce:
      ordering_rankings: lista di liste di feature ordinate.
    """
    ordering_rankings = []
    for ranking in rankings:
        ordering = [feature_types[i] for i in np.argsort(ranking)]
        ordering_rankings.append(ordering)
    return ordering_rankings

def compute_ranking_weights_numeric(rankings, w):
    """
    Calcola i pesi basati sulla likelihood per ogni ranking osservato.
    
    Questi pesi derivano dalla log-likelihood calcolata con il vettore di pesi ottimizzato.
    
    Parametri:
      rankings: array NumPy dei ranking numerici.
      w: vettore dei pesi ottimizzati.
    
    Restituisce:
      Array dei pesi (uno per ciascun ranking osservato).
    """
    weights = []
    for ranking in rankings:
        ll = 0.0
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if ranking[i] < ranking[j]:
                    ll += np.log(bradley_terry_probability(w, i, j))
                else:
                    ll += np.log(bradley_terry_probability(w, j, i))
        weights.append(np.exp(ll))
    return np.array(weights)

#########################################
# 5. Consensus Methods
#########################################

def consensus_method_2(rankings, ranking_weights, feature_types):
    """
    Metodo di Consenso 2: Matrice di confronto a coppie ponderata.
    
    Per ogni ranking osservato (lista ordinata di feature) e il suo peso,
    viene costruita una matrice dei "vittoriosi" per ogni coppia.
    
    Parametri:
      rankings: lista di ranking ordinati (liste di feature).
      ranking_weights: array dei pesi per ciascun ranking.
      feature_types: lista dei nomi delle feature.
    
    Restituisce:
      consensus_ranking: lista di feature ordinate per consenso (dal migliore al peggiore).
    """
    feature_to_index = {feature: idx for idx, feature in enumerate(feature_types)}
    num_features = len(feature_types)
    win_matrix = np.zeros((num_features, num_features))
    
    for ranking, weight in zip(rankings, ranking_weights):
        ranking_indices = [feature_to_index[item] for item in ranking if item in feature_to_index]
        for i in range(len(ranking_indices)):
            for j in range(i + 1, len(ranking_indices)):
                winner = ranking_indices[i]
                loser = ranking_indices[j]
                win_matrix[winner, loser] += weight

    consensus_scores = np.sum(win_matrix, axis=1)
    consensus_ranking = [feature_types[i] for i in np.argsort(-consensus_scores)]
    return consensus_ranking

def pairwise_disagreements(candidate, ranking):
    """
    Calcola il numero di disaccordi a coppie tra un ranking candidato e uno osservato.
    
    Parametri:
      candidate: ranking candidato (lista di feature).
      ranking: ranking osservato (lista di feature).
    
    Restituisce:
      Numero di disaccordi a coppie.
    """
    order_obs = {item: pos for pos, item in enumerate(ranking)}
    disagreements = 0
    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            if candidate[i] not in order_obs or candidate[j] not in order_obs:
                continue
            if order_obs[candidate[i]] > order_obs[candidate[j]]:
                disagreements += 1
    return disagreements

def total_weighted_KY_distance(candidate, rankings, ranking_weights):
    """
    Calcola la distanza totale pesata secondo il criterio Kemeny-Young
    per un ranking candidato.
    
    Ogni ranking osservato contribuisce con un costo (disaccordi) ponderato.
    
    Parametri:
      candidate: ranking candidato (lista di feature).
      rankings: lista di ranking ordinati osservati.
      ranking_weights: array dei pesi per ciascun ranking.
    
    Restituisce:
      Distanza totale pesata.
    """
    total_distance = 0.0
    for idx, ranking in enumerate(rankings):
        dis = pairwise_disagreements(candidate, ranking)
        total_distance += ranking_weights[idx] * dis
    return total_distance

def kendall_tau_distance(rank1, rank2):
    """
    Calcola la Kendall tau distance tra due ranking.
    
    La distanza corrisponde al numero di coppie discordanti.
    
    Parametri:
      rank1, rank2: due ranking (liste di feature).
    
    Restituisce:
      Distanza Kendall tau (numero intero).
    """
    n = len(rank1)
    pos2 = {item: i for i, item in enumerate(rank2)}
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos2[rank1[i]] > pos2[rank1[j]]:
                discordant += 1
    return discordant

def refined_consensus_method_3(rankings, ranking_weights, items, mle_order_names, lam=1.0, max_iter=1000):
    """
    Metodo di Consenso 3: Ranking di consenso regolarizzato con ricerca locale.
    
    Obiettivo:
      Costo totale = Distanza Kemeny-Young pesata + λ * Kendall Tau Distance
      (rispetto all'ordine MLE)
    
    Si parte dall'ordine MLE e si eseguono scambi a coppie per minimizzare il costo.
    
    Parametri:
      rankings: lista di ranking ordinati osservati.
      ranking_weights: array dei pesi per ciascun ranking.
      items: lista dei nomi delle feature.
      mle_order_names: ranking indotto dai pesi MLE (lista di feature).
      lam: parametro di regolarizzazione.
      max_iter: numero massimo di iterazioni per la ricerca locale.
    
    Restituisce:
      best_candidate: ranking di consenso finale.
      best_cost: costo totale della soluzione finale.
    """
    def objective(candidate):
        ky_distance = total_weighted_KY_distance(candidate, rankings, ranking_weights)
        kt_distance = kendall_tau_distance(candidate, mle_order_names)
        return ky_distance + lam * kt_distance

    current_candidate = list(mle_order_names)
    current_cost = objective(current_candidate)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        for i in range(len(current_candidate)):
            for j in range(i + 1, len(current_candidate)):
                new_candidate = current_candidate.copy()
                new_candidate[i], new_candidate[j] = new_candidate[j], new_candidate[i]
                new_cost = objective(new_candidate)
                if new_cost < current_cost:
                    current_candidate = new_candidate
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        iter_count += 1
    return current_candidate, current_cost

#########################################
# 6. Main Function
#########################################

def main():
    # Carica i ranking numerici e i nomi delle feature dal CSV
    file_path = 'list.csv'  # Specifica il percorso del file CSV
    numeric_rankings, feature_types = load_rankings_from_csv(file_path)
    
    # Ottimizzazione dei pesi con approccio multi-start
    normalized_weights, optimized_weights, solutions = optimize_weights_multistart(numeric_rankings)
    
    print("Optimized Weights (raw):", optimized_weights)
    print("Normalized Weights:", normalized_weights)
    print("\nRisultati delle ottimizzazioni:")
    for name, sol in solutions:
        print(f"{name}: Valore Obiettivo = {sol.fun}")
    
    # Creazione di un DataFrame per i pesi normalizzati per ciascuna feature
    normalized_weights_df = pd.DataFrame({
        "Feature": feature_types,
        "Normalized Weight": normalized_weights
    })
    print("\nNormalized Weights per Feature:")
    print(normalized_weights_df)
    
    # Converte i ranking numerici in ordinamenti (liste di feature, best-first)
    ordering_rankings = convert_numeric_to_ordering(numeric_rankings, feature_types)
    
    # Calcola i pesi (likelihood) per ciascun ranking osservato
    ranking_weights = compute_ranking_weights_numeric(numeric_rankings, optimized_weights)
    print("\nLikelihood Weights per Ranking:", ranking_weights)
    
    # Deriva l'ordine MLE basato sui pesi normalizzati
    mle_order_indices = list(np.argsort(-normalized_weights))
    mle_order_names = [feature_types[i] for i in mle_order_indices]
    print("\nMLE Ranking Order:", mle_order_names)
    
    # Metodo di Consenso 2: Matrice di confronto a coppie ponderata
    consensus2 = consensus_method_2(ordering_rankings, ranking_weights, feature_types)
    print("\nConsensus Ranking Method 2:")
    print(consensus2)
    ky_distance2 = total_weighted_KY_distance(consensus2, ordering_rankings, ranking_weights)
    print("Total Weighted Kemeny-Young Distance (Method 2):", ky_distance2)
    
    # Metodo di Consenso 3: Ranking regolarizzato con ricerca locale
    lam = 1.0  # Parametro di regolarizzazione (modificabile)
    consensus3, cost3 = refined_consensus_method_3(ordering_rankings, ranking_weights, feature_types, mle_order_names, lam)
    print("\nConsensus Ranking Method 3 (Regularized):")
    print(consensus3)
    print("Total Regularized Cost (Method 3):", cost3)
    
    # Ranking finale basato sui pesi MLE normalizzati
    ranking_indices = np.argsort(normalized_weights)[::-1]
    final_ranking = [feature_types[idx] for idx in ranking_indices]
    print("\nFinal Ranking delle Feature (basato sui pesi MLE normalizzati):")
    for i, idx in enumerate(ranking_indices, 1):
        print(f"{i}. {feature_types[idx]} (Weight: {normalized_weights[idx]:.4f})")

if __name__ == "__main__":
    main()
