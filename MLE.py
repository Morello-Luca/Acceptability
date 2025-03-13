import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

# Funzione per caricare i ranking da un file CSV
def load_rankings_from_csv(file_path):
    df = pd.read_csv(file_path)                  # Carica il CSV in un DataFrame di pandas
    feature_types = list(df.columns[1:])         # Esclude la prima colonna ('Respondent')
    rankings = df[feature_types].to_numpy()      # Converte i ranking in una matrice NumPy
    return rankings, feature_types

# Modello di Bradley-Terry per calcolare la probabilità di preferenza a coppie
def bradley_terry_probability(w, i, j):
    return w[i] / (w[i] + w[j])

# Funzione di likelihood (qui log-likelihood) da massimizzare (minimizziamo il negativo)
def likelihood(w, rankings):
    total_log_likelihood = 0.0  # Inizializza il log-likelihood totale
    
    # Itera su ogni ranking (ogni persona)
    for ranking in rankings:
        respondent_log_likelihood = 0.0  # Log-likelihood per questo ranking
        
        # Calcola il contributo per ogni coppia di feature (audio types)
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if ranking[i] < ranking[j]:  # Se la feature i è preferita a j
                    respondent_log_likelihood += np.log(bradley_terry_probability(w, i, j))
                else:  # Se j è preferita a i
                    respondent_log_likelihood += np.log(bradley_terry_probability(w, j, i))
        
        total_log_likelihood += respondent_log_likelihood
    
    return -total_log_likelihood  # Restituisce il negativo (per minimizzare)

# Funzione per ottimizzare i pesi utilizzando una strategia multi-start e ibrida
def optimize_weights_multistart(rankings, n_starts=5, local_methods=["L-BFGS-B", "SLSQP"]):
    num_weights = len(rankings[0])
    # Per i metodi locali, impostiamo dei limiti: ogni peso deve essere >= 0; per Differential Evolution 
    # occorrono limiti finiti, ad esempio (0, 10)
    bounds = [(0, 10)] * num_weights

    best_solution = None
    best_obj = np.inf  # Obiettivo migliore (minore = migliore)
    solutions = []    # Per raccogliere i risultati di ogni ottimizzazione

    # 1. Strategia globale: Differential Evolution
    result_global = differential_evolution(lambda w: likelihood(w, rankings), bounds)
    solutions.append(("Differential Evolution", result_global))
    if result_global.fun < best_obj:
        best_obj = result_global.fun
        best_solution = result_global

    # 2. Multi-start: esegue ottimizzazioni locali da diversi punti iniziali
    for method in local_methods:
        for i in range(n_starts):
            # Genera un punto iniziale casuale nei bounds
            initial_weights = np.random.uniform(0, 10, num_weights)
            result_local = minimize(likelihood, initial_weights, args=(rankings,), method=method, bounds=bounds)
            solutions.append((f"Local {method} (start {i})", result_local))
            if result_local.fun < best_obj:
                best_obj = result_local.fun
                best_solution = result_local

    # 3. Strategia ibrida: raffina la soluzione migliore trovata con un metodo locale (es. L-BFGS-B)
    refined = minimize(likelihood, best_solution.x, args=(rankings,), method="L-BFGS-B", bounds=bounds)
    solutions.append(("Raffinato L-BFGS-B", refined))
    if refined.fun < best_obj:
        best_obj = refined.fun
        best_solution = refined

    # Normalizzazione: scala i pesi in modo che il massimo sia 1
    optimized_weights = best_solution.x
    normalized_weights = optimized_weights / np.max(optimized_weights)
    
    return normalized_weights, optimized_weights, solutions

# Funzione principale
def main():
    file_path = 'list.csv'  # Percorso del file CSV con i ranking
    rankings, feature_types = load_rankings_from_csv(file_path)

    normalized_weights, optimized_weights, solutions = optimize_weights_multistart(rankings)

    print("Optimized Weights (prima della normalizzazione):", optimized_weights)
    print("Normalized Weights (pesi di soddisfazione finali):", normalized_weights)
    print("\nConfronto dei risultati:")
    for name, sol in solutions:
        print(f"{name}: Valore Obiettivo = {sol.fun}")

    # Ordinamento finale in base ai pesi normalizzati (dal più preferito al meno)
    ranking = np.argsort(normalized_weights)[::-1]
    ranked_audio = [feature_types[i] for i in ranking]
    ranked_audio_weights = [normalized_weights[i] for i in ranking]

    print("\nRanking finale delle feature (più a meno preferite):")
    for i, (audio, weight) in enumerate(zip(ranked_audio, ranked_audio_weights), 1):
        print(f"{i}. {audio} (Normalized Weight: {weight:.4f})")

if __name__ == "__main__":
    main()
