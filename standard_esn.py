import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Ridge, Reservoir
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse
from sklearn.metrics import r2_score

import hierarchical_genomes as hg

def main():
    # 1. Load Mackey-Glass data
    data = mackey_glass(10001)  # Generate data with one extra point for creating lagged values
    X = data[:-1]  # Input values
    Y = data[1:]   # Target values

    train_length = 500
    test_length = 500

    X_train, X_test = X[:train_length], X[train_length:train_length+test_length]
    Y_train, Y_test = Y[:train_length], Y[train_length:train_length+test_length]
    
    genome = [[0,1,0.5],[1,2,0.5],[2,3,0.1]]
    n_generations = 50
    population = 10
    genome_population = [genome for _ in range(population)]
    
    # 2. Prepare ESN
    for generation in range(n_generations):
        for genome in genome_population:
            if len(genome) > 2:
                mutated_genome, mutation = hg.mutate_genome_with_hox(genome)
                weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(mutated_genome)
                
                reservoir = Reservoir(W=weight_matrix)
                readout = Ridge(ridge=1e-6)
                esn = reservoir >> readout
                
                # Train the ESN
                forecast = esn.fit(X_train, Y_train).run(X_test)

                # 3. Evaluate the ESN
                current_rmse = rmse(forecast, Y_test)
                r2 = r2_score(Y_test, forecast)
                
                print(f"Generation {generation}, Genome {genome_population.index(genome)+1}")
                print(f"RMSE: {current_rmse}, R2: {r2}")

                # 4. Plotting actual vs predictions
                steps_ahead = [1, 2, 4, 10]
                for step in steps_ahead:
                    plt.figure(figsize=(10, 6))
                    plt.plot(Y_test[:100], 'b', label="Actual")
                    plt.plot(forecast[:100-step], 'r', label=f"Prediction (Step {step})")
                    plt.legend(loc="upper right")
                    plt.title(f"Actual vs Prediction (Step {step})")
                    plt.show()
                
    return

if __name__ == "__main__":
    main()
