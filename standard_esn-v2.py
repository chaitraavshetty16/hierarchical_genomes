import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Ridge, Reservoir
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse, rsquare

import hierarchical_genomes as hg

# Data Generation
def generate_mackey_glass_data(n_timesteps=2001):  # Adjusted the number to fit the prediction
    """Generate the Mackey-Glass dataset."""
    return mackey_glass(n_timesteps=n_timesteps)

# Visualization
def plot_comparison(actual, predictions, title="Actual vs Predictions"):
    """Visualize actual data against predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='dodgerblue', label='Actual', linewidth=2)
    plt.plot(predictions, color='darkorange', label='Predicted', linestyle='dashed', linewidth=2)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc="upper right")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def main():
    # Load Mackey-Glass data
    X = generate_mackey_glass_data()
    train_length = 1500
    test_length = 500
    genome = [[0,1,0.5],[1,2,0.5],[2,3,0.1]]
    n_generations = 50
    population = 10
    genome_population = [genome for _ in range(population)]

    # Main Loop
    for generation in range(n_generations):
        for genome in genome_population:
            if len(genome) > 2:
                mutated_genome, mutation = hg.mutate_genome_with_hox(genome)
                weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(mutated_genome)

                reservoir = Reservoir(W=weight_matrix)
                readout = Ridge(output_dim=1, ridge=1e-6)
                esn = reservoir >> readout
                
                steps_to_predict = [1, 2, 4, 10]
                for steps in steps_to_predict:
                    esn.fit(X[:train_length-steps], X[steps:train_length], warmup=100)
                    predictions = esn.run(X[train_length-steps:train_length+test_length-steps])

                    # Visualize and compare results
                    #plot_comparison(X[train_length:train_length+len(predictions)], predictions, title=f"Actual vs Predictions for {steps} timesteps ahead")

                    # Gather the model's performance data
                    current_rmse = rmse(X[train_length:train_length+len(predictions)], predictions)
                    current_rsquare = rsquare(X[train_length:train_length+len(predictions)], predictions)

                    # Print performance metrics
                    print(f"Generation {generation}, Genome {genome_population.index(genome)+1}")
                    print(f"Predictions for {steps} timesteps ahead - RMSE:", current_rmse)
                    print(f"Predictions for {steps} timesteps ahead - R^2 score:", current_rsquare)
                    
    return

if __name__ == "__main__":
    main()
