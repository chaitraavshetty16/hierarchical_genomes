import numpy as np
from reservoirpy import nodes, datasets
from expt_helper_functions_v2 import (create_initial_genome, mutate_hox,transcribe_hierarchical_genome_to_weight_matrix, mae, mse)
import matplotlib.pyplot as plt
import copy

import reservoirpy as rpy
rpy.verbosity(0)

# Load the Mackey-Glass dataset
X = datasets.mackey_glass(n_timesteps=1000, sample_len=2000)
train_end = int(len(X) * 0.7)
test_start = train_end + 1

# Initialize the population
population_size = 5
genome = create_initial_genome(input_size=1, output_size=1, initial_connections=5)
population = [copy.deepcopy(genome) for _ in range(population_size)]
#population = [create_initial_genome(input_size=..., output_size=...) for _ in range(population_size)]


# Run the evolutionary process
best_fitness_scores = []
for i in range(50):
    fitness_scores = []
    new_population = []
    for genome in population:
        mutated_genome = mutate_hox(copy.deepcopy(genome), 0.1)
        new_population.append(mutated_genome)
        weight_matrix = transcribe_hierarchical_genome_to_weight_matrix(mutated_genome)
        esn = nodes.Reservoir(Win=np.ones((weight_matrix.shape[0], 1)), W=weight_matrix, bias=np.zeros((weight_matrix.shape[0], 1))) >> nodes.Ridge(ridge=1e-6)
        
        forecast = esn.fit(X[:train_end], X[1:train_end+1]).run(X[test_start:])
        fitness_rmse = np.sqrt(mse(forecast, X[test_start:]))
        fitness_mae = mae(forecast, X[test_start:])
        
        fitness_scores.append({'rmse': fitness_rmse, 'mae': fitness_mae})
    
    # Selection based on MAE
    best_genome_index = np.argmin([score['mae'] for score in fitness_scores])
    best_genome = new_population[best_genome_index]
    best_fitness_scores.append(fitness_scores[best_genome_index])
    
        # Reproduction (cloning the best genome for the next generation)
    population = [copy.deepcopy(best_genome) for _ in range(population_size)]
    
# Gather the best fitness scores for plotting
best_rmse_scores = [score['rmse'] for score in best_fitness_scores]
best_mae_scores = [score['mae'] for score in best_fitness_scores]
best_mse_scores = [mse(score['mae'], X[test_start:]) for score in best_fitness_scores]

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(best_rmse_scores, label='Best RMSE per Generation')
plt.plot(best_mae_scores, label='Best MAE per Generation')
plt.plot(best_mse_scores, label='Best MSE per Generation')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.title('Evolution of Fitness Scores Over Generations')
plt.legend()
plt.show()

