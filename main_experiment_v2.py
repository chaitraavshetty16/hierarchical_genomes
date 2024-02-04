import numpy as np
from reservoirpy import nodes, datasets, observables
import reservoirpy as rpy
import hierarchical_genomes as hg
import matplotlib.pyplot as plt
from expt_helper_functions_v2 import (create_initial_genome, select_best_genomes, reproduce, log_generation_results, analyze_results, mae, mse, calculate_diversity_score, should_increase_timestep)

# Main parameters setup
population_size = 100
n_generations = 200
mutation_probability = 0.1
num_input_nodes, num_output_nodes = 10, 10
elitism_factor = 0.1
num_elites = int(elitism_factor * population_size)

# Initialize population
genome_population = [create_initial_genome(num_input_nodes, num_output_nodes) for _ in range(population_size)]

# Fitness evaluation function
def evaluate_fitness(genome):
    weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(genome)
    esn = nodes.Reservoir(W=weight_matrix) >> nodes.Ridge(ridge=1e-6)
    forecast = esn.fit(X[:train_end], X[1:train_end+1]).run(X[test_start:])
    return {'rmse': observables.rmse(forecast, X[test_start:]), 'mae': mae(forecast, X[test_start:]), 'mse': mse(forecast, X[test_start:])}

# Evolutionary loop with timestep logic
initial_timestep = 1000
current_timestep = initial_timestep
timestep_increment = 10
increment_interval = 5
best_rmse_scores, best_mae_scores, best_mse_scores, diversity_scores = [], [], [], []

for generation in range(n_generations):
    X = datasets.mackey_glass(current_timestep, sample_len=1000)
    train_end = int(len(X) * 0.5)
    test_start = train_end + 1

    fitness_scores = [evaluate_fitness(genome) for genome in genome_population]
    best_rmse = min(fitness_scores, key=lambda x: x['rmse'])['rmse']
    best_mae = min(fitness_scores, key=lambda x: x['mae'])['mae']
    best_mse = min(fitness_scores, key=lambda x: x['mse'])['mse']
    best_rmse_scores.append(best_rmse)
    best_mae_scores.append(best_mae)
    best_mse_scores.append(best_mse)

    if should_increase_timestep(generation, best_rmse_scores, 10, increment_interval):
        current_timestep += timestep_increment
        print(f"Increasing timestep to {current_timestep}")

    selected_genomes = select_best_genomes(genome_population, fitness_scores, 5, elitism_factor)
    current_diversity_score = calculate_diversity_score(genome_population)
    diversity_scores.append(current_diversity_score)

    new_population = reproduce(selected_genomes, population_size, mutation_probability, num_input_nodes, num_output_nodes, num_elites)
    genome_population = new_population

    log_generation_results(generation, selected_genomes, fitness_scores, "/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt")

analyze_results("/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt", best_rmse_scores, best_mae_scores, best_mse_scores, diversity_scores)
