# Import necessary libraries
import numpy as np
from reservoirpy import nodes, datasets, observables
import reservoirpy as rpy
rpy.verbosity(0)
#rpy.set_seed(42)
import hierarchical_genomes as hg
import networkx as nx
import matplotlib.pyplot as plt
import copy

# Import functions from helper file
from expt_helper_functions import create_initial_genome, select_best_genomes, reproduce, log_generation_results, analyze_results,mae,mse,calculate_diversity_score

# Set up the main parameters
population_size = 100 #50
n_generations =    200  #100
mutation_probability = 0.1
insertion_probability = 0.1

# Define the number of input and output nodes for your neural network
num_input_nodes = 10  
num_output_nodes = 10  

# Load the Mackey-Glass dataset with specified timesteps
n_timesteps = 1000  
X = datasets.mackey_glass(n_timesteps=n_timesteps, sample_len=1000)
train_end = int(len(X) * 0.5)
test_start = train_end + 1


# Automatic timestep calculation
train_split = 0.5
train_timesteps = int(train_split * n_timesteps)
test_timesteps = train_timesteps + 1

elitism_factor = 0.1  # Assuming you want to carry over 10% of the population
num_elites = int(elitism_factor * population_size)



# Initialize your population
genome_population = [create_initial_genome() for _ in range(population_size)]



# Define a fitness evaluation function
def evaluate_fitness(genome):
    # Convert genome to neural network (function from your thesis code)
    weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(genome)
    
    # Setup the Echo State Network
    esn = nodes.Reservoir(W=weight_matrix) >> nodes.Ridge(ridge=1e-6)
    
    # Train and forecast using the ESN
    #forecast = esn.fit(X[:500], X[1:501]).run(X[503:])
    
    # Calculate fitness (e.g., using RMSE)
    #fitness_score = observables.rmse(forecast, X[503:])
    
    # Train and forecast using the ESN
    forecast = esn.fit(X[:train_end], X[1:train_end+1]).run(X[test_start:])
    
    # Calculate fitness (using RMSE and MAE)
    fitness_rmse = observables.rmse(forecast, X[test_start:])
    fitness_mae = mae(forecast, X[test_start:])
    fitness_mse = mse(forecast, X[test_start:])
    
    return {'rmse': fitness_rmse, 'mae': fitness_mae, 'mse': fitness_mse}

    #return fitness_score

# Track best fitness score per generation
best_fitness_scores = []

# Initialize arrays to store the best RMSE and MAE scores
best_rmse_scores = []
best_mae_scores = []
best_mse_scores = []


previous_best_score = float('inf')
# Early stopping parameters
stagnation_threshold = 10  # Number of generations without improvement
stagnation_counter = 0  # Counter for stagnation

diversity_scores = []
    
# Evolutionary loop
for generation in range(n_generations):
    fitness_scores = [evaluate_fitness(genome) for genome in genome_population]
    
    #generation_best_score = min(fitness_scores, key=lambda x: x['rmse'])['rmse']
    #best_fitness_scores.append(generation_best_score)
    
    best_rmse = min(fitness_scores, key=lambda x: x['rmse'])['rmse']
    best_mae = min(fitness_scores, key=lambda x: x['mae'])['mae']
    best_mse = min(fitness_scores, key=lambda x: x['mse'])['mse']
    
    # Check for early stopping if the best score doesn't improve
    # if best_rmse >= previous_best_score:
    #     stagnation_counter += 1
    # else:
    #     stagnation_counter = 0
    #     previous_best_score = best_rmse
    
    # if stagnation_counter >= 10:  # stop if no improvement in 10 generations
    #     print(f"Early stopping at generation {generation} due to lack of improvement.")
    #     break
    
    if fitness_scores and fitness_scores[0]['rmse'] < previous_best_score:
        previous_best_score = fitness_scores[0]['rmse']
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    if stagnation_counter >= stagnation_threshold:
        print(f"\nEarly stopping at generation {generation} due to lack of improvement.")
        break
    
    
    best_rmse_scores.append(best_rmse)
    best_mae_scores.append(best_mae)
    best_mse_scores.append(best_mse)

    #print(f"Generation {generation}: Best Fitness Score: {generation_best_score}")
    
    
    
    
    selected_genomes = select_best_genomes(genome_population, fitness_scores)
    
    # Calculate and record the population's genetic diversity for this generation
    current_diversity_score = calculate_diversity_score(genome_population)
    diversity_scores.append(current_diversity_score)
    print(f"Generation {generation}: Diversity Score: {current_diversity_score}")

    
    # Update the call to 'reproduce' to include node counts
    new_population = reproduce(selected_genomes, population_size, mutation_probability, num_input_nodes, num_output_nodes, num_elites)
    genome_population = new_population

    # Correct file path for logging
    log_path = "/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt"
    log_generation_results(generation, selected_genomes, fitness_scores, log_file=log_path)

# Perform post-experiment analysis with the correct file path
analyze_log_path = "/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt"
analyze_results(log_file=analyze_log_path, best_rmse_scores=best_rmse_scores, best_mae_scores=best_mae_scores, best_mse_scores=best_mse_scores)
