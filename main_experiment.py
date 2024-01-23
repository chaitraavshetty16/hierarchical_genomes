# Import necessary libraries
import numpy as np
from reservoirpy import nodes, datasets, observables
import hierarchical_genomes as hg
import networkx as nx
import matplotlib.pyplot as plt
import copy

# Import functions from helper file
from expt_helper_functions import create_initial_genome, select_best_genomes, reproduce, log_generation_results, analyze_results

# Set up the main parameters
population_size = 100 #50
n_generations = 300  #100
mutation_probability = 0.1
insertion_probability = 0.1

# Define the number of input and output nodes for your neural network
num_input_nodes = 10  
num_output_nodes = 10  

# Load the Mackey-Glass dataset with specified timesteps
n_timesteps = 1000  
X = datasets.mackey_glass(n_timesteps=n_timesteps, sample_len=1000)

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
    forecast = esn.fit(X[:500], X[1:501]).run(X[502:])
    
    # Calculate fitness (e.g., using RMSE)
    fitness_score = observables.rmse(forecast, X[502:])
    return fitness_score

# Evolutionary loop
for generation in range(n_generations):
    fitness_scores = [evaluate_fitness(genome) for genome in genome_population]
    selected_genomes = select_best_genomes(genome_population, fitness_scores)
    
    # Update the call to 'reproduce' to include node counts
    new_population = reproduce(selected_genomes, population_size, mutation_probability, num_input_nodes, num_output_nodes, num_elites)
    genome_population = new_population

    # Correct file path for logging
    log_path = "/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt"
    log_generation_results(generation, selected_genomes, fitness_scores, log_file=log_path)

# Perform post-experiment analysis with the correct file path
analyze_log_path = "/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt"
analyze_results(log_file=analyze_log_path)
