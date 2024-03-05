import numpy as np
from reservoirpy import nodes, datasets, observables
import reservoirpy as rpy
rpy.verbosity(0)
import hierarchical_genomes as hg
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random

# Import your helper functions from a separate file if they're not already here
from expt_helper_functions import create_initial_genome, reproduce, mae, mse, transcribe_hierarchical_genome_to_weight_matrix, select_best_genomes


# Set up the main parameters
population_size = 50
n_generations = 100
mutation_probability = 0.1

# Global settings
desired_spectral_radius = 0.95
train_split = 0.7

# Load the dataset
n_timesteps = 2000
X = datasets.mackey_glass(n_timesteps=n_timesteps, sample_len=2000)
X = np.nan_to_num(X, nan=0.0)

# Define training and testing splits
num_input_nodes = X.shape[1]
num_output_nodes = 1
train_end = int(len(X) * train_split)
test_start = train_end + 1

elitism_factor = 0.1  # Assuming you want to carry over 10% of the population
num_elites = int(elitism_factor * population_size)

# Define the fitness evaluation function (reuse from main experiment)
def evaluate_fitness(genome):
    # Transcribe genome to a weight matrix
    weight_matrix = transcribe_hierarchical_genome_to_weight_matrix(genome)
    
    # Check and adjust the spectral radius
    spectral_radius = np.max(np.abs(np.linalg.eigvals(weight_matrix)))
    if spectral_radius >= 1:
        weight_matrix *= desired_spectral_radius / spectral_radius
    
    # Set up the Echo State Network
    esn = nodes.Reservoir(W=weight_matrix, spectral_radius=desired_spectral_radius) >> nodes.Ridge(ridge=1e-4)
    
    num_reservoir_nodes = weight_matrix.shape[0]  # Assuming weight_matrix is a square matrix
    num_input_features = 1  # Adjust based on your specific input features

    # Setup the Echo State Network with adjusted Win and bias
    ###esn = nodes.Reservoir(Win=np.ones((num_reservoir_nodes, num_input_features)), W=weight_matrix, bias=np.zeros((num_reservoir_nodes, 1)), spectral_radius=desired_spectral_radius) >> nodes.Ridge(ridge=1e-6)
    ###esn = nodes.Reservoir(Win=np.ones((8,1)), W=weight_matrix, bias=np.zeros((8,1))) >> nodes.Ridge(ridge=1e-6)

    
    try:
        # Train and forecast using the ESN
        esn.fit(X[:train_end], X[1:train_end+1])
        forecast = esn.run(X[test_start:])
        
        # Calculate fitness
        fitness_rmse = observables.rmse(forecast, X[test_start:])
        fitness_mae = mae(forecast, X[test_start:])
        fitness_mse = mse(forecast, X[test_start:])
        
        return {'rmse': fitness_rmse, 'mae': fitness_mae, 'mse': fitness_mse, 'forecast': forecast}
    except Exception as e:
        print(f"An exception occurred during fitness evaluation: {e}")
        return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf'), 'forecast': None}


# Hox without mutation
def run_hox_without_mutation():
    genome_population = [create_initial_genome() for _ in range(population_size)]
    best_fitness_scores = []
    
    for generation in range(n_generations):
        fitness_scores = [evaluate_fitness(genome) for genome in genome_population]
        selected_genomes = select_best_genomes(genome_population, fitness_scores)
        # Reproduce without mutation
        genome_population = reproduce(selected_genomes, population_size, 0, num_input_nodes, num_output_nodes, num_elites)
        
        # Logging and analysis
        best_fitness = min(fitness_scores, key=lambda x: x['rmse'])
        best_fitness_scores.append(best_fitness['rmse'])
        print(f"Generation {generation} - Best RMSE: {best_fitness['rmse']}")
    
    # Plotting the results
    plot_fitness_scores(best_fitness_scores)

# Hox with mutation
def run_hox_with_mutation():
    genome_population = [create_initial_genome() for _ in range(population_size)]
    best_fitness_scores = []
    
    for generation in range(n_generations):
        fitness_scores = [evaluate_fitness(genome) for genome in genome_population]
        selected_genomes = select_best_genomes(genome_population, fitness_scores)
        # Reproduce with mutation
        genome_population = reproduce(selected_genomes, population_size, mutation_probability, num_input_nodes, num_output_nodes, num_elites)
        
        # Logging and analysis
        best_fitness = min(fitness_scores, key=lambda x: x['rmse'])
        best_fitness_scores.append(best_fitness['rmse'])
        print(f"Generation {generation} - Best RMSE: {best_fitness['rmse']}")
    
    # Plotting the results
    plot_fitness_scores(best_fitness_scores)

def plot_fitness_scores(best_fitness_scores):
    import matplotlib.pyplot as plt
    plt.plot(best_fitness_scores)
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness Score (RMSE)')
    plt.title('Best Fitness Score by Generation')
    plt.show()




if __name__ == "__main__":
    run_hox_without_mutation()
    run_hox_with_mutation()
