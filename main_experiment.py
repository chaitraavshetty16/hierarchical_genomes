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
import random
#import json

# Loading configuration
#with open('config.json', 'r') as config_file:
#    config = json.load(config_file)

# Example of accessing a config value
#population_size = config['population_size']


# Import functions from helper file
from expt_helper_functions import create_initial_genome, select_best_genomes, reproduce, log_generation_results, analyze_results,mae,mse,calculate_diversity_score, should_increase_timestep,visualize_genome, calculate_fitness_numba,mutate_hox_nested,adjust_mutation_rate,novelty_search,plot_actual_vs_predicted,validate_genome,hox_like_mutation, transcribe_hierarchical_genome_to_weight_matrix

# Set up the main parameters
population_size = 100 #50
n_generations =    200  #100
mutation_probability = 0.1 #0.2
insertion_probability = 0.1

# Define the number of input and output nodes for your neural network
#num_input_nodes = 10  
#num_output_nodes = 10  

# Load the Mackey-Glass dataset with specified timesteps
n_timesteps = 2000  
X = datasets.mackey_glass(n_timesteps=n_timesteps, sample_len=2000)

# Check for NaN values in the dataset
if np.any(np.isnan(X)):
    print("Warning: NaN values found in the dataset. Attempting to handle them.")
    # Handle NaN values here. For example, you might choose to fill them with a specific value,
    # interpolate, or drop the rows/columns containing NaNs depending on your requirement.
    # Here's an example to fill NaNs with 0, which might not be the best approach for your case:
    X = np.nan_to_num(X, nan=0.0)
else:
    print("No NaN values found in the dataset.")
    
num_input_nodes = X.shape[1]  # Dynamically set based on the number of features in X
num_output_nodes = 1
train_end = int(len(X) * 0.7)
test_start = train_end + 1


# Automatic timestep calculation
train_split = 0.5
train_timesteps = int(train_split * n_timesteps)
test_timesteps = train_timesteps + 1
timestep_increment = 10

elitism_factor = 0.1  # Assuming you want to carry over 10% of the population
num_elites = int(elitism_factor * population_size)

actual_series = X[test_start:]  # The actual series data
generations = range(n_generations)  # Generation numbers
predicted_series = []

# Initialize your population
genome_population = [create_initial_genome() for _ in range(population_size)]


# Define a fitness evaluation function
def evaluate_fitness(genome):
    
    # Ensure the input data has no NaN values
    if np.any(np.isnan(X)):
        print("Input data contains NaN values.")
        return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf'), 'forecast': None}


    # Convert genome to neural network (function from your thesis code)
    weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(genome)
    ####weight_matrix = transcribe_hierarchical_genome_to_weight_matrix(genome)
    
    # Check the spectral radius of the reservoir
    ##spectral_radius = np.max(np.abs(np.linalg.eigvals(weight_matrix)))
    ##if spectral_radius >= 1:
    ##    print("Warning: Spectral radius >= 1 might lead to instability")
        
    # Assuming weight_matrix is the weight matrix of the reservoir
    ####current_spectral_radius = np.max(np.abs(np.linalg.eigvals(weight_matrix)))
    ####desired_spectral_radius = 0.95  # for example, less than 1
    ####if current_spectral_radius >= 1:
    ####    weight_matrix *= desired_spectral_radius / current_spectral_radius

    
    # Setup the Echo State Network
    ####esn = nodes.Reservoir(W=weight_matrix) >> nodes.Ridge(ridge=1e-4)
    #####esn = nodes.Reservoir(W=weight_matrix, spectral_radius=0.95) >> nodes.Ridge(ridge=1e-4)

    
    # Setup the Echo State Network with explicit Win and bias
    #esn = nodes.Reservoir(Win=np.ones((num_input_nodes, 1)), W=weight_matrix, bias=np.zeros((len(weight_matrix), 1))) >> nodes.Ridge(ridge=1e-6)
    
    num_reservoir_nodes = weight_matrix.shape[0]  # Assuming weight_matrix is a square matrix
    num_input_features = 1  # Adjust based on your specific input features

    # Setup the Echo State Network with adjusted Win and bias
    ####esn = nodes.Reservoir(Win=np.ones((num_reservoir_nodes, num_input_features)), W=weight_matrix, bias=np.zeros((num_reservoir_nodes, 1)), spectral_radius=desired_spectral_radius) >> nodes.Ridge(ridge=1e-6)
    esn = nodes.Reservoir(Win=np.ones((num_reservoir_nodes, num_input_features)), W=weight_matrix, bias=np.zeros((num_reservoir_nodes, 1))) >> nodes.Ridge(ridge=1e-6)

    # Train and forecast using the ESN
    #forecast = esn.fit(X[:500], X[1:501]).run(X[503:])
    
    # Calculate fitness (e.g., using RMSE)
    #fitness_score = observables.rmse(forecast, X[503:])
    
    # Train and forecast using the ESN
    #####forecast = esn.fit(X[:train_end], X[1:train_end+1]).run(X[test_start:])
    ###predicted_series = esn.fit(X[:train_end], X[1:train_end+1]).run(X[test_start:])  # The predicted series data
    # Check for NaN in the weight matrix
    if np.any(np.isnan(weight_matrix)):
        print("Weight matrix contains NaN values.")
        return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf'), 'forecast': None}

    try:
        # Train and forecast using the ESN
        esn.fit(X[:train_end], X[1:train_end+1])
        forecast = esn.run(X[test_start:])

        # Check if forecast contains nan values
        if np.any(np.isnan(forecast)):
            print(f"Warning: Forecast contains nan values for genome: {genome}")
            return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf'), 'forecast': None}

        # Calculate fitness (using RMSE and MAE)
        fitness_rmse = observables.rmse(forecast, X[test_start:])
        fitness_mae = mae(forecast, X[test_start:])
        fitness_mse = mse(forecast, X[test_start:])
        
        # Add a penalty for genome complexity
        ####complexity_penalty = 0.001 * len(genome)
        ####fitness_rmse += complexity_penalty
        
        return {'rmse': fitness_rmse, 'mae': fitness_mae, 'mse': fitness_mse, 'forecast': forecast}

    except Exception as e:
        print(f"An exception occurred during fitness evaluation: {e}")
        return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf'), 'forecast': None}
    
    
    """ # Debugging: Check if forecast contains nan values
    if np.any(np.isnan(forecast)):
        print(f"Warning: Forecast contains nan values for genome: {genome}")
        return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf'), 'forecast': forecast}

    else:
        # Calculate fitness (using RMSE and MAE)
        fitness_rmse = observables.rmse(forecast, X[test_start:])
        #fitness_rmse = calculate_fitness_numba(forecast, X[test_start:])
        fitness_mae = mae(forecast, X[test_start:])
        fitness_mse = mse(forecast, X[test_start:])
        
        # Check if any values are NaN and handle them
        ###if np.isnan(fitness_rmse) or np.isnan(fitness_mae) or np.isnan(fitness_mse):
        ###    return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf')}
        
        
        #return {'rmse': fitness_rmse, 'mae': fitness_mae, 'mse': fitness_mse}

        #complexity_penalty = len(genome) * 0.05 #complexity_penalty_rate (0.01 to 0.05)  # Define complexity_penalty_rate
        #return {'rmse': fitness_rmse + complexity_penalty, 'mae': fitness_mae, 'mse': fitness_mse}

        # Add a penalty for genome complexity
        complexity_penalty = 0.001 * len(genome)  # Adjust penalty coefficient as needed
        fitness_rmse += complexity_penalty
        
        if None in [fitness_rmse, fitness_mae, fitness_mse]:
            # Handle the case where fitness calculations failed
            return {'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf') , 'forecast': forecast}
        else:
            # If all values are calculated, return them
            return {'rmse': fitness_rmse, 'mae': fitness_mae, 'mse': fitness_mse, 'forecast': forecast}
"""
        #return {'rmse': fitness_rmse, 'mae': fitness_mae, 'mse': fitness_mse}

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

# Timesteps
initial_timestep = 1  # Define an initial value for the timestep
current_timestep = initial_timestep  # Define an initial timestep
timestep_increment = 1  # Define how much to increase timesteps each time
stagnation_threshold = 10  # Number of generations to consider for stagnation
increment_interval = 5     # Check for stagnation every 5 generations



forecasts = []

# Evolutionary loop
for generation in range(n_generations):
    
    # Load and preprocess data with current timestep
    #X = datasets.mackey_glass(current_timestep, sample_len=2000)
    
    # for genome in genome_population:
    #     #forecast1 = evaluate_fitness(genome, X[:train_end], X[test_start:])
    #     forecast1 = evaluate_fitness(genome)
    #     forecasts.append(forecast1)
    
    fitness_scores = [evaluate_fitness(genome) for genome in genome_population]
    forecasts.append(fitness_scores)
    
    selected_genomes = select_best_genomes(genome_population, fitness_scores)
    genome_population = reproduce(selected_genomes, population_size, mutation_probability, num_input_nodes, num_output_nodes, num_elites)
            
    
    valid_scores = [score for score in fitness_scores if score is not None]

    if valid_scores:
        # Before attempting to find the best scores, check if fitness_scores is empty
        if fitness_scores:
            best_rmse = min(valid_scores, key=lambda x: x['rmse'])['rmse']
            best_mae = min(valid_scores, key=lambda x: x['mae'])['mae']
            best_mse = min(valid_scores, key=lambda x: x['mse'])['mse']
            
            best_rmse_scores.append(best_rmse)
            best_mae_scores.append(best_mae)
            best_mse_scores.append(best_mse)
            
            best_fitness_scores.append(best_rmse)
    

            ###generation_best_score = min(fitness_scores, key=lambda x: x['rmse'])['rmse']
            # Assuming fitness_scores is a list of dictionaries or None
            ####fitness_scores = [score for score in fitness_scores if score is not None]
            ####if fitness_scores:
            ####    generation_best_score = min(fitness_scores, key=lambda x: x['rmse'])['rmse']
            ####else:
                # Handle the case where all fitness scores are None
                # For example, you might choose to log this situation and skip this generation
            ####    continue

            ####best_fitness_scores.append(generation_best_score)
            
            ####best_rmse = min(fitness_scores, key=lambda x: x['rmse'])['rmse']
            ####best_mae = min(fitness_scores, key=lambda x: x['mae'])['mae']
            ####best_mse = min(fitness_scores, key=lambda x: x['mse'])['mse']
            
            # Check for early stopping if the best score doesn't improve
            # if best_rmse >= previous_best_score:
            #     stagnation_counter += 1
            # else:
            #     stagnation_counter = 0
            #     previous_best_score = best_rmse
            
            # if stagnation_counter >= 10:  # stop if no improvement in 10 generations
            #     print(f"Early stopping at generation {generation} due to lack of improvement.")
            #     break
            
            # Early stopping logic
            # if fitness_scores and fitness_scores[0]['rmse'] < previous_best_score:
            #     previous_best_score = fitness_scores[0]['rmse']
            #     stagnation_counter = 0
            # else:
            #     stagnation_counter += 1

            # if stagnation_counter >= stagnation_threshold:
            #     print(f"\nEarly stopping at generation {generation} due to lack of improvement.")
            #     break
            
                    
            # Logic to increase timestep
            # if should_increase_timestep(generation, best_fitness_scores, stagnation_threshold, increment_interval):
            #     current_timestep += timestep_increment
            #     print(f"Increasing timestep to {current_timestep}")
            #     X = datasets.mackey_glass(n_timesteps=current_timestep + n_timesteps, sample_len=2000)  # Regenerate dataset with new timestep

            #     train_end = int(len(X) * 0.7)  # Recalculate the index for the end of the training data
            #     test_start = train_end + 1  # Recalculate the index for the start of the test data
                
            #if should_increase_timestep(generation, best_fitness_scores, 10, 5):
            #    n_timesteps += 1  # Increase complexity
            #    X = datasets.mackey_glass(n_timesteps=n_timesteps, sample_len=n_timesteps)
            #    train_end = int(len(X) * 0.7)
            #    test_start = train_end + 1
            
            
            
            ####best_rmse_scores.append(best_rmse)
            ####best_mae_scores.append(best_mae)
            ####best_mse_scores.append(best_mse)
            
            #best_fitness_scores.append(best_rmse)

            #print(f"Generation {generation}: Best Fitness Score: {generation_best_score}")
            
            
            
            
            
            
            # Instead of visualize_genome(selected_genomes), use:
            #best_genome = select_best_genomes(genome_population, fitness_scores, 1)[0]  # Get the best genome
            #visualize_genome(best_genome)  # Visualize the best genome
            
            

            """ # Calculate and record the population's genetic diversity for this generation
            current_diversity_score = calculate_diversity_score(genome_population)
            diversity_scores.append(current_diversity_score)
            print(f"Generation {generation}: Diversity Score: {current_diversity_score}")

            
            # Update the call to 'reproduce' to include node counts
            #new_population = reproduce(selected_genomes, population_size, mutation_probability, num_input_nodes, num_output_nodes, num_elites)
            #genome_population = new_population
            
            if should_increase_timestep(generation, best_fitness_scores, stagnation_threshold, increment_interval):
                current_timestep += 1
                print(f"Increasing timestep to {current_timestep}")
                X = datasets.mackey_glass(n_timesteps=current_timestep, sample_len=2000)
                
                # Check for NaN values in the dataset
                if np.any(np.isnan(X)):
                    print("Warning: NaN values found in the dataset. Attempting to handle them.")
                    # Handle NaN values here. For example, you might choose to fill them with a specific value,
                    # interpolate, or drop the rows/columns containing NaNs depending on your requirement.
                    # Here's an example to fill NaNs with 0, which might not be the best approach for your case:
                    X = np.nan_to_num(X, nan=0.0)
                else:
                    print("No NaN values found in the dataset.")
    
                train_end = int(len(X) * 0.7)
                test_start = train_end + 1
            
            # Apply reproduction with the new mutation function
            # Mutate and create the next population
            diversity_threshold = 0.1  # Define a suitable threshold
            if calculate_diversity_score(genome_population) < diversity_threshold:
                mutation_probability = adjust_mutation_rate(fitness_scores)
            #genome_population = [hox_like_mutation(genome, mutation_probability) for genome in selected_genomes]
            ###genome_population = reproduce(selected_genomes, population_size, mutation_probability, num_input_nodes, num_output_nodes, num_elites)
            
            ####genome_population = [genome for genome in genome_population if validate_genome(genome)]
            
            # Select a genome for visualization
            #genome_for_visualization = copy.deepcopy(random.choice(genome_population))
            #visualize_genome(genome_for_visualization)  # Visualize before mutation

            # Mutate and visualize
            #mutated_genome = reproduce([genome_for_visualization], 1, mutation_probability, num_input_nodes, num_output_nodes, 0)[0]
            #visualize_genome(mutated_genome)  # Visualize after mutation
"""
            


            # Correct file path for logging
            #log_path = "/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt"
            log_path = "evolution_log.txt"
            log_generation_results(generation, selected_genomes, fitness_scores, diversity_scores, log_file=log_path)

        else:
            # Handle the case when fitness_scores is empty
            print("Warning: No valid fitness scores to select from.")

# Perform post-experiment analysis with the correct file path
#analyze_log_path = "/Users/chaitravshetty/Downloads/Advanced-Genomes-for-Evolutionary-Computing-main 3/hierarchical_genomes/evolution_log.txt"
analyze_log_path = "evolution_log.txt"
# Flatten the list of forecasts to only contain arrays and not None values
forecasts = [f['forecast'] for f in fitness_scores if f is not None and 'forecast' in f]
analyze_results(X, test_start, forecasts,log_file=analyze_log_path, best_rmse_scores=best_rmse_scores, best_mae_scores=best_mae_scores, best_mse_scores=best_mse_scores)
#plot_actual_vs_predicted(X, test_start, forecasts)
#plot_fitness_scores(generation,best_rmse_scores=best_rmse_scores, best_mae_scores=best_mae_scores, best_mse_scores=best_mse_scores)


