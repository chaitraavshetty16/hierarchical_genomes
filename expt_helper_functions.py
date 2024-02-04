import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
import struct


mutation_probability = 0.1  # 10% chance of mutation for each gene

def mae(prediction, target):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(prediction - target))

def mse(prediction, target):
    """Calculate Mean Squared Error."""
    return np.mean((prediction - target)**2)

def create_initial_genome(input_size=10, output_size=10, initial_connections=20):
    """
    Create an initial random genome for the evolutionary process.

    Args:
    - input_size (int): The number of input nodes in the neural network.
    - output_size (int): The number of output nodes in the neural network.
    - initial_connections (int): The initial number of connections in the genome.

    Returns:
    - genome (list): A list of connection genes, where each gene is represented as [input_node, output_node, weight].
    """

    # Initialize an empty list to store the connection genes
    genome = []

    # Create a specified number of random connections
    for _ in range(initial_connections):
        # Randomly select input and output nodes for the connection
        input_node = np.random.randint(0, input_size)
        output_node = np.random.randint(0, output_size)

        # Generate a random weight for the connection
        weight = np.random.uniform(-1.0, 1.0)

        # Append the new connection gene to the genome
        genome.append([input_node, output_node, weight])

    return genome

def select_best_genomes(genome_population, fitness_scores, num_to_select=5, elitism_factor=0.1):
    """
    Select the best genomes based on fitness scores and include elitism.

    Args:
    - genome_population (list): The current population of genomes.
    - fitness_scores (list): The fitness scores corresponding to the genomes.
    - num_to_select (int): The number of top genomes to select including elite genomes.
    - elitism_factor (float): The proportion of the population to carry over as elites.

    Returns:
    - selected_genomes (list): The top-performing genomes based on fitness scores.
    """
    
    num_elites = int(elitism_factor * len(genome_population))
    scored_genomes = list(zip(genome_population, fitness_scores))
    #scored_genomes.sort(key=lambda x: x[1])
    
    # Sort based on the 'rmse' value of the fitness score dictionaries
    #scored_genomes.sort(key=lambda x: x[1]['rmse'])

    # Elitism: directly carry over a proportion of the best genomes
    #elite_genomes = [genome for genome, score in scored_genomes[:num_elites]]

    # Selection: select additional genomes based on fitness scores
    #selected_genomes = elite_genomes + [genome for genome, score in scored_genomes[num_elites:num_to_select+num_elites]]

    #return selected_genomes

    scored_genomes = sorted(zip(genome_population, fitness_scores), key=lambda x: x[1]['rmse'])
    return [genome for genome, _ in scored_genomes[:num_to_select]]


def crossover(parent1, parent2):
    """ Perform a single point crossover between two genomes """
    # Choose crossover point, should be within the range of the genomes' length
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    # Create children by combining the parents' genes
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(genome, mutation_rate, num_input_nodes, num_output_nodes):
    """ Randomly mutate parts of the genome based on the mutation rate """
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            # Pass the additional arguments to the mutate_gene function
            genome[i] = mutate_gene(genome[i], num_input_nodes, num_output_nodes)
    return genome

def mutate_hox(genome, mutation_rate):
    """
    Apply mutations to a genome using a nested list structure similar to Hox gene mutations.
    """
    def apply_mutation(subgenome):
        # Base case: if subgenome is a connection gene, apply mutation
        #if isinstance(subgenome, list) and all(isinstance(item, int) or isinstance(item, float) for item in subgenome):
        if isinstance(subgenome, list) and all(isinstance(item, (int, float)) for item in subgenome):
            if random.random() < mutation_rate:
                # Example mutation: small perturbation in the weight
                weight_index = 2  # Assuming the weight is the third item in the gene
                subgenome[weight_index] += np.random.uniform(-0.1, 0.1)
            return subgenome

        # Recursive case: apply mutation to sublists
        mutated_subgenome = []
        for item in subgenome:
            if isinstance(item, list):
                mutated_subgenome.append(apply_mutation(item))
            else:
                mutated_subgenome.append(item)
        return mutated_subgenome

    # Start the recursive mutation process
    return apply_mutation(genome)

def reproduce(selected_genomes, population_size, mutation_rate, num_input_nodes, num_output_nodes, num_elites):
    """
    Create a new generation of genomes from the selected ones, preserving elites and using mutation only (no crossover).

    Args:
    - selected_genomes (list): Genomes selected based on fitness, including elites.
    - population_size (int): The size of the population to maintain.
    - mutation_rate (float): The probability of mutating a gene.
    - num_input_nodes (int): The number of input nodes in the neural network.
    - num_output_nodes (int): The number of output nodes in the neural network.
    - num_elites (int): The number of elite genomes to carry over unchanged.

    Returns:
    - new_population (list): A new population of genomes.
    """

    # Start new population with the elite genomes unchanged
    new_population = selected_genomes[:num_elites]

    # Ensure there is at least one genome to select from, avoiding IndexError
    if len(selected_genomes) > num_elites:
        # Use a while loop to fill up the remaining spots in the population
        while len(new_population) < population_size:
            if len(selected_genomes) > num_elites:
                # Randomly select a parent genome from the non-elite genomes for cloning
                parent = random.choice(selected_genomes[num_elites:])
                
                # Clone and mutate the selected genome
                #child = mutate(copy.deepcopy(parent), mutation_rate, num_input_nodes, num_output_nodes)
                # Clone and apply Hox mutation to the selected genome
                #child = mutate_hox(copy.deepcopy(parent), mutation_probability)
                # Optionally mutate the parent or use mutate_hox directly on it
                child = mutate_hox(random.choice([parent, mutate(copy.deepcopy(parent), mutation_rate, num_input_nodes, num_output_nodes)]), mutation_rate)
                
                
                # Add the new child to the new population
                new_population.append(child)
                
                # Ensure the population does not exceed the desired size
                if len(new_population) > population_size:
                    new_population = new_population[:population_size]

    return new_population

def mutate_gene(gene, num_input_nodes, num_output_nodes):
    mutation_type = random.choice(["weight_mutation", "structural_mutation"])
    if mutation_type == "weight_mutation":
        gene[2] += np.random.uniform(-0.5, 0.5)
    else:
        gene[0], gene[1] = random.randint(0, num_input_nodes - 1), random.randint(0, num_output_nodes - 1)
    return gene

def mutate_gene1(gene, num_input_nodes, num_output_nodes):
    """
    Mutate a single gene.

    Args:
    - gene (list): A gene represented by [input_node, output_node, weight].
    - num_input_nodes (int): Total number of input nodes in the network.
    - num_output_nodes (int): Total number of output nodes in the network.

    Returns:
    - gene (list): The mutated gene.
    """
    mutation_type = random.choice(["weight_mutation", "structural_mutation"])
    
    if mutation_type == "weight_mutation":
        # Apply a random change to the weight
        weight_change = np.random.uniform(-0.5, 0.5)
        gene[2] += weight_change  # Assuming the weight is at index 2

    elif mutation_type == "structural_mutation":
        # Randomly choose new input and output nodes
        new_input_node = random.randint(0, num_input_nodes - 1)
        new_output_node = random.randint(0, num_output_nodes - 1)
        gene[0] = new_input_node
        gene[1] = new_output_node

    return gene


def log_generation_results(generation, selected_genomes, fitness_scores, log_file="evolution_log.txt"):
    """
    Log the results of each generation.

    Args:
    - generation (int): The current generation number.
    - selected_genomes (list): The genomes selected based on fitness.
    - fitness_scores (list): The fitness scores of the selected genomes.
    - log_file (str): File path to save the log.
    """

    # with open(log_file, "a") as file:
    #     file.write(f"Generation {generation}\n")
    #     file.write(f"Top Fitness Scores: {fitness_scores[:5]}\n")  # Logging top 5 fitness scores
    #     file.write("Selected Genome Structures:\n")
    #     for genome in selected_genomes:
    #         file.write(f"{genome}\n")
    #     file.write("\n")
    with open(log_file, "a") as file:
        file.write(f"Generation {generation}\n")
        file.write(f"Best RMSE Score: {min(fitness_scores, key=lambda x: x['rmse'])['rmse']}\n")
        file.write(f"Best MAE Score: {min(fitness_scores, key=lambda x: x['mae'])['mae']}\n")
        file.write("Selected Genome Structures:\n")
        for genome in selected_genomes:
            file.write(f"{genome}\n")
        file.write("\n")


def analyze_results(log_file, best_rmse_scores, best_mae_scores, best_mse_scores):
    """
    Analyze and plot the results from the evolutionary process.

    Args:
    - log_file (str): File path of the saved log.
    - best_rmse_scores (list): List of the best RMSE scores per generation.
    - best_mae_scores (list): List of the best MAE scores per generation.
    """
    # Read generations from the log file
    generations = []
    with open(log_file, "r") as file:
        for line in file:
            if line.startswith("Generation"):
                current_generation = int(line.strip().split(" ")[1])
                generations.append(current_generation)
    
    min_length = min(len(generations), len(best_rmse_scores), len(best_mae_scores), len(best_mse_scores))
    
    if min_length != len(generations):
        print("Warning: Mismatch in the number of generations and recorded scores.")
        
        generations = generations[:min_length]
        best_rmse_scores = best_rmse_scores[:min_length]
        best_mae_scores = best_mae_scores[:min_length]
        best_mse_scores = best_mse_scores[:min_length]
    

    # Print top RMSE and MAE scores per generation
    for i in range(min_length):
        if(i==0):
            print(f"\n") 
            print(f"Generation {generations[i]}: Best RMSE Score: {best_rmse_scores[i]}, Best MAE Score: {best_mae_scores[i]}, Best MSE Score: {best_mse_scores[i]}")
        else:
            print(f"Generation {generations[i]}: Best RMSE Score: {best_rmse_scores[i]}, Best MAE Score: {best_mae_scores[i]}, Best MSE Score: {best_mse_scores[i]}")

    # Perform statistical analysis
    mean_rmse = np.mean(best_rmse_scores)
    mean_mae = np.mean(best_mae_scores)
    mean_mse = np.mean(best_mse_scores)
    print(f"\n")
    print(f"Average top RMSE score: {mean_rmse}")
    print(f"Average top MAE score: {mean_mae}")
    print(f"Average top MSE score: {mean_mse}")

    # Plot fitness scores over generations
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_rmse_scores, label='Best RMSE per Generation')
    plt.plot(generations, best_mae_scores, label='Best MAE per Generation')
    plt.plot(generations, best_mse_scores, label='Best MSE per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Evolution of Fitness Scores Over Generations')
    plt.legend()
    plt.show()


# This is a modified version of the mutate function to incorporate nested list structure for mutations.
def mutate_hox(genome, mutation_rate):
    """
    Apply mutations to a genome using a nested list structure similar to Hox gene mutations.
    """
    def apply_mutation(subgenome):
        # Base case: if subgenome is a connection gene, apply mutation
        if isinstance(subgenome, list) and all(isinstance(item, int) or isinstance(item, float) for item in subgenome):
            if random.random() < mutation_rate:
                # Apply some mutation logic here
                pass  # Replace with actual mutation logic
            return subgenome

        # Recursive case: apply mutation to sublists
        mutated_subgenome = []
        for item in subgenome:
            if isinstance(item, list):
                mutated_subgenome.append(apply_mutation(item))
            else:
                mutated_subgenome.append(item)
        return mutated_subgenome

    # Start the recursive mutation process
    return apply_mutation(genome)

# def calculate_diversity_score(genome_population):
#     # A simple example using Hamming distance
#     diversity_scores = []
#     for i, genome1 in enumerate(genome_population):
#         for j, genome2 in enumerate(genome_population):
#             if i < j:
#                 diversity_scores.append(hamming_distance(genome1, genome2))
#     return np.mean(diversity_scores)

# More sophisticated diversity calculation
def calculate_diversity_score(genome_population):
    diversity_scores = []
    for i, genome1 in enumerate(genome_population):
        for j, genome2 in enumerate(genome_population):
            if i < j:
                # Convert genomes to binary strings or a suitable format for hamming distance
                binary_genome1 = genome_to_binary(genome1)
                binary_genome2 = genome_to_binary(genome2)
                diversity_scores.append(hamming_distance(binary_genome1, binary_genome2))
    return np.mean(diversity_scores)

def genome_to_binary(genome):
    """
    Convert genome to a binary string representation.
    Each gene is assumed to be a list with numeric values, which are converted to binary format.
    """
    binary_genome = ""
    for gene in genome:
        for value in gene:
            # Handle floating-point numbers
            if isinstance(value, float):
                # Convert float to binary representation
                binary_value = float_to_binary(value)
            else:
                # Convert integers directly to binary
                binary_value = format(value, 'b')
            binary_genome += binary_value
    return binary_genome

def float_to_binary(num):
    """Convert a float to a binary string."""
    # Represent the float as a binary string
    # Note: Adjust the precision as needed for your application
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def hamming_distance(genome1, genome2):
    # Assuming genome1 and genome2 are of the same length and are lists of integers
    return sum(g1 != g2 for g1, g2 in zip(genome1, genome2))

def should_increase_timestep(current_generation, best_fitness_scores, stagnation_threshold, increment_interval):
    """
    Determine whether to increase the timestep based on stagnation of fitness improvement.

    Args:
    - current_generation (int): The current generation number.
    - best_fitness_scores (list): List of best fitness scores from each generation.
    - stagnation_threshold (int): Number of generations to wait with no improvement before increasing timestep.
    - increment_interval (int): Interval at which to consider increasing timesteps.

    Returns:
    - (bool): True if it's time to increase timestep, False otherwise.
    """
    if current_generation % increment_interval != 0:
        # Only check for timestep increment at specified intervals
        return False
    
    # Check if there has been improvement in the last 'stagnation_threshold' generations
    recent_generations = best_fitness_scores[-stagnation_threshold:]
    if len(recent_generations) < stagnation_threshold:
        # Not enough data to make a decision
        return False

    # Check if there's been any improvement
    if all(score >= recent_generations[0] for score in recent_generations):
        # No improvement, so increase timestep
        return True

    return False
