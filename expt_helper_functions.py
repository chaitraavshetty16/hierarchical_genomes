import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
import struct
import networkx as nx
from numba import jit
#from heirarchical_genomes.evolution_step_functions import mutate_genome_without_hox 
from evolution_step_functions import *

@jit(nopython=True)
def calculate_fitness_numba(forecast, target):
    mse = ((forecast - target) ** 2).mean()
    return np.sqrt(mse)  # Return RMSE

mutation_probability = 0.1  # 10% chance of mutation for each gene

def mae(prediction, target):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(prediction - target))

def mse(prediction, target):
    """Calculate Mean Squared Error."""
    return np.mean((prediction - target)**2)

def create_initial_genome1(num_input_nodes=10, num_output_nodes=10):
    genome = {
        "nodes": [],
        "edges": [],
        "input_nodes": [],
        "output_nodes": [],
        "input_edges": [],
        "output_edges": [],
        "input_node_count": num_input_nodes,
        "output_node_count": num_output_nodes
    }

    # Add input nodes
    for i in range(num_input_nodes):
        genome["input_nodes"].append({"id": i, "type": "input"})

    # Add output nodes
    for i in range(num_output_nodes):
        genome["output_nodes"].append({"id": i + num_input_nodes, "type": "output"})

    return genome

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

def select_best_genomes_old(genome_population, fitness_scores, num_to_select=5, elitism_factor=0.1):
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

def select_best_genomes(genome_population, fitness_scores, num_to_select=5):
    selected_genomes = []
    # Filter out None values before proceeding
    fitness_scores = [fs for fs in fitness_scores if fs is not None]
    
    # Ensure we have enough genomes to sample from
    sample_size = min(len(fitness_scores), 3)
    
    for _ in range(num_to_select):
        if sample_size > 0:
            # Ensure 'rmse' key is present
            contenders = random.sample(
                [(genome, scores) for genome, scores in zip(genome_population, fitness_scores) if 'rmse' in scores],
                k=sample_size
            )
            # Use 'rmse' for comparison
            winner = min(contenders, key=lambda x: x[1]['rmse'])
            selected_genomes.append(winner[0])
        else:
            print("Not enough genomes with valid 'rmse' scores to select from.")
            break  # Exit the loop if there are no valid genomes to select from

    # Add novelty search solutions if applicable
    unique_solutions = novelty_search_ctrl_expt(genome_population, fitness_scores)
    selected_genomes.extend(unique_solutions[:num_to_select - len(selected_genomes)])

    return selected_genomes[:num_to_select]



def validate_genome(genome):
    valid = True
    seen_connections = set()
    for gene in genome:
        if (gene[0] == gene[1]) or ((gene[0], gene[1]) in seen_connections):
            valid = False
            break
        seen_connections.add((gene[0], gene[1]))
    return valid

def crossover(parent1, parent2):
    """ Perform a single point crossover between two genomes """
    # Choose crossover point, should be within the range of the genomes' length
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    # Create children by combining the parents' genes
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate_old(genome, mutation_rate, num_input_nodes, num_output_nodes):
    """ Randomly mutate parts of the genome based on the mutation rate """
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            # Pass the additional arguments to the mutate_gene function
            genome[i] = mutate_gene(genome[i], num_input_nodes, num_output_nodes)
    return genome

def mutate(genome, mutation_rate, num_input_nodes, num_output_nodes):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            gene = genome[i]
            # Introduce a larger mutation step size
            step_size = np.random.uniform(-1.0, 1.0)  
            gene[2] += step_size  # mutate the weight
            # Randomize connections for structural mutations
            if random.random() < 0.1:  # 10% chance for structural mutation
                gene[0] = random.randint(0, num_input_nodes - 1)
                gene[1] = random.randint(0, num_output_nodes - 1)
            genome[i] = gene
    return genome


def hox_like_mutation(genome, mutation_rate):
    """
    Apply Hox gene-like mutations to a genome with nested list structure.
    """
    def mutate_gene(gene):
        if random.random() < mutation_rate and isinstance(gene, list):
            # Introduce mutation
            weight_index = 2
            mutation_amount = np.random.uniform(-0.1, 0.1)
            gene[weight_index] += mutation_amount
            # Ensure weight remains within bounds
            gene[weight_index] = np.clip(gene[weight_index], -1, 1)
        return gene


    def mutate_module(module):
        # Apply a mutation to a module of genes
        # Example: Reverse the order of genes in the module
        return list(reversed(module))

    # Apply mutations with some probability
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            if isinstance(genome[i], list) and all(isinstance(gene, list) for gene in genome[i]):
                # Module-level mutation
                genome[i] = mutate_module(genome[i])
            elif isinstance(genome[i], list):
                # Gene-level mutation
                genome[i] = mutate_gene(genome[i])
            # Handle more mutation types if needed

    return genome

def mutate_hox_nested(genome, mutation_rate):
    def mutate_gene(gene):
        # Make sure the gene is a list before mutating
        # if isinstance(gene, list) and len(gene) >= 3:
        #     gene[2] += np.random.uniform(-0.1, 0.1)  # Assume gene[2] is the weight
        # return gene
        if random.random() < mutation_rate and isinstance(gene, list):
            return gene + random.choice(gene)
        return gene

    for i in range(len(genome)):
        if np.random.rand() < mutation_rate:
            if isinstance(genome[i], list):
                for j in range(len(genome[i])):
                    if isinstance(genome[i][j], list) and len(genome[i][j]) >= 3:
                        genome[i][j] = mutate_gene(genome[i][j])
            else:
                genome[i] = mutate_gene(genome[i])

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

def mutate_hox_new(genome, mutation_rate, num_input_nodes, num_output_nodes):
    # Example mutation: Adding a new node
    if random.random() < mutation_rate:
        new_node = generate_new_node(genome, num_input_nodes, num_output_nodes)  # function based on genome structure
        genome.append(new_node)
    return genome

def generate_new_node(genome, num_input_nodes, num_output_nodes):
    # Generate a new connection with random nodes and weight
    input_node = np.random.randint(0, num_input_nodes)
    output_node = np.random.randint(num_input_nodes, num_input_nodes + num_output_nodes)
    weight = np.random.uniform(-1.0, 1.0)
    new_gene = [input_node, output_node, weight]

    # Append the new gene to the genome
    genome.append(new_gene)
    return genome

def reproduce2(selected_genomes, population_size, mutation_rate, num_input_nodes, num_output_nodes, num_elites):
    new_population = selected_genomes[:num_elites]  # Elites unchanged

    while len(new_population) < population_size:
        parent = random.choice(selected_genomes)
        child = copy.deepcopy(parent)
        child = mutate_hox_nested(child, mutation_rate)
        new_population.append(child)

    return new_population[:population_size]

def reproduce1(selected_genomes, population_size, mutation_rate, num_input_nodes, num_output_nodes, num_elites):
    new_population = selected_genomes[:num_elites]  # Start with the elites

    # Check if there are non-elite genomes available for cloning
    non_elite_genomes = selected_genomes[num_elites:] if num_elites < len(selected_genomes) else []

    while len(new_population) < population_size:
        if non_elite_genomes:  # Ensure there are genomes to select from
            # Select a random genome to clone and mutate from non-elites
            parent = random.choice(non_elite_genomes)
            child = copy.deepcopy(parent)
            child = hox_like_mutation(child, mutation_rate)  # Apply Hox-like mutation
            new_population.append(child)
        else:
            # If no non-elites are available, clone and mutate from elites
            parent = random.choice(new_population)
            child = copy.deepcopy(parent)
            child = hox_like_mutation(child, mutation_rate)
            new_population.append(child)

    return new_population[:population_size]  # Ensure we don't exceed population size


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
                ###child = hox_like_mutation(random.choice([parent, mutate(copy.deepcopy(parent), mutation_rate, num_input_nodes, num_output_nodes)]), mutation_rate)
                child = mutate_genome_without_hox(copy.deepcopy(parent))
                #child = mutate_hox(copy.deepcopy(parent), mutation_rate)
                
                # Add the new child to the new population
                new_population.append(child)
                
                # Ensure the population does not exceed the desired size
                if len(new_population) > population_size:
                    new_population = new_population[:population_size]

    return new_population

def enhanced_mutate(genome, mutation_rate, num_input_nodes, num_output_nodes):
    for gene in genome:
        if random.random() < mutation_rate:
            # Perturb the weight with a small change
            gene[2] += np.random.uniform(-0.1, 0.1)  # Assuming index 2 is the weight

            # With a smaller chance, perform a structural mutation
            if random.random() < 0.1:  # 10% chance for a structural mutation
                gene[0] = random.randint(0, num_input_nodes - 1)  # Randomize the input node
                gene[1] = random.randint(0, num_output_nodes - 1)  # Randomize the output node

    return genome

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


def log_generation_results(generation, selected_genomes, fitness_scores, diversity_score, log_file="evolution_log.txt"):
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
        if fitness_scores:
            file.write(f"Best RMSE Score: {min(fitness_scores, key=lambda x: x['rmse'])['rmse']}\n")
            file.write(f"Best MAE Score: {min(fitness_scores, key=lambda x: x['mae'])['mae']}\n")
            file.write(f"Best MSE Score: {min(fitness_scores, key=lambda x: x['mse'])['mse']}\n")
        else:
            file.write("Log: No valid fitness scores available for this generation.\n")
        
        file.write("Selected Genome Structures:\n")
        for genome in selected_genomes:
            file.write(f"{genome}\n")
        file.write("\n")
        # Log diversity score
        file.write(f"Diversity Score: {diversity_score}\n")
        file.write("\n")


def analyze_results(X,test_start ,forecasts , log_file, best_rmse_scores, best_mae_scores, best_mse_scores):
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
    #plt.plot(generations, best_rmse_scores, label='Best RMSE per Generation')
    #plt.plot(generations, best_mae_scores, label='Best MAE per Generation')
    #plt.plot(generations, best_mse_scores, label='Best MSE per Generation')
    plt.plot(generations, best_rmse_scores, 'b-', label='Best RMSE per Generation')
    plt.plot(generations, best_mae_scores, 'r-', label='Best MAE per Generation')
    plt.plot(generations, best_mse_scores, 'g-', label='Best MSE per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Evolution of Fitness Scores Over Generations')
    plt.legend()
    plt.show()
    
    
def plot_actual_vs_predicted(X, test_start, forecasts):
    plt.figure(figsize=(15, 7))
    actual_series = X[test_start:].flatten()
    predicted_series = np.array(forecasts).flatten()
    time_steps = np.arange(len(actual_series))
    plt.plot(time_steps, actual_series, label='Actual Series', color='blue')
    plt.plot(time_steps, predicted_series, label='Predicted Series', color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Series Value')
    plt.title('Actual vs. Predicted Time Series')
    plt.show()



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
    # Calculate diversity based on continuous gene values
    diversity_scores = []
    for i, genome1 in enumerate(genome_population):
        for j, genome2 in enumerate(genome_population):
            if i < j:
                # Convert genomes to binary strings or a suitable format for hamming distance
                binary_genome1 = genome_to_binary(genome1)
                binary_genome2 = genome_to_binary(genome2)
                diversity_scores.append(hamming_distance(binary_genome1, binary_genome2))
    return np.mean(diversity_scores)

def calculate_diversity_score_new(genome_population):
    diversity_scores = []
    for i in range(len(genome_population)):
        for j in range(i + 1, len(genome_population)):
            diversity_score = np.sqrt(sum((g1 - g2)**2 for g1, g2 in zip(genome_population[i], genome_population[j])))
            diversity_scores.append(diversity_score)
    return np.mean(diversity_scores) if diversity_scores else 0


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

    
    #return False
    
    # Additional condition to increase complexity based on variance of fitness scores
    fitness_variance = np.var(best_fitness_scores[-stagnation_threshold:])
    if fitness_variance < 5: #some_threshold (1 to 5%):  # Define some_threshold based on experiment
        return True
    return False

    



def visualize_genome_old(genome):
    # Convert genome to NetworkX graph
    G = nx.Graph()
    for input_node, output_node, weight in genome:
        G.add_edge(input_node, output_node, weight=weight)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=700, font_size=15, 
            width=[(G[u][v]['weight']+1)/2 for u,v in G.edges()])
    plt.show()
    
def visualize_genome_old2(genome):
    G = nx.Graph()
    for gene in genome:
        # Check if gene structure is simple or complex
        if isinstance(gene[0], list) or isinstance(gene[1], list):
            # Handle complex gene structure here, e.g., by extracting node IDs or flattening
            input_node, output_node = str(gene[0]), str(gene[1])  # Convert to string or handle differently
        else:
            input_node, output_node = gene[0], gene[1]
        weight = gene[2]
        G.add_edge(input_node, output_node, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=15, width=[(G[u][v]['weight']+1)/2 for u,v in G.edges()])
    plt.show()

def visualize_genome(genome):
    G = nx.DiGraph()
    for module in genome:
        if isinstance(module, list):
            for gene in module:
                G.add_edge(gene[0], gene[1], weight=gene[2])
        else:
            G.add_edge(module[0], module[1], weight=module[2])
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=15, width=[(G[u][v]['weight']+1)/2 for u,v in G.edges()])
    plt.show()

#adjust the mutation rate dynamically
def adjust_mutation_rate(fitness_scores, threshold=0.01):
    # Filter out None values and ensure 'rmse' key is present
    rmse_scores = [fs['rmse'] for fs in fitness_scores if fs and 'rmse' in fs]
    std_fitness = np.std(rmse_scores) if rmse_scores else float('inf')
    
    if std_fitness < threshold:
        return min(1.0, mutation_probability * 1.1)  # Increase by 10%
    return mutation_probability


#introduce novelty search
def novelty_search(genome_population, fitness_scores, threshold=0.05):
    unique_solutions = []
    for genome, fitness in zip(genome_population, fitness_scores):
        if fitness is not None and 'rmse' in fitness:
            is_novel = all(abs(fitness['rmse'] - other_fitness.get('rmse', float('inf'))) > threshold for other_fitness in fitness_scores if other_fitness != fitness)
            if is_novel:
                unique_solutions.append(genome)
    return unique_solutions

def novelty_search_ctrl_expt(genome_population, fitness_scores, threshold=0.05):
    unique_solutions = []
    for i, (genome, fitness) in enumerate(zip(genome_population, fitness_scores)):
        if fitness is not None and 'rmse' in fitness:
            is_novel = True
            for j, other_fitness in enumerate(fitness_scores):
                if i != j and other_fitness is not None and 'rmse' in other_fitness:
                    # Ensure we're comparing single values, not arrays
                    difference = abs(fitness['rmse'] - other_fitness['rmse'])
                    if not isinstance(difference, np.ndarray):
                        if difference <= threshold:
                            is_novel = False
                            break
                    else:
                        raise ValueError("RMSE values are arrays, not single values. Check your fitness calculations.")
            if is_novel:
                unique_solutions.append(genome)
    return unique_solutions


def transcribe_hierarchical_genome_to_weight_matrix(genome):
    """
    Transcribes the genome to a weight matrix, ensuring no NaN values are introduced.
    """
    connection_genes = find_connection_genes(genome)
    max_node_nr = find_max_postive_int_in_nested_list(connection_genes)
    weight_matrix = np.zeros((max_node_nr + 1, max_node_nr + 1))

    for gene in connection_genes:
        if isinstance(gene, list) and len(gene) == 3:
            out_node, in_node, weight = gene
            if not np.isnan(weight):
                weight_matrix[out_node, in_node] = weight
            else:
                print("Warning: NaN weight encountered in connection gene.")
                return None  # Return None to indicate an error in the transcription
        else:
            print("Warning: Invalid connection gene format encountered.")
            return None  # Return None to indicate an error in the transcription

    if np.isnan(weight_matrix).any():
        print("Warning: NaN values found in the weight matrix after transcription.")
        return None  # Return None to indicate an error in the transcription

    return weight_matrix


def find_connection_genes(genome, connection_genes=None):
    """
    Recursively finds all connection genes in a genome and returns them in a list.
    """
    if connection_genes is None:
        connection_genes = []

    if isinstance(genome, list):
        if len(genome) == 3 and all(isinstance(item, (int, float)) for item in genome):
            # Base case: gene is a list of an output node, input node, and weight
            connection_genes.append(genome)
        else:
            for item in genome:
                find_connection_genes(item, connection_genes)
    else:
        print("Warning: Non-list item encountered in genome.")

    return connection_genes

def find_max_postive_int_in_nested_list(nested_list):
    """
    Finds the largest positive integer in a nested list.
    """
    max_value = 0
    if isinstance(nested_list, list):
        for item in nested_list:
            max_value = max(max_value, find_max_postive_int_in_nested_list(item))
    elif isinstance(nested_list, int) and nested_list > max_value:
        max_value = nested_list
    return max_value
