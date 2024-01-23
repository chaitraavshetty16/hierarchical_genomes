import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

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

# def select_best_genomes(genome_population, fitness_scores, num_to_select=5):
#     """
#     Select the best genomes based on fitness scores.

#     Args:
#     - genome_population (list): The current population of genomes.
#     - fitness_scores (list): The fitness scores corresponding to the genomes.
#     - num_to_select (int): The number of top genomes to select.

#     Returns:
#     - selected_genomes (list): The top-performing genomes based on fitness scores.
#     """
    
#     # Pair each genome with its fitness score
#     scored_genomes = list(zip(genome_population, fitness_scores))
    
#     # Sort the list of tuples based on fitness scores in ascending order
#     # since lower scores may indicate better performance (e.g., lower error is better)
#     scored_genomes.sort(key=lambda x: x[1])
    
#     # Select the top-performing genomes
#     selected_genomes = [genome for genome, score in scored_genomes[:num_to_select]]
    
#     return selected_genomes

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
    scored_genomes.sort(key=lambda x: x[1])

    # Elitism: directly carry over a proportion of the best genomes
    elite_genomes = [genome for genome, score in scored_genomes[:num_elites]]

    # Selection: select additional genomes based on fitness scores
    selected_genomes = elite_genomes + [genome for genome, score in scored_genomes[num_elites:num_to_select+num_elites]]

    return selected_genomes


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


# def reproduce(selected_genomes, population_size, mutation_rate, num_input_nodes, num_output_nodes):
#     """
#     Create a new generation of genomes from the selected ones.

#     Args:
#     - selected_genomes (list): Genomes selected based on fitness.
#     - population_size (int): The size of the population to maintain.
#     - mutation_rate (float): The probability of mutating a gene.

#     Returns:
#     - new_population (list): A new population of genomes.
#     """
#     new_population = []
#     while len(new_population) < population_size:
#         # Randomly select two parents from the selected genomes
#         parent1, parent2 = random.sample(selected_genomes, 2)
#         # Perform crossover to produce new children
#         child1, child2 = crossover(parent1, parent2)
#         # Mutate the children's genomes
        
#         child1 = mutate(child1, mutation_rate, num_input_nodes, num_output_nodes)
#         child2 = mutate(child2, mutation_rate, num_input_nodes, num_output_nodes)
    
#         # Add the new children to the new population
#         new_population.extend([child1, child2])
#     # If the population size is odd, remove one genome
#     if len(new_population) > population_size:
#         new_population.pop()
#     return new_population

def reproduce(selected_genomes, population_size, mutation_rate, num_input_nodes, num_output_nodes, num_elites):
    """
    Create a new generation of genomes from the selected ones, preserving elites.

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
    

        # Use a while loop to fill up the remaining spots in the population
    while len(new_population) < population_size:
        # Ensure that we don't pick elite genomes for reproduction
        non_elite_genomes = random.sample(selected_genomes[num_elites:], 2)
        parent1, parent2 = non_elite_genomes
        
        # Perform crossover to produce new children
        child1, child2 = crossover(parent1, parent2)
        
        # Mutate the children's genomes
        child1 = mutate(child1, mutation_rate, num_input_nodes, num_output_nodes)
        child2 = mutate(child2, mutation_rate, num_input_nodes, num_output_nodes)
    
        # Add the new children to the new population
        new_population.extend([child1, child2])
        
        # Ensure the population does not exceed the desired size
        if len(new_population) > population_size:
            new_population = new_population[:population_size]

    return new_population




def mutate_gene(gene, num_input_nodes, num_output_nodes):
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

    with open(log_file, "a") as file:
        file.write(f"Generation {generation}\n")
        file.write(f"Top Fitness Scores: {fitness_scores[:5]}\n")  # Logging top 5 fitness scores
        file.write("Selected Genome Structures:\n")
        for genome in selected_genomes:
            file.write(f"{genome}\n")
        file.write("\n")

def analyze_results(log_file="evolution_log.txt"):
    """
    Analyze the results from the evolutionary process.

    Args:
    - log_file (str): File path of the saved log.
    """

    generations = []
    fitness_scores = []
    with open(log_file, "r") as file:
        for line in file:
            if line.startswith("Generation"):
                current_generation = int(line.strip().split(" ")[1])
            if line.startswith("Top Fitness Scores:"):
                scores_str = line.strip().split(": ")[1]
                try:
                    scores = [float(score.strip()) for score in scores_str.strip('[]').split(",")]
                    top_score = min(scores)
                    generations.append(current_generation)
                    fitness_scores.append(top_score)
                except ValueError as e:
                    print(f"Error processing line: {line}. Error: {e}")

    # Perform statistical analysis
    mean_fitness = np.mean(fitness_scores)
    print(f"Average top fitness score across generations: {mean_fitness}")

    # Plot fitness scores over generations
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_scores, label='Top Fitness Score per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Evolution of Top Fitness Scores Over Generations')
    plt.legend()
    plt.show()
