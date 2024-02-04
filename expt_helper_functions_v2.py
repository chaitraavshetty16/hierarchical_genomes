import numpy as np
import random
import struct
import matplotlib.pyplot as plt
import pandas as pd
import copy

def mae(prediction, target):
    return np.mean(np.abs(prediction - target))

def mse(prediction, target):
    return np.mean((prediction - target)**2)

def create_initial_genome(input_size=10, output_size=10, initial_connections=20):
    genome = []
    for _ in range(initial_connections):
        input_node = np.random.randint(0, input_size)
        output_node = np.random.randint(0, output_size)
        weight = np.random.uniform(-1.0, 1.0)
        genome.append([input_node, output_node, weight])
    return genome

def select_best_genomes(genome_population, fitness_scores, num_to_select=5, elitism_factor=0.1):
    num_elites = int(elitism_factor * len(genome_population))
    scored_genomes = sorted(zip(genome_population, fitness_scores), key=lambda x: x[1]['rmse'])
    return [genome for genome, score in scored_genomes[:num_to_select]]

def crossover(parent1, parent2):
    # Choose crossover point, should be within the range of the genomes' length
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    # Create children by combining the parents' genes
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(genome, mutation_rate, num_input_nodes, num_output_nodes):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] = mutate_gene(genome[i], num_input_nodes, num_output_nodes)
    return genome

def mutate_hox(genome, mutation_rate):
    def apply_mutation(subgenome):
        if isinstance(subgenome, list):
            mutated_subgenome = [apply_mutation(item) if isinstance(item, list) else item for item in subgenome]
            return mutated_subgenome
        elif random.random() < mutation_rate:
            return subgenome + np.random.uniform(-0.1, 0.1)
        return subgenome
    return apply_mutation(genome)

def reproduce(selected_genomes, population_size, mutation_rate, num_input_nodes, num_output_nodes, num_elites):
    new_population = selected_genomes[:num_elites]
    # Ensure there is at least one genome to select from, avoiding IndexError
    if len(selected_genomes) > num_elites:
        while len(new_population) < population_size:
            # Safely select a parent from the available non-elite genomes
            parent = random.choice(selected_genomes[num_elites:])
            # Optionally mutate the parent or use mutate_hox directly on it
            child = mutate_hox(random.choice([parent, mutate(copy.deepcopy(parent), mutation_rate, num_input_nodes, num_output_nodes)]), mutation_rate)
            new_population.append(child)
    return new_population[:population_size]

def mutate_gene(gene, num_input_nodes, num_output_nodes):
    mutation_type = random.choice(["weight_mutation", "structural_mutation"])
    if mutation_type == "weight_mutation":
        gene[2] += np.random.uniform(-0.5, 0.5)
    else:
        gene[0] = random.randint(0, num_input_nodes - 1)
        gene[1] = random.randint(0, num_output_nodes - 1)
    return gene

def calculate_diversity_score(genome_population):
    diversity_scores = []
    for i, genome1 in enumerate(genome_population):
        for j, genome2 in enumerate(genome_population):
            if i < j:
                diversity_scores.append(hamming_distance(genome_to_binary(genome1), genome_to_binary(genome2)))
    return np.mean(diversity_scores) if diversity_scores else 0

def genome_to_binary(genome):
    return ''.join(format(value, '08b') if isinstance(value, int) else float_to_binary(value) for gene in genome for value in gene)

def float_to_binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def should_increase_timestep(current_generation, best_fitness_scores, stagnation_threshold, increment_interval):
    if current_generation % increment_interval != 0:
        return False
    if len(best_fitness_scores) < stagnation_threshold:
        return False
    if all(score >= best_fitness_scores[-stagnation_threshold] for score in best_fitness_scores[-stagnation_threshold:]):
        return True
    return False

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
