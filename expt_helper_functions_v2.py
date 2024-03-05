import numpy as np
import copy
import random

def create_initial_genome(input_size=10, output_size=10, initial_connections=20):
    genome = []
    for _ in range(initial_connections):
        input_node = random.randint(0, input_size - 1)
        output_node = random.randint(0, output_size - 1)
        weight = random.uniform(-1.0, 1.0)
        genome.append([input_node, output_node, weight])
    return genome

def mutate_hox(genome, mutation_rate):
    def apply_mutation(gene):
        if np.random.rand() < mutation_rate:
            gene[2] += np.random.uniform(-0.1, 0.1)
        return gene

    mutated_genome = []
    for i in range(len(genome)):
        if isinstance(genome[i], list) and isinstance(genome[i][0], list):  # Handling nested list
            nested_mutation = []
            for nested_gene in genome[i]:
                nested_mutation.append(apply_mutation(nested_gene))
            mutated_genome.append(nested_mutation)
        elif isinstance(genome[i], list):
            mutated_genome.append(apply_mutation(genome[i]))
    return mutated_genome


def transcribe_hierarchical_genome_to_weight_matrix(genome):
    size = max(genome, key=lambda gene: max(gene[0], gene[1]))[1] + 1  # Find the largest node index
    W = np.zeros((size, size))  # Create a square matrix of zeros
    for gene in genome:
        input_node, output_node, weight = gene
        W[input_node, output_node] = weight  # Assign weights to the matrix
    return W


def mae(prediction, target):
    return np.mean(np.abs(prediction - target))

def mse(prediction, target):
    return np.mean((prediction - target)**2)

# Other necessary functions can be added here
