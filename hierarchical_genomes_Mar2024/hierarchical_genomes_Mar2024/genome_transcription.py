import numpy as np
from .helper_functions import find_max_postive_int_in_nested_list, find_connection_genes



def transcribe_hierarchical_genome_to_weight_matrix_old(genome):
    """
    Transcribes the genome to a weight matrix.
    """
    
    # Finds all connection genes in the genome
    # i.e all genes that connect two nodes and the weight between them
    connection_genes = find_connection_genes(genome)

    # Finds the largest node number to determine size of weight matrix
    max_node_nr = find_max_postive_int_in_nested_list(connection_genes)
    
    # Initialize weight matrix
    weight_matrix = np.zeros((max_node_nr + 1, max_node_nr + 1))
    
    # Add the weights to the weight matrix
    for connection_gene in connection_genes:

        out_node = connection_gene[0]
        in_node = connection_gene[1]
        weight = connection_gene[2]
        
        weight_matrix[out_node, in_node] = weight


    return weight_matrix


def transcribe_hierarchical_genome_to_weight_matrix(genome):
    def is_valid_gene(gene):
        return len(gene) == 3 and all(isinstance(gene[i], int) for i in range(2)) and isinstance(gene[2], float)
    
    valid_connection_genes = [gene for gene in genome if is_valid_gene(gene)]

    if not valid_connection_genes:
        return None  # Return None to indicate an issue with the genome structure

    max_node = find_max_postive_int_in_nested_list(genome) + 1
    weight_matrix = np.zeros((max_node, max_node))

    for gene in valid_connection_genes:
        input_node, output_node, weight = gene
        weight_matrix[input_node][output_node] = weight

    return weight_matrix


