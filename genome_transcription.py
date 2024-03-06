import numpy as np
from helper_functions import find_max_postive_int_in_nested_list, find_connection_genes

def transcribe_hierarchical_genome_to_weight_matrix_old(genome):
    """
    Transcribes the genome to a weight matrix.
    """

    # Finds all connection genes in the genome
    # i.e., all genes that connect two nodes and the weight between them
    connection_genes = find_connection_genes(genome)

    # Finds the largest node number to determine the size of the weight matrix
    max_node_nr = find_max_postive_int_in_nested_list(connection_genes)

    # Initialize weight matrix
    weight_matrix = np.zeros((max_node_nr + 1, max_node_nr + 1))
    
    # Add the weights to the weight matrix
    for connection_gene in connection_genes:
        out_node = int(connection_gene[0])  # Ensure node numbers are integers
        in_node = int(connection_gene[1])
        weight = connection_gene[2]
        
        weight_matrix[out_node, in_node] = weight

    return weight_matrix



def transcribe_hierarchical_genome_to_weight_matrix_updated(genome):
    """
    Transcribes the genome to a weight matrix, ensuring all inputs are valid and handling potential errors.
    """
    try:
        # Validate the genome's structure
        if not all(isinstance(gene, (list, tuple)) and len(gene) == 3 for gene in genome):
            raise ValueError("Invalid genome structure")

        # Extract connection genes ensuring all are unique if necessary
        connection_genes = {tuple(gene[:2]): gene[2] for gene in genome}

        # Finds the largest node number to determine the size of the weight matrix
        max_node_nr = max(max(gene[0], gene[1]) for gene in connection_genes)

        # Initialize a weight matrix with zeros
        weight_matrix = np.zeros((max_node_nr + 1, max_node_nr + 1))

        # Fill the weight matrix with weights from connection genes
        for (out_node, in_node), weight in connection_genes.items():
            weight_matrix[int(out_node), int(in_node)] = weight

        return weight_matrix

    except Exception as e:
        # Handle potential errors in transcription
        print(f"An error occurred during genome transcription: {e}")
        # Depending on the context, you might want to re-raise the error or return a default value
        raise


def transcribe_hierarchical_genome_to_weight_matrix_old1(genome):
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
    """
    Transcribes a hierarchical genome into a weight matrix for an ESN.
    
    Parameters:
    - genome: The hierarchical genome, expected to be a list of connection genes.
              Each connection gene should be a list or tuple: [out_node, in_node, weight].
              
    Returns:
    - weight_matrix: A square matrix representing the connections and weights between nodes.
    """
    # Validate genome structure and extract connection genes
    #connection_genes = [g for g in flatten_genome(genome) if isinstance(g, (list, tuple)) and len(g) == 3]
    connection_genes = [g for g in genome if isinstance(g, (list, tuple)) and len(g) == 3]
    
    if not connection_genes:
        raise ValueError("No valid connection genes found in the genome.")
    
    # Determine the size of the weight matrix
    max_node = max(max(g[0], g[1]) for g in connection_genes)
    weight_matrix = np.zeros((max_node + 1, max_node + 1))
    
    # Populate the weight matrix
    for out_node, in_node, weight in connection_genes:
        weight_matrix[out_node, in_node] = weight
    
    return weight_matrix

def flatten_genome(genome):
    """
    Flattens a hierarchical genome structure into a list of connection genes.
    
    Parameters:
    - genome: The hierarchical genome structure.
    
    Returns:
    - A flattened list of connection genes.
    """
    if isinstance(genome, (list, tuple)):
        return [item for sublist in genome for item in flatten_genome(sublist)]
    else:
        return [genome]

