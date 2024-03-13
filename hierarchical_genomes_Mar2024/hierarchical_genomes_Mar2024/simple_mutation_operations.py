import random 
import numpy as np
from .helper_functions import find_max_postive_int_in_nested_list



### Addition and removal of genes (nodes)


def add_node(genome, max_node_nr, mutation_probability):

    # Check if the current genome element is a gene (tuple).
    if isinstance(genome, tuple) and len(genome) == 3:
        return False  # Base case: cannot add a node directly to a gene, so return False.

    mutated = False
    # Iterate over each sub-genome (or sub-list) and attempt to add a node.
    for i in range(len(genome)):
        subgenome = genome[i]
        if isinstance(subgenome, list):  # Ensure subgenome is a list before attempting to mutate.
            mutation_result = add_node(subgenome, max_node_nr, mutation_probability)
            if mutation_result:  # If mutation was successful in a subgenome, set mutated to True.
                mutated = True
                break  # Stop iterating as we've successfully mutated one path in the genome.

    # If no mutation has occurred and we're allowed to add a new node at this level of the genome:
    if not mutated and random.random() < mutation_probability:
        # Construct a new connection with the new node.
        new_node_connection = [random.choice(range(max_node_nr + 1)), max_node_nr + 1, np.random.uniform(-0.1, 0.1)]
        genome.append(new_node_connection)  # Add the new connection to the genome.
        mutated = True  # Indicate that mutation has occurred.

    return mutated



def add_node_old(genome, max_node_nr, mutation_probability):
    '''
    This should go through a single random path to the leaf node of the genome
    and then walk back up again with some probability of a mutation happening at each
    backwards step. This is done to make the probability of mutating higher at the lower levels
    than the higher
    
    '''
    if type(genome[0]) == int and type(genome[1]) == int:
        return False
    
    
    target_sub_genome = random.choice(genome)
    mutated = add_node(target_sub_genome, max_node_nr, mutation_probability)
    
    if mutated == False:
        random_variable = np.random.uniform(0,1,1)
        if random_variable < mutation_probability:
            # to do: make bi directional by 50-50 prob
            out_node = random.choice(range(max_node_nr))
            new_node_connection = [out_node, max_node_nr + 1, float(np.random.uniform(-0.1,0.1,1)[0])]
            genome.append(new_node_connection)
            return genome
        else:
            return False
    else:
        return genome
    

def mutation_add_node(genome, mutation_probability):
    
    mutated = False
    max_node_nr = find_max_postive_int_in_nested_list(genome)
    
    while mutated == False:
        mutated = add_node(genome, max_node_nr, mutation_probability)
    
    return mutated

    
def remove_node(genome):
    """
    Recursively chooses a random sub genome until it reaches a leaf node, then removes this node.
    i.e walks down a random path through the genome to a gene and then removes it
    """
    if genome is not None and type(genome[0]) == int and type(genome[1]) == int :
        return True
    
    target_sub_genome = random.choice(genome)
    leaf_node = remove_node(target_sub_genome)
    if leaf_node == True:
        genome.remove(target_sub_genome)
        if len(genome) == 0:
            return True
    return genome

def mutation_remove_node(genome):
    """
    Removes a random gene (node) from the genome.
    """
    remove_node(genome)
    return genome



## Connection Mutations
############################################################################################################

def mutate_connection(genome, max_node_nr):
    # Check if the genome is a gene (a leaf node in the context of a genome structure).
    if isinstance(genome, tuple) and len(genome) == 3:
        # Perform the mutation logic here. For example:
        random_variable = np.random.uniform(0, 1)
        if random_variable < 0.5:
            new_input_node = random.choice(range(max_node_nr + 1))
            # Create a new tuple for the mutated gene.
            return (new_input_node, genome[1], genome[2])
        else:
            new_output_node = random.choice(range(max_node_nr + 1))
            # Create a new tuple for the mutated gene.
            return (genome[0], new_output_node, genome[2])
    elif isinstance(genome, list):  # If the genome is a list of genes.
        # Create a new list to hold the mutated genes.
        mutated_genome = []
        for subgenome in genome:
            # Recursively apply the mutation to each subgenome.
            mutated_subgenome = mutate_connection(subgenome, max_node_nr)
            mutated_genome.append(mutated_subgenome)
        return mutated_genome
    else:
        # Return the genome as is if it's not a gene or list of genes.
        return genome




def mutate_connection_old(genome, max_node_nr):
    if type(genome[0]) == int and type(genome[1]) == int:
        random_variable = np.random.uniform(0,1,1)
        if random_variable < 0.5:
            # ToDo: should it be max node nr + 1? Otherwise it can't connect to the largest node nr
            genome[0] = random.choice(range(max_node_nr + 1))
        else:
            genome[1] = random.choice(range(max_node_nr + 1))
        return genome
    else:
        target_sub_genome = random.choice(genome)
        mutate_connection(target_sub_genome, max_node_nr)
        
    return genome

def mutate_weight(genome, mutation_value):
    # Base case: If the genome is a gene (a tuple of two ints and a float),
    # create a new tuple with the mutated weight.
    if isinstance(genome, tuple) and len(genome) == 3 and isinstance(genome[2], float):
        input_node, output_node, weight = genome
        # Apply the mutation value to the weight.
        mutated_weight = weight + mutation_value
        # Return a new tuple representing the mutated gene.
        return (input_node, output_node, mutated_weight)

    # Recursive case: If the genome is a list, apply the mutation to each element.
    elif isinstance(genome, list):
        return [mutate_weight(gene, mutation_value) for gene in genome]

    # If the genome is neither a tuple representing a gene nor a list of genes,
    # return it unchanged.
    else:
        return genome



def mutate_weight_old(genome, mutation_value):
    """
    Takes a genome and a value by wich to mutate the weight
    Recursively chooses a random sub genome until it reaches a leaf node, then adds the mutation value to the weight at this gene.
    """
    if type(genome[0]) == int and type(genome[1]) == int:
        genome[2] += mutation_value
        genome[2] = float(genome[2])
        return genome
    else:
        target_sub_genome = random.choice(genome)
        mutate_weight(target_sub_genome, mutation_value)
    return genome


def add_connection(genome, max_node_nr, mutation_probability):
    """
    Adds new connection between two nodes in the genome. Can add connection to a new node.
    Adds the new gene to the top level of the genome. (ToDO: should it be added to a random sub genome?)
    """
    if type(genome[0]) == int and type(genome[1]) == int:
        return False
    
    
    target_sub_genome = random.choice(genome)
    mutated = add_node(target_sub_genome, max_node_nr, mutation_probability)
    
    if mutated == False:
        random_variable = np.random.uniform(0,1,1)
        if random_variable < mutation_probability:
            # to do: make bi directional by 50-50 prob
            out_node = random.choice(range(max_node_nr))
            in_node = random.choice(range(max_node_nr))
            new_node_connection = [out_node, in_node, float(np.random.uniform(-0.1,0.1,1)[0])]
            genome.append(new_node_connection)
            return genome
        else:
            return False
    else:
        return genome
    
def mutation_add_connection(genome, mutation_probability):


    mutated = False
    max_node_nr = find_max_postive_int_in_nested_list(genome)
    
    while mutated == False:
        mutated = add_connection(genome, max_node_nr, mutation_probability)
    
    return mutated

def mutate_gene(gene, mutation_details):
    # Assuming gene is a tuple like (input_node, output_node, weight)
    # and mutation_details contains information necessary for the mutation
    if isinstance(gene, tuple):
        input_node, output_node, weight = gene
        # Apply some mutation logic to determine new_weight
        new_weight = weight + calculate_mutation(mutation_details)
        # Construct a new tuple with the mutated weight
        return (input_node, output_node, new_weight)
    return gene

def calculate_mutation(mutation_details):
    # Example: Calculating a weight mutation using a normal distribution
    mutation_value = np.random.normal(mutation_details['mean'], mutation_details['std'])
    return mutation_value

def apply_mutation_to_genome(genome, mutation_details):
    if isinstance(genome, list):  # For genomes structured as lists of genes
        return [apply_mutation_to_genome(gene, mutation_details) for gene in genome]
    elif isinstance(genome, tuple):
        return mutate_gene(genome, mutation_details)
    else:
        return genome





        


