import random 
import numpy as np
from helper_functions import find_max_postive_int_in_nested_list



### Addition and removal of genes (nodes)

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
    
def add_node(genome, max_node_nr, mutation_probability):
    random_variable = np.random.uniform(0,1,1)
    if random_variable < mutation_probability:
        # Ensure there's always at least one node to connect to, hence max_node_nr + 2
        out_node = random.choice(range(max_node_nr + 1)) # Adjusted from max_node_nr to max_node_nr + 1
        new_node_connection = [out_node, max_node_nr + 1, float(np.random.uniform(-0.1,0.1,1)[0])]
        genome.append(new_node_connection)
        return True
    return False


def mutation_add_node_old(genome, mutation_probability):
    
    mutated = False
    max_node_nr = find_max_postive_int_in_nested_list(genome)
    
    while mutated == False:
        mutated = add_node(genome, max_node_nr, mutation_probability)
    
    return mutated

def mutation_add_node(genome, mutation_probability):
    mutated = False
    
    max_node_nr = find_max_postive_int_in_nested_list(genome) + 1  # Adjusted to avoid IndexError
    if max_node_nr <= 1:  # Prevent mutation if insufficient nodes
        return genome
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
    # Convert genome to a list if it's a tuple
    if isinstance(genome, tuple):
        genome = list(genome)
    
    # Check if the current genome represents a direct connection
    if isinstance(genome, list) and len(genome) == 3 and isinstance(genome[0], int) and isinstance(genome[1], int):
        # Apply mutation directly to this connection
        random_variable = np.random.uniform(0, 1)
        if random_variable < 0.5:
            genome[0] = np.random.randint(0, max_node_nr + 1)
        else:
            genome[1] = np.random.randint(0, max_node_nr + 1)
        return genome

    # Recursive case: process sub-genomes
    elif isinstance(genome, list):
        # Check each element of the list
        for i in range(len(genome)):
            sub_genome = genome[i]
            # Check if sub_genome is a direct connection or a list that could represent connections
            if isinstance(sub_genome, (list, tuple)) and len(sub_genome) > 0:
                if isinstance(sub_genome, tuple):
                    # If it's a tuple, convert to list for mutation
                    sub_genome = list(sub_genome)
                # Recurse or mutate directly based on the sub_genome type
                genome[i] = mutate_connection(sub_genome, max_node_nr)
        return genome

    # In case genome is not a list or a direct connection, return it unchanged
    return genome





def mutate_connection_old(genome, max_node_nr):
    # Ensure genome is in a mutable form
    if isinstance(genome, tuple):
        genome = list(genome)
        
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
        
    #return genome
    # Convert back to tuple if needed
    return list(genome)


def mutate_weight(genome, mutation_value):
    # The mutation_value is received as an array, extract its first element to apply
    mutation_value = mutation_value[0]  # Assuming mutation_value is a numpy array
    
    # Check if genome is directly a connection gene
    if isinstance(genome, list) and len(genome) == 3 and all(isinstance(item, (int, float)) for item in genome[:2]):
        # Directly mutate the weight of the connection gene
        genome[2] += mutation_value
        return genome
    elif isinstance(genome, list):
        # Recursively apply mutation to each item in the genome
        for i in range(len(genome)):
            if isinstance(genome[i], list):
                genome[i] = mutate_weight(genome[i], [mutation_value])  # Pass mutation_value as array
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
 




        


