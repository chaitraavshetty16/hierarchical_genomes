
############################################################################################################
def add_value_to_int_in_nested_list(nested_list, value):
    """
    Finds all integers in a nested list and adds a value to them.
    """

    if type(nested_list) == int:
        return nested_list + value
    
    elif type(nested_list) == float:
        return nested_list
    
    elif type(nested_list) == list:
        l = []
        for sub_list in nested_list:
            r = add_value_to_int_in_nested_list(sub_list, value)
            l.append(r)

        return l

     
def add_symmetric_connection(connectivity_matrix, out_node, in_node):
    """
    Takes a connectivity matrix and adds a symmetric connection between two nodes.
    """
    connectivity_matrix[[out_node, in_node], [in_node, out_node]] = 1
                        
    return connectivity_matrix

    

# Functions that finds stuff from nested lists
############################################################################################################
def find_unique_ints_in_nested_list1(nested_list, out = set()):
    """
    Find all unique integers in a nested list.
    """
    if type(nested_list) == int:
        return {nested_list} 
    else:
        s = set()
        for sub_list in nested_list:
            s = s.union(find_unique_ints_in_nested_list(sub_list))
        return s 

def find_max_postive_int_in_nested_list_old(nested_list, maximum = 0):
    """
    Finds the largest positive integer in a nested list.
    """
    #print(type(nested_list), nested_list)
    if type(nested_list) == list:
        #print(nested_list)
        for sub_list in nested_list:
            sub_maximum = find_max_postive_int_in_nested_list(sub_list, maximum)
            #print(sub_maximum)
            if maximum < sub_maximum:
                maximum = sub_maximum
        return maximum 
    elif type(nested_list) == int:
        #print(nested_list)
        if nested_list > maximum:
            maximum = nested_list
        return maximum
    elif type(nested_list) == float:
        return maximum
    
def find_max_postive_int_in_nested_list(nested_list, maximum=0):
    if isinstance(nested_list, list):
        for sub_list in nested_list:
            sub_maximum = find_max_postive_int_in_nested_list(sub_list, maximum)
            maximum = max(maximum, sub_maximum if sub_maximum is not None else 0)
    elif isinstance(nested_list, (int, float)):
        maximum = max(maximum, nested_list)
    return maximum

def find_min_postive_int_in_nested_list(nested_list, minimum = 0):
    if isinstance(nested_list, list):
        for sub_list in nested_list:
            sub_minimum = find_min_postive_int_in_nested_list(sub_list, minimum)
            minimum = min(minimum, sub_minimum if sub_minimum is not None else 0)
    elif isinstance(nested_list, (int, float)):
        minimum = max(minimum, nested_list)
    return minimum

def find_min_postive_int_in_nested_list_old(nested_list, minimum = float("inf")):
    """
    Finds minimum positive integer in a nested list.
    """

    if type(nested_list) == list:
        #print(nested_list)
        for sub_list in nested_list:
            sub_min = find_min_postive_int_in_nested_list(sub_list, minimum)
            if sub_min < minimum:
                minimum = sub_min
        return minimum 
    elif type(nested_list) == int:
        #print(nested_list)
        if nested_list < minimum:
            minimum = nested_list
        return minimum
    elif type(nested_list) == float:
        return minimum
    

def find_connection_genes(genome):
    """
    Correctly finds and returns all connection genes within the genome.
    Assumes that connection genes are structured as lists: [out_node, in_node, weight].
    """
    connection_genes = []
    # Recursively search for connection genes within the genome structure
    if isinstance(genome, list):
        for element in genome:
            if isinstance(element, list) and len(element) == 3:  # Assuming a connection gene has exactly three elements
                connection_genes.append(element)
            elif isinstance(element, list):
                connection_genes.extend(find_connection_genes(element))
    return connection_genes


def find_connection_genes_old(genome):
    """
    Finds all connection genes in a genome.
    i.e identifies all genes that connect two nodes and the weight between them.
    and return them in a list.
    """
    if len(genome) == 0:
        del(genome)
        return 
    
    if type(genome[0]) == int and type(genome[1]) == int:
        gene = genome
        return gene
    else:
        genes = []
        for sub_genome in genome:
            gene = find_connection_genes(sub_genome)
            if type(gene[0]) == int and type(gene[1]) == int:
                genes.append(gene)
            else:
                genes.extend(gene)
        return genes
    
def find_connection_genes1(genome):
    """
    Finds all connection genes in a genome.
    i.e., identifies all genes that connect two nodes and the weight between them
    and returns them in a list.
    """
    genes = []
    if isinstance(genome, list):
        for sub_genome in genome:
            if isinstance(sub_genome, list):
                gene = find_connection_genes(sub_genome)
                if gene:  # Ensure gene is not None or empty
                    genes.extend(gene)  # Use extend to merge lists
            elif isinstance(sub_genome, (int, float)):
                # Handle the case where the sub_genome is directly a gene component
                genes.append(sub_genome)
    elif isinstance(genome, (int, float)):
        # Directly return the genome if it's a gene component (this case may need adjustment based on your genome structure)
        return [genome]

    return genes


def find_unique_ints_in_nested_list(nested_list):
    """
    Finds all unique integers in a nested list.
    """
    if not isinstance(nested_list, list) or not nested_list:
        return set()
    
    if len(nested_list) == 0:
        del (nested_list)
        return set()
    #print(nested_list)
    #if len(nested_list) < 2:
        #ToDo: Figure out what's wrong here, how come it goes to the empty set?
    #    return set()
    if type(nested_list[0]) == int and type(nested_list[1]) == int:
        nodes = nested_list[:2]
        #print(nodes)
        return set(nodes)
    else:
        nodes = set()
        
        for sub_list in nested_list:
            sub_nodes = find_unique_ints_in_nested_list(sub_list)
            nodes = nodes.union(sub_nodes)
        return nodes
    


# Functions that compress genomes
############################################################################################################

def compress_node_nr_difference(genome):
    # Identify all unique node numbers in the genome
    current_node_values = sorted(list(find_unique_ints_in_nested_list(genome)))
    
    # Create a mapping from current node numbers to a continuous range starting from 0
    translator = {current_value: new_value for new_value, current_value in enumerate(current_node_values)}
    
    # Apply the mapping to compress node numbers in the genome
    compressed_genome = swap_values_in_nested_list(genome, translator)
    return compressed_genome


def compress_node_nr_difference_old(genome):
    """
    Changes the node nrs in a genome such that they are continious
    from 0 to the number of nodes in the genome.
    This is done in case some nodes are removed during evolution leaving 
    gaps in the node nrs.
    """
    maximum = find_max_postive_int_in_nested_list(genome)
    minimum = find_min_postive_int_in_nested_list(genome)
    #print(maximum, minimum)
    
    current_node_values = find_unique_ints_in_nested_list(genome)
    current_node_values = list(current_node_values)
    target_node_values = list(range((maximum - minimum)+1))
    
    translator = {}
    for i, c_node in enumerate(current_node_values):
        
        translator[c_node] = target_node_values[i]
    
    compressed_genome = swap_values_in_nested_list(genome, translator)
    return compressed_genome


def swap_values_in_nested_list(nested_list, translator):
    """
    Helper function for compress_node_nr_difference
    Swaps the node numbers in the original genome to the new node numbers.
    """
    if type(nested_list) == int:
        return translator[nested_list]

    elif type(nested_list) == float:
        return nested_list 
        
    elif type(nested_list):
        l = []
        for sub_list in nested_list:
            r = swap_values_in_nested_list(sub_list, translator)
            l.append(r)
        return l

def delete_empty_lists(genome):
    """
    Searches through a nested list and deletes all empty lists.
    """
    if type(genome) == list:
        if len(genome) == 0:
            return True
        else:
            for sub_genome in genome:
                empty = delete_empty_lists(sub_genome)
                if empty == True:
                    genome.remove(sub_genome)
            return genome