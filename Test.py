import hierarchical_genomes as hg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

import reservoirpy as rpy
from reservoirpy.nodes import Ridge, Reservoir
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse

genome = [[0,1,0.5],[1,2,0.5],[2,3,0.1]]
n_generations=100
population=50 
genome_population=[copy.deepcopy(genome) for i in range(population)]
X = mackey_glass(1000)


for generation in range(n_generations):
    genome, mutation = hg.mutate_genome_with_hox(genome)
    
    
    weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(genome)
    
    for genome in genome_population:
        if(len(genome)>2):
            mutated_genome, mutation = hg.mutate_genome_with_hox(genome)
        
            
            weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(mutated_genome)
            
            #data = Input(input_dim=1)
            reservoir = Reservoir(W=weight_matrix)
            readout = Ridge(ridge=1e-6)


            esn = reservoir >> readout

            forecast = esn.fit(X[:500], X[1:501]).run(X[502:])
            fitness = rmse(forecast,X[502:])