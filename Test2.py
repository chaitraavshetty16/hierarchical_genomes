import matplotlib
matplotlib.use('TkAgg')  # Setting the backend to 'TkAgg' 

import hierarchical_genomes as hg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
import reservoirpy as rpy
from reservoirpy.nodes import Ridge, Reservoir
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse

genome = [[0,1,0.5],[1,2,0.5],[2,3,0.1],[0,1,0.5],[1,2,0.5],[2,3,0.1],[0,1,0.5],[1,2,0.5],[2,3,0.1]]
n_generations = 50  # Reduced for testing
population = 10     # Reduced for testing
genome_population = [copy.deepcopy(genome) for i in range(population)]
X = mackey_glass(500)  # Reduced size for testing

fitness_history = []

for generation in range(n_generations):
    list_of_genomes = []
    list_of_fitness = []
    
    for genome in genome_population:
        if len(genome) > 2:
            mutated_genome, mutation = hg.mutate_genome_with_hox(genome)
            
            # Add checks or logging to see genome status
            print(f"Gen {generation}, Genome Length: {len(mutated_genome)}")

            weight_matrix = hg.transcribe_hierarchical_genome_to_weight_matrix(mutated_genome)
            reservoir = Reservoir(W=weight_matrix)
            readout = Ridge(ridge=1e-6)
            esn = reservoir >> readout

            forecast = esn.fit(X[:250], X[1:251]).run(X[252:])  # Reduced size for testing
            fitness = rmse(forecast, X[252:])  # Reduced size for testing

            list_of_genomes.append(genome)
            list_of_fitness.append(fitness)
    
    best_rmse = np.argmin(list_of_fitness)
    best_genome = list_of_genomes[best_rmse]
    fitness_history.append(best_rmse)
    genome_population = [copy.deepcopy(best_genome) for i in range(population)]
    
plt.plot(fitness_history)
plt.show()
