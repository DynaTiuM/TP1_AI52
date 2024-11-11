import numpy as np
import random as rd
from random import randint
import pandas as pd
import time
import matplotlib.pyplot as plt

# Données du problème générées aléatoirement
capacite_max = 10  # La capacité du sac

# Paramètres de l'algorithme génétique
nbr_generations_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # Différentes valeurs de générations
capacite_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
solutions_par_pop = 8  # Taille de la population
csv_filename_generations = 'resultats_knapsack_generations.csv'
csv_filename_capacites = 'resultats_knapsack_capacites.csv'

# Liste des fichiers CSV à utiliser
csv_files = ['knapsack1.csv', 'knapsack2.csv', 'knapsack3.csv', 'knapsack4.csv']

# Ajustement des individus pour respecter la capacité
def change_individual(individual, capacite, poids):
    indices = list(range(len(individual)))
    rd.shuffle(indices)
    poids_total = 0

    for j in indices:
        if individual[j] == 1:
            if poids[j] + poids_total > capacite:
                individual[j] = 0
            else:
                individual[j] = 1
                poids_total += poids[j]

# Calcul du fitness pour la méthode A
def cal_fitness_A(poids, valeur, population, capacite):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * valeur)
        S2 = np.sum(population[i] * poids)

        if S2 <= capacite:
            fitness[i] = S1
        else:
            fitness[i] = capacite - S2

    return fitness.astype(int)

# Calcul du fitness pour la méthode B
def cal_fitness_B(poids, valeur, population, capacite):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        change_individual(population[i], capacite, poids)
        S1 = np.sum(population[i] * valeur)  # Valeur totale

        fitness[i] = S1

    return fitness.astype(int)

# Sélection des parents
def selection(fitness, nbr_parents, population):
    fitness = list(fitness)
    parents = np.empty((nbr_parents, population.shape[1]))

    for i in range(nbr_parents):
        indice_max_fitness = np.where(fitness == np.max(fitness))
        parents[i, :] = population[indice_max_fitness[0][0], :]
        fitness[indice_max_fitness[0][0]] = -999999

    return parents

# Croisement des parents pour générer des enfants
def croisement(parents, nbr_enfants):
    enfants = np.empty((nbr_enfants, parents.shape[1]))
    point_de_croisement = int(parents.shape[1] / 2)  # Croisement au milieu
    taux_de_croisement = 0.8
    i = 0

    while i < nbr_enfants:
        indice_parent1 = i % parents.shape[0]
        indice_parent2 = (i + 1) % parents.shape[0]
        x = rd.random()
        if x > taux_de_croisement:  # Probabilité de parents stériles
            continue
        enfants[i, 0:point_de_croisement] = parents[indice_parent1, 0:point_de_croisement]
        enfants[i, point_de_croisement:] = parents[indice_parent2, point_de_croisement:]
        i += 1

    return enfants

# Mutation des enfants
def mutation(enfants):
    mutants = np.empty((enfants.shape))
    taux_mutation = 0.5
    for i in range(mutants.shape[0]):
        random_valeur = rd.random()
        mutants[i, :] = enfants[i, :]
        if random_valeur > taux_mutation:
            continue
        int_random_valeur = randint(0, enfants.shape[1] - 1)  # Choisir aléatoirement le bit à inverser
        mutants[i, int_random_valeur] = 1 if mutants[i, int_random_valeur] == 0 else 0
    return mutants

# Fonction d'optimisation
def optimize(poids, methode, valeur, population, pop_size, nbr_generations, capacite):
    start_time = time.time()

    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0] // 2
    nbr_enfants = pop_size[0] - nbr_parents

    for _ in range(nbr_generations):
        if methode == 'A':
            fitness = cal_fitness_A(poids, valeur, population, capacite)
        else:
            fitness = cal_fitness_B(poids, valeur, population, capacite)

        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)
        enfants = croisement(parents, nbr_enfants)
        mutants = mutation(enfants)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    if methode == 'A':
        fitness_derniere_generation = cal_fitness_A(poids, valeur, population, capacite)
    else:
        fitness_derniere_generation = cal_fitness_B(poids, valeur, population, capacite)

    max_fitness = np.max(historique_fitness)
    sol_opt.append(population[np.argmax(fitness_derniere_generation), :])

    execution_time = time.time() - start_time
    
    '''historique_fitness_moyenne = [np.mean(fitness) for fitness in historique_fitness]
    historique_fitness_max = [np.max(fitness) for fitness in historique_fitness]
    plt.plot(list(range(nbr_generations)), historique_fitness_moyenne, label='Valeurs moyennes')
    plt.plot(list(range(nbr_generations)), historique_fitness_max, label='Valeur maximale')
    plt.legend()
    plt.title('Evolution de la Fitness à travers les générations en Euros')
    plt.xlabel('Générations')
    plt.ylabel('Fitness')
    plt.show()'''

    return sol_opt, historique_fitness, execution_time, max_fitness

# Fonction pour créer la population initiale
def create_population(pop_size):
    return np.random.randint(2, size=pop_size).astype(int)

# Fonction générique pour exécuter sur différentes générations ou capacités
def execute_with_param_list(param_list, param_type, csv_filename, pop_size):
    for methode in ['A', 'B']:
        for param in param_list:
            param_label = 'générations' if param_type == 'generation' else 'capacité'
            print(f"Lancement pour {param} {param_label} avec la méthode {methode} sur le fichier {csv_file}...")

            population_initiale = create_population(pop_size)
            sol_opt, historique_fitness, execution_time, fitness_max = optimize(
                poids, methode, valeur, population_initiale, pop_size,
                param if param_type == 'generation' else nbr_generations_list[-1],
                param if param_type == 'capacite' else capacite_max
            )

            print(f"Solution optimale pour la méthode {methode}:")
            print(f"Objets sélectionnés: {[i for i, j in enumerate(sol_opt[0]) if j != 0]}")
            print(f"Fitness maximale: {fitness_max}, Temps d'exécution: {execution_time} s\n")

            # Enregistrement des résultats dans le fichier CSV
            with open(csv_filename, 'a') as f:
                f.write(f'{csv_file},{param},{methode},{fitness_max},{execution_time}\n')

# Fonction pour exécuter l'algorithme avec plusieurs fichiers CSV et méthodes
def execute(csv_file):
    # Charger les données depuis le fichier CSV
    df = pd.read_csv(csv_file)
    global poids, valeur
    poids = df['Poids'].values
    valeur = df['Valeur'].values

    nombre_objets = len(df)
    pop_size = (solutions_par_pop, nombre_objets)  # Taille de la population

    execute_with_param_list(nbr_generations_list, 'generation', csv_filename_generations, pop_size)
    execute_with_param_list(capacite_list, 'capacite', csv_filename_capacites, pop_size)

with open(csv_filename_generations, 'w') as f:
    f.write('Fichier,Nombre_generations,Methode,Max_fitness,Temps_execution\n')
with open(csv_filename_capacites, 'w') as f:
    f.write('Fichier,Capacites,Methode,Max_fitness,Temps_execution\n')

# Exécution pour chaque fichier CSV
for csv_file in csv_files:
    execute(csv_file)