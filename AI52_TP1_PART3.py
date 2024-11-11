import numpy as np
import random as rd
from random import randint
import pandas as pd
import time

# Données du portefeuille générées aléatoirement
budget_total = 100000  # Le budget total disponible

# Paramètres de l'algorithme génétique
nbr_generations_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,  75, 80, 85, 90, 95, 100]  # Différentes valeurs de générations
solutions_par_pop = 8  # Taille de la population
csv_filename_generations = 'resultats_wallet_generations.csv'

# Liste des fichiers CSV à utiliser
csv_files = ['wallet1.csv', 'wallet2.csv', 'wallet3.csv', 'wallet4.csv']

# Ajustement des individus pour respecter la contrainte des 20% du budget
def change_individual(individual, budget_total, prix):
    indices = list(range(len(individual)))
    rd.shuffle(indices)
    cout_total = sum(individual[j] * prix[j] for j in range(len(individual)))

    for j in indices:
        # Calculer le nombre maximal de parts pouvant être achetées pour ce titre
        max_parts = int(0.2 * budget_total / prix[j])

        # S'assurer qu'on ne dépasse pas le nombre maximal de parts
        current_parts = individual[j]

        if cout_total > budget_total:
            individual[j] = 0
            cout_total -= individual[j] * prix[j]

        if current_parts < max_parts:
            cout = prix[j]  # Coût pour une part
            if cout_total + cout <= budget_total:
                individual[j] += 1  # Acheter une part supplémentaire
                cout_total += cout

# Calcul de la fitness pour chaque portefeuille
def cal_fitness(prix, gain, population, budget_total):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        change_individual(population[i], budget_total, prix)
        total_gain = np.sum(population[i] * gain)  # Gain total

        # Si le prix total dépasse le budget, on applique une pénalité
        fitness[i] = total_gain

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
def optimize(prix, gain, population, pop_size, nbr_generations, budget_total):
    start_time = time.time()

    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0] // 2
    nbr_enfants = pop_size[0] - nbr_parents

    for _ in range(nbr_generations):
        fitness = cal_fitness(prix, gain, population, budget_total)

        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)
        enfants = croisement(parents, nbr_enfants)
        mutants = mutation(enfants)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    fitness_derniere_generation = cal_fitness(prix, gain, population, budget_total)

    max_fitness = np.max(historique_fitness)
    sol_opt.append(population[np.argmax(fitness_derniere_generation), :])

    execution_time = time.time() - start_time

    return sol_opt, execution_time, max_fitness

# Fonction pour créer la population initiale
def create_population(pop_size):
    return np.random.randint(2, size=pop_size).astype(int)

# Fonction générique pour exécuter sur différentes générations
def execute_with_param_list(generations, csv_filename, pop_size):
    for nbr_generations in generations:
        print(f"Lancement pour {nbr_generations} générations sur le fichier {csv_file}...")

        population_initiale = create_population(pop_size)
        sol_opt, execution_time, fitness_max = optimize(
            prix, gain, population_initiale, pop_size,
            nbr_generations, budget_total
        )

        # Créer un dictionnaire pour stocker le nombre de parts pour chaque titre
        titres_selectionnes = {i: sol_opt[0][i] for i in range(len(sol_opt[0])) if sol_opt[0][i] > 0}

        # Afficher le nombre de parts pour chaque titre sélectionné
        for titre_index, nombre_parts in titres_selectionnes.items():
            print(f"Titre {titre_index}: {nombre_parts} parts")

        # Afficher le gain maximal et le temps d'exécution
        print(f"Gain maximal: {fitness_max}, Temps d'exécution: {execution_time} s\n")

        # Enregistrement des résultats dans le fichier CSV
        with open(csv_filename, 'a') as f:
            f.write(f'{csv_file},{nbr_generations},{fitness_max},{execution_time}\n')

# Fonction pour exécuter l'algorithme avec plusieurs fichiers CSV
def execute(csv_file):
    # Charger les données depuis le fichier CSV
    df = pd.read_csv(csv_file)
    global prix, gain
    prix = df['Prix'].values
    gain = df['Gain'].values

    nombre_titres = len(df)
    pop_size = (solutions_par_pop, nombre_titres)  # Taille de la population

    execute_with_param_list(nbr_generations_list, csv_filename_generations, pop_size)

with open(csv_filename_generations, 'w') as f:
    f.write('Fichier,Nombre_generations,Max_gain,Temps_execution\n')

# Exécution pour chaque fichier CSV
for csv_file in csv_files:
    execute(csv_file)

