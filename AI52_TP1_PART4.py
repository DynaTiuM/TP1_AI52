import numpy as np
import random as rd
from random import randint
import pandas as pd
import time

budget_total = 100000  # Le budget total disponible

# Paramètres de l'algorithme génétique
nbr_generations_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # Différentes valeurs de générations
solutions_par_pop = 8  # Taille de la population
csv_filename_generations = 'resultats_wallet_generations_optimized.csv'

# Liste des fichiers CSV à utiliser
csv_files = ['wallet1.csv', 'wallet2.csv', 'wallet3.csv', 'wallet4.csv']

def change_individual(individual, budget_total, prix):
    """
    Modifie un individu pour respecter le budget total en ajustant le nombre de parts
    achetées pour chaque titre. Réduit les parts si le budget est dépassé et en ajoute
    si possible tout en respectant la limite de 20% du budget par titre.
    """

    indices = list(range(len(individual)))
    rd.shuffle(indices)
    cout_total = sum(individual[j] * prix[j] for j in range(len(individual)))

    # Réduire les parts si le coût total dépasse le budget
    for j in indices:
        if cout_total > budget_total:
            # Réduire proportionnellement les parts de ce titre
            reduction = min(individual[j], int((cout_total - budget_total) / prix[j]))
            individual[j] -= reduction
            cout_total -= reduction * prix[j]

    # Ajouter des parts supplémentaires si le budget le permet
    for j in indices:
        max_parts = int(0.2 * budget_total / prix[j])
        current_parts = individual[j]

        if current_parts < max_parts:
            parts_to_add = min(max_parts - current_parts, int((budget_total - cout_total) / prix[j]))
            if parts_to_add > 0:
                individual[j] += parts_to_add
                cout_total += parts_to_add * prix[j]

# Calcul de la fitness pour chaque portefeuille
def cal_fitness(prix, gain, population, budget_total):
    """
    Calcule la fitness de chaque portefeuille dans la population. La fitness correspond
    au gain total du portefeuille, après ajustement pour respecter le budget total.
    """

    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        change_individual(population[i], budget_total, prix)
        total_gain = np.sum(population[i] * gain)  # Gain total

        # Si le prix total dépasse le budget, on applique une pénalité
        fitness[i] = total_gain

    return fitness.astype(int)

# Sélection des parents
def selection(fitness, nbr_parents, population):
    """
    Sélectionne les meilleurs individus de la population (ceux avec la meilleure fitness)
    pour les utiliser comme parents dans le croisement.
    """
    fitness = list(fitness)
    parents = np.empty((nbr_parents, population.shape[1]))

    for i in range(nbr_parents):
        indice_max_fitness = np.where(fitness == np.max(fitness))
        parents[i, :] = population[indice_max_fitness[0][0], :]
        fitness[indice_max_fitness[0][0]] = -999999

    return parents

# Croisement des parents pour générer des enfants
def croisement(parents, nbr_enfants):
    """
    Réalise un croisement entre les parents pour générer de nouveaux enfants.
    Le croisement se fait au milieu du génome des parents.
    """
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
    """
    Applique des mutations sur certains enfants, avec un certain taux de mutation.
    """
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
    """
    Fonction principale qui exécute l'algorithme génétique sur un certain nombre de générations.
    Elle sélectionne les meilleurs individus, les croise et applique des mutations pour
    chercher une solution optimale.
    """
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

def create_population(pop_size):
    """
    Crée une population initiale de solutions aléatoires.
    """
    return np.random.randint(2, size=pop_size).astype(int)

def execute_generations(generations, csv_filename, pop_size):
    """
    Exécute l'algorithme génétique pour une liste de nombres de générations,
    et enregistre les résultats dans un fichier CSV.
    """
    for nbr_generations in generations:
        print(f"Lancement pour {nbr_generations} générations sur le fichier {csv_file}...")

        population_initiale = create_population(pop_size)
        sol_opt, execution_time, fitness_max = optimize(
            prix, gain, population_initiale, pop_size,
            nbr_generations, budget_total
        )

        # On crée un dictionnaire pour stocker le nombre de parts pour chaque titre
        titres_selectionnes = {i: sol_opt[0][i] for i in range(len(sol_opt[0])) if sol_opt[0][i] > 0}

        # On affiche le nombre de parts pour chaque titre sélectionné
        for titre_index, nombre_parts in titres_selectionnes.items():
            print(f"Titre {titre_index}: {nombre_parts} parts")

        # On affiche le gain maximal et le temps d'exécution
        print(f"Gain maximal: {fitness_max}, Temps d'exécution: {execution_time} s\n")

        # Enfin, on enregistre les résultats dans le fichier CSV
        with open(csv_filename, 'a') as f:
            f.write(f'{csv_file},{nbr_generations},{fitness_max},{execution_time}\n')


def execute(csv_file):
    """
    Exécute l'algorithme avec un fichier CSV précis.
    """
    df = pd.read_csv(csv_file)
    global prix, gain
    prix = df['Prix'].values
    gain = df['Gain'].values

    nombre_titres = len(df)
    pop_size = (solutions_par_pop, nombre_titres)  # Taille de la population

    execute_generations(nbr_generations_list, csv_filename_generations, pop_size)


with open(csv_filename_generations, 'w') as f:
    f.write('Fichier,Nombre_generations,Max_gain,Temps_execution\n')

# Exécution pour chaque fichier CSV
for csv_file in csv_files:
    execute(csv_file)

