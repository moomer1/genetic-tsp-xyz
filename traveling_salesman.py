from typing import List, Tuple
from random import shuffle, choices
import numpy as np
import random
#Maps points into a tuple
points3D = Tuple[float, float, float]

#Parse input file into an array
def parse_input_file(path: str) -> List[points3D]:
    with open(path, "r") as f:
        n = int(f.readline().strip())
        return [tuple(map(float, f.readline().split())) for _ in range(n)]

#Shuffles the indices of points *population_size* times to represent our initial population
def initialize_population(num_cities: list, population_size: int) -> List[List[int]]:
    population = []
    for _ in range(population_size):
        tour = num_cities[:]
        shuffle(tour)
        population.append(tour)
    return population

#Initializes matrix from which distances scores are calculated
def initialize_matrix(cities: list) -> List[List[float]]:
    matrix = []
    for p1 in cities:
        tour = []
        for p2 in cities:
            arr1 = np.array(p1)
            arr2 = np.array(p2)
            squared_dist = np.sum((arr1-arr2)**2, axis=0)
            dist = np.sqrt(squared_dist)
            tour.append(float(dist))
        matrix.append(tour)
    return matrix

#Calculates fitness according to the scores calculated in the distance matrix
def calculate_fitness(matrix: List[List[float]], population: List[List[int]]) -> List[float]:
    fitness = []
    for i in range(len(population)):
        curr = 0
        for j in range(1, len(population[i])):
            curr += matrix[population[i][j-1]][population[i][j]]
        curr += matrix[population[i][-1]][population[i][0]]
        fitness.append(curr)
    return fitness

#Choose parents on a tournament based selection
def choose_parents(population: List[List[int]], fitness: List[float]) -> List[int]:
    contenders = random.sample(range(len(population)), 5)      # step 1
    winner_idx = min(contenders, key=lambda i: fitness[i])     # step 2
    parentA_idx = winner_idx
    parentA = population[parentA_idx]                          # step 3

    # second parent
    contenders = random.sample(range(len(population)), 5)      # step 4 (repeat)
    winner_idx = min(contenders, key=lambda i: fitness[i])
    parentB_idx = winner_idx
    parentB = population[parentB_idx]
    return parentA, parentB

#Permutes one part of parentA with parentB
def single_point_crossover(a: List[int], b: List[int]):
    if len(a) != len(b):
        raise ValueError("parentA and parentB not matching")
    length = len(a)
    if length < 2:
        return a, b
    p = random.randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

#swap-based: keeps it a permutation
def mutation(path: List[int], num: int = 1, probability: float = 0.5) -> List[int]:
    for _ in range(num):
        if random.random() < probability:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
    return path


#Replace duplicates with missing cities, preserving order as much as possible
def repair_to_permutation(child: List[int], n: int) -> List[int]:
    seen = set()
    missing = [c for c in range(n) if c not in set(child)]
    missing_idx = 0
    fixed = []
    for gene in child:
        if gene not in seen:
            fixed.append(gene)
            seen.add(gene)
        else:
            fixed.append(missing[missing_idx])
            missing_idx += 1
    return fixed

def write_output(filename: str, best_length: float,
                 best_tour: List[int], cities: List[Tuple[float,float,float]]) -> None:
    with open(filename, "w") as f:
        f.write(f"{best_length:.3f}\n")              
        for idx in best_tour:
            x, y, z = cities[idx]
            f.write(f"{int(x)} {int(y)} {int(z)}\n") 

def main():
    cities = parse_input_file("input.txt")
    initial_list = list(range(len(cities)))
    population = initialize_population(initial_list, 10)
    matrix = initialize_matrix(cities)

    generations = 100 

    for _ in range(generations):
        fitness = calculate_fitness(matrix, population)
        next_generation: List[List[int]] = []

        # keep making children until next_generation matches current population size
        while len(next_generation) < len(population):
            parentA, parentB = choose_parents(population, fitness)

            # crossover -> two children
            childA, childB = single_point_crossover(parentA, parentB)

            # REPAIR: make each child a valid permutation (no dups/missing)
            n = len(parentA)
            childA = repair_to_permutation(childA, n)
            childB = repair_to_permutation(childB, n)

            # mutate and append (guard in case of odd pop size)
            childA = mutation(childA)
            if len(next_generation) < len(population):
                next_generation.append(childA)

            childB = mutation(childB)
            if len(next_generation) < len(population):
                next_generation.append(childB)


        # move to next generation
        population = next_generation

    # final report
    fitness = calculate_fitness(matrix, population)
    best_idx = min(range(len(population)), key=lambda i: fitness[i])
    best_length = fitness[best_idx]
    best_tour = population[best_idx]

    write_output("output.txt", best_length, best_tour, cities)



if __name__ == "__main__":
    main()