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

def single_point_crossover(a: List[int], b: List[int]):
    if len(a) != len(b):
        raise ValueError("parentA and parentB not matching")
    length = len(a)
    if length < 2:
        return a, b
    p = random.randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def main():
    cities = parse_input_file("input.txt")
    initial_list = list(range(len(cities)))
    population = initialize_population(initial_list, 10)
    matrix = initialize_matrix(cities)
    fitness = calculate_fitness(matrix, population)

    #choose 2 parents
    parentA, parentB = choose_parents(population, fitness)
    
    print(parentA, parentB)
    #Crossover genomes from chosen parents
    childA, childB = single_point_crossover(parentA, parentB)
    print(childA, childB)

if __name__ == "__main__":
    main()