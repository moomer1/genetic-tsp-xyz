from typing import List, Tuple
from random import shuffle
import numpy as np
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


def main():
    cities = parse_input_file("input.txt")
    initial_list = list(range(len(cities)))
    population = initialize_population(initial_list, 160)
    matrix = initialize_matrix(cities)
    for arr in matrix:
        print(arr)
if __name__ == "__main__":
    main()