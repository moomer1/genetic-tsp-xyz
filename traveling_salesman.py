from typing import List, Tuple
from random import shuffle

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
    for i in range(160):
        tour = num_cities[:]
        shuffle(tour)
        population.append(tour)
    return population

def initialize_matrix(cities):
    print("placeholder")

def main():
    cities = parse_input_file("input.txt")
    initial_list = list(range(len(cities)))
    population = initialize_population(initial_list, 160)
    print(population)
if __name__ == "__main__":
    main()