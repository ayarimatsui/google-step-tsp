# Google STEP Program week5 homework
# Genetic Algorithm

import sys
import math
import random
import pandas as pd
import numpy as np
from common import print_tour, read_input
import solver_random


class GA():

    def __init__(self, cities, pop_size=None, mutation_rate=None, generations=None):
        self.cities = cities
        self.N = len(cities)
        if pop_size is None:
            self.pop_size = 100  # a parameter to change
        else:
            self.pop_size = pop_size
        self.elite_size = int(0.3 * self.pop_size)  # define elite_size as 20% of the pop_size
        if mutation_rate is None:
            self.mutation_rate = 0.01  # a parameter to change
        else:
            self.mutation_rate = mutation_rate
        if generations is None:
            self.generations = 1000  # a parameter to change
        else:
            self.generations = generations


    # create a tour randomly
    def create_rand_tour(self):
        tour = solver_random.solve(self.cities)
        return tour


    # initialize with random tours
    def initialize_population(self):
        population_list = []
        for _ in range(self.pop_size):
            population_list.append(self.create_rand_tour())
        return population_list


    # calculate distance between two cities
    def distance(self, node1, node2):
        city1, city2 = self.cities[node1], self.cities[node2]
        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


    # calculate the total distance of the tour
    def total_dist(self, tour):
        total_distance = 0
        for i in range(self.N):
            total_distance += self.distance(tour[i % self.N], tour[(i + 1) % self.N])
        return total_distance
    

    # make the rankings of tours
    def rank_tours(self, population_list):
        dist_dic = {}
        for i in range(len(population_list)):
            dist_dic[i] = self.total_dist(population_list[i])
        return sorted(dist_dic.items(), key=lambda x:x[1])


    # selection process, returning a list of tour IDs
    def selection(self, ranked_tours):
        selection_results = []
        df = pd.DataFrame(np.array(ranked_tours), columns=["Index","Distance"])
        df['cum_sum'] = df.Distance.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Distance.sum()
        
        # elitism
        for i in range(self.elite_size):
            selection_results.append(ranked_tours[i][0])

        # select rest
        pick = 100 * random.random()
        for _ in range(len(ranked_tours) - self.elite_size):
            for i in range(len(ranked_tours)):
                if pick <= df.iat[i,3]:
                    selection_results.append(ranked_tours[i][0])
                    break
        
        return selection_results


    # make up mating pool
    def mating_pool(self, population_list, selection_results):
        mating_pool = []
        for ID in selection_results:
            mating_pool.append(population_list[ID])
        return mating_pool


    # breeding process
    def breed(self, parent1, parent2):
        child = []
        childP1 = []
        childP2 = []
        
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(parent1[i])
            
        childP2 = [item for item in parent2 if item not in childP1]

        child = childP1 + childP2
        return child


    # return a list of breeded population
    def breed_population(self, matingpool):
        children = []
        length = len(matingpool) - self.elite_size
        pool = random.sample(matingpool, len(matingpool))

        for i in range(self.elite_size):
            children.append(matingpool[i])
        
        for i in range(length):
            # make a child by breeding
            child = self.breed(pool[i], pool[len(matingpool)-i-1])
            children.append(child)
        return children


    # mutation process (mutation by swapping two cities)
    def mutate(self, individual):
        for swapped in range(len(individual)):
            if(random.random() < self.mutation_rate):
                # swap two cities randomly
                swapWith = int(random.random() * len(individual))
                city1 = individual[swapped]
                city2 = individual[swapWith]
                
                individual[swapped] = city2
                individual[swapWith] = city1
        return individual


    # make mutated population list
    def mutate_population(self, population_list):
        mutated_pop = []
        for individual in population_list:
            mutated = self.mutate(individual)
            mutated_pop.append(mutated)

        return mutated_pop


    # create a new generation by mixing breeding and mutationg process
    def next_generation(self, current_generation):
        ranked_tours = self.rank_tours(current_generation)
        selection_results = self.selection(ranked_tours)
        mating_pool = self.mating_pool(current_generation, selection_results)
        children = self.breed_population(mating_pool)
        next_gen = self.mutate_population(children)
        return next_gen


    # solve by genetic algorithm
    def genetic_algorithm(self):
        population_list = self.initialize_population()
        print("Initial distance: {}".format(self.rank_tours(population_list)[0][1]))
        
        for _ in range(self.generations):
            population_list = self.next_generation(population_list)
        
        print("Final distance: is {}".format(self.rank_tours(population_list)[0][1]))
        best_tour_idx = self.rank_tours(population_list)[0][0]
        best_tour = population_list[best_tour_idx]
        return best_tour



# solve function
def solve(cities):
    ga = GA(cities)
    tour = ga.genetic_algorithm()
    return tour



if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)
    #print_tour(tour)