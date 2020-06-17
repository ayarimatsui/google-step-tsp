# Google STEP Program week5 homework
# Simulated Annealing Algorithm

import sys
import math
import random
from common import print_tour, read_input
import solver_random
import solver_greedy
import solver_2_opt
import solver_ts


class MySA():

    def __init__(self, cities, initialize, T=None, alpha=None, stopping_T=None, stopping_iter=None):
        self.cities = cities
        self.N = len(cities)
        if T is None:
            self.T = math.sqrt(self.N)
        else:
            self.T = T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        if alpha is None:
            self.alpha = 0.9999  # initial
        else:
            self.alpha = alpha
        if stopping_T is None:
            self.stopping_temperature = 1e-50
        else:
            self.stopping_temperature = stopping_T
        if stopping_iter is None:
            self.stopping_iter = 1000000
        else:
            self.stopping_iter = stopping_iter
        self.iteration = 1
        self.nodes = [i for i in range(self.N)]
        self.best_tour = None
        self.best_distance = float("Inf")
        self.distance_list = []
        self.initialize = initialize


    # greedy algorithm
    def solve_greedy(self):

        dist = [[0] * self.N for i in range(self.N)]
        for i in range(self.N):
            for j in range(i, self.N):
                dist[i][j] = dist[j][i] = self.distance(i, j)

        current_city = random.randrange(self.N)
        unvisited_cities = list(range(self.N))
        unvisited_cities.pop(current_city)
        tour = [current_city]

        while unvisited_cities:
            next_city = min(unvisited_cities,
                            key=lambda city: dist[current_city][city])
            unvisited_cities.remove(next_city)
            tour.append(next_city)
            current_city = next_city
        return tour


    def initialize_tour(self):
        if self.initialize is None:
            if self.N < 50:
                # initialize a tour with random
                tour = random.sample(list(range(self.N)), self.N)
            else:
                # initialize a tour by greedy algorithm
                tour = self.solve_greedy()

            cur_distance = self.total_dist(tour)

            if cur_distance < self.best_distance:  # If best found so far, update best distance
                self.best_distance = cur_distance
                self.best_tour = tour
            self.distance_list.append(cur_distance)
            return tour, cur_distance
        else:
            tour = self.initialize
            distance = self.total_dist(tour)
            if distance < self.best_distance:  # If best found so far, update best distance
                self.best_distance = distance
                self.best_tour = tour
            self.distance_list.append(distance)
            return tour, distance


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


    # calculate probability of accepting candidate if it's worse than current. 
    def p_accept(self, candidate_distance):
        return math.exp(-abs(candidate_distance - self.cur_distance) / self.T)


    # accept if the candidate is better than current, and accept with probability p_accept if it is worse than current
    def accept(self, candidate_tour):
        candidate_distance = self.total_dist(candidate_tour)
        if candidate_distance < self.cur_distance:
            self.cur_tour = candidate_tour
            self.cur_distance = candidate_distance
            # if it is better than the best
            if candidate_distance < self.best_distance:
                self.best_tour = candidate_tour
                self.best_distance = candidate_distance
        else:
            if random.random() < self.p_accept(candidate_distance):
                self.cur_tour = candidate_tour
                self.cur_distance = candidate_distance


    # annealing process
    def anneal(self):
        # Initialize tour.
        self.cur_tour, self.cur_distance = self.initialize_tour()

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate_tour = list(self.cur_tour)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate_tour[i : (i + l)] = reversed(candidate_tour[i : (i + l)])
            self.accept(candidate_tour)
            if self.iteration >= 10000:
                self.alpha = 0.996
            self.T *= self.alpha
            self.iteration += 1

            self.distance_list.append(self.cur_distance)
            

        print("Best distance obtained: ", self.best_distance)
        improvement = 100 * (self.distance_list[0] - self.best_distance) / (self.distance_list[0])
        print(f"Improvement over first distance: {improvement : .2f}%")
        return self.best_tour


# solve function
def solve(cities, initialize=None):
    my_SA = MySA(cities, initialize=None)
    tour = my_SA.anneal()
    return tour

def solve_with_initial(cities, initialize, alpha, stopping_T, stopping_iter):
    my_SA = MySA(cities, initialize, alpha=alpha, stopping_T=stopping_T, stopping_iter=stopping_iter)
    tour = my_SA.anneal()
    return tour


if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)
    print_tour(tour)

    