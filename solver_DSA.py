# Google STEP Program week7 homework
# TSP Problem
# divide and conquer algorithm with Simulated Annealing Algorithm

#################################
# 1st Step. initialize a tour with 2-opt algorithm
# 2nd Step. choose a city (= city A) randomly and define split number (= split). (split = 32 for Challenge 6, split = 96 for Challenge 7)
# 3rd Step. According to current tour, choose (split - 1) cities which are 1 * (N / split), 2 * (N / split), 3 * (N / split), ..., (split - 1) * (N / split) away from city A as split points
# 4th Step. Split the whole tour into split number by the split points and optimize each tours with SA algorithm.
# 5th Step. Unite all optimized splited tours into one tour and if the total distance is better than the record, update the whole tour.
# 6th Step. repeat 2nd Step ~ 5th Step for n times. (n = 2000 for Challenge 6, n = 200 for Challenge 7)
#################################

import sys
import math
import random
import multiprocessing
from multiprocessing import Pool
from common import print_tour, read_input
import solver_my_sa
import solver_2_opt
from solver_greedy import distance


class SAWithStartandGoal():

    def __init__(self, cities, initial_tour, city_distance_list, T=None, alpha=None, stopping_T=None, stopping_iter=None):
        self.cities = cities
        self.initial_tour = initial_tour
        self.N = len(initial_tour)
        self.start = initial_tour[0]
        self.goal = initial_tour[-1]
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
        self.best_tour = initial_tour
        self.best_distance = self.total_dist(initial_tour)
        self.city_dist_list = city_distance_list
        self.distance_list = []
        

    # greedy algorithm with start and goal
    def solve_greedy(self):
        current_city = self.start
        goal_city = self.goal
        unvisited_cities = self.initial_tour[1:-1]
        tour = [current_city]

        while unvisited_cities:
            next_city = min(unvisited_cities,
                            key=lambda city: self.city_dist_list[current_city][city])
            unvisited_cities.remove(next_city)
            tour.append(next_city)
            current_city = next_city

        tour.append(goal_city)

        return tour


    def initialize_tour(self):
        if self.N < 50:
            # initialize a tour with random
            mid_tour = self.initial_tour[1:-1]
            tour = random.sample(mid_tour, len(mid_tour))
            tour.insert(0, self.start)
            tour.append(self.goal)
        else:
            # initialize a tour by greedy algorithm
            tour = self.solve_greedy()

        cur_distance = self.total_dist(tour)

        if cur_distance < self.best_distance:  # If best found so far, update best distance
            self.best_distance = cur_distance
            self.best_tour = tour
        self.distance_list.append(cur_distance)
        return tour, cur_distance
        


    # calculate distance between two cities
    def distance(self, node1, node2):
        city1, city2 = self.cities[node1], self.cities[node2]
        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


    # calculate the total distance of the tour
    def total_dist(self, tour):
        total_distance = 0
        for i in range(self.N - 1):
            total_distance += self.distance(tour[i], tour[i + 1])
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

        #print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate_tour = list(self.cur_tour)
            l = random.randint(2, self.N - 2)
            i = random.randint(1, self.N - l)
            candidate_tour[i : (i + l)] = reversed(candidate_tour[i : (i + l)])
            self.accept(candidate_tour)
            if self.iteration >= 10000:
                self.alpha = 0.996
            self.T *= self.alpha
            self.iteration += 1

            self.distance_list.append(self.cur_distance)
            

        #print("Best distance obtained: ", self.best_distance)
        improvement = 100 * (self.distance_list[0] - self.best_distance) / (self.distance_list[0])
        #print(f"Improvement over first distance: {improvement : .2f}%")
        return self.best_tour


# optimize each splited tours
def solve_each_sa(proc_num, proc_dic, cities, initial_tour, city_distance_list):
    each_SA = SAWithStartandGoal(cities, initial_tour, city_distance_list)
    tour = each_SA.anneal()
    proc_dic[str(proc_num)] = tour


# calculate the total distance of the tour
def total_distance(cities, tour):
    N = len(tour)
    total_distance = 0
    for i in range(N):
        total_distance += distance(cities[tour[i % N]], cities[tour[(i + 1) % N]])
    return total_distance


#2-opt法
def solve_2opt(cities, tour):
    N = len(cities)
    while True:
        count = 0
        for i in range(N - 2):
            i_next = i + 1
            for j in range(i + 2, N):
                if j == N - 1:
                    j_next = 0
                else:
                    j_next = j + 1
                if i != 0 or j_next != 0:
                    original_dist = distance(cities[tour[i]], cities[tour[i_next]]) + distance(cities[tour[j]], cities[tour[j_next]])
                    new_dist = distance(cities[tour[i]], cities[tour[j]]) + distance(cities[tour[i_next]], cities[tour[j_next]])
                    if new_dist < original_dist:
                        new_route = tour[i_next:j + 1]
                        tour[i_next:j + 1] = reversed(new_route)
                        count += 1
        if count == 0:
            break
    return tour


def solve(cities):
    N = len(cities)

    city_distance_list = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(i, N):
            city_distance_list[i][j] = city_distance_list[j][i] = distance(cities[i], cities[j])

    best_distance = float('Inf')
    best_tour = None

    # initialize with 2-opt tour
    original_tour = solver_2_opt.solve(cities)
    best_dist = total_distance(cities, original_tour)
    print('the first distance is : {}'.format(best_dist))


    for _ in range(200):  # iterate for 2000 times for challenge 6, 200 times for challenge 7
        # split number 
        split = 96   # split in 32 for challenge 6, in 96 for challenge 7

        # choose a city to start randomly
        points = []  # a list that contains the city ids which are the first of each splited route.
        for i in range(split):
            if i == 0:
                points.append(random.randrange(N))
            else:
                point = (points[-1] + N // split)
                points.append(point)
        
        tours = []  # contains splited tours
        doubled_original_tour = original_tour + original_tour
        for i in range(split):
            if i < split - 1:
                splited_tour = doubled_original_tour[points[i % split] : points[(i + 1) % split]]
            else:
                splited_tour = doubled_original_tour[points[i % split] : points[0] + N]
            tours.append(splited_tour)

        # use multiprocess
        manager = multiprocessing.Manager()
        proc_dic = manager.dict()
        p = Pool(split)
        for i in range(split):
            p.apply_async(solve_each_sa, args=(i, proc_dic, cities, tours[i], city_distance_list,))
            
        p.close()
        p.join()

        # unite the each optimized tours into one large tour
        tour = proc_dic[str(0)]
        for i in range(1, split):
            tour += proc_dic[str(i)]
        
        tour = solve_2opt(cities, tour)
        dist = total_distance(cities, tour)

        print('the distance is {}'.format(dist))
        
        if dist < best_distance:
            best_distance = dist
            best_tour = tour
            original_tour = tour  # update the original tour


    print('the final distance is {}'.format(best_distance))
    return best_tour, best_distance


if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)