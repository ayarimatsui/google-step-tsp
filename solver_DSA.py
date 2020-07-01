# Google STEP Program week7 homework
# TSP Problem
# divide and conquer algorithm with Simulated Annealing Algorithm
# 2/3DSA

import sys
import math
import random
import multiprocessing
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
        

    # greedy algorithm
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
        '''
        # initialize a tour with random
        mid_tour = self.initial_tour[1:-1]
        tour = random.sample(mid_tour, len(mid_tour))
        tour.insert(0, self.start)
        tour.append(self.goal)
        '''
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

    # initialize with the SA tour
    sa_tour = solver_my_sa.solve(cities)
    sa_tour = solve_2opt(cities, sa_tour)
    # initialize with 2-opt tour
    #sa_tour = solver_2_opt.solve(cities)
    best_dist = total_distance(cities, sa_tour)
    print('the first distance is : {}'.format(best_dist))

    '''
    # initialize with the SA tour
    sa_tour = solver_my_sa.solve(cities)
    sa_tour = solve_2opt(cities, sa_tour)
    best_distance = total_distance(cities, sa_tour)
    best_tour = sa_tour'''

    for _ in range(10):
        # 分割数
        split = 24
        # choose a city to start randomly
        points = []
        for i in range(split):
            if i == 0:
                points.append(random.randrange(N))
            else:
                point = (points[-1] + N // split)
                points.append(point)
        
        tours = []
        doubled_sa_tour = sa_tour + sa_tour
        for i in range(split):
            if i < split - 1:
                splited_tour = doubled_sa_tour[points[i % split] : points[(i + 1) % split]]
            else:
                splited_tour = doubled_sa_tour[points[i % split] : points[0] + N]
            tours.append(splited_tour)

        '''
        if N % 3 == 0:
            one_third_from_start = (start + N // 3) % N
            two_third_from_start = (one_third_from_start + N // 3) % N
        elif N % 3 == 1:
            one_third_from_start = (start + N // 3) % N
            two_third_from_start = (one_third_from_start +  1 + N // 3) % N
        else:
            one_third_from_start = (start + 1 +  N // 3) % N
            two_third_from_start = (one_third_from_start + N // 3) % N'''
        '''
        if one_third_from_start < start:
            first_tour = sa_tour[start : N] + sa_tour[:one_third_from_start]
            second_tour = sa_tour[one_third_from_start : two_third_from_start]
            third_tour = sa_tour[two_third_from_start : start]
        elif two_third_from_start < one_third_from_start:
            first_tour = sa_tour[start : one_third_from_start]
            second_tour = sa_tour[one_third_from_start : N] + sa_tour[:two_third_from_start]
            third_tour = sa_tour[two_third_from_start : start]
        else:
            first_tour = sa_tour[start : one_third_from_start]
            second_tour = sa_tour[one_third_from_start : two_third_from_start]
            third_tour = sa_tour[two_third_from_start : N] + sa_tour[:start]'''


        '''
        print(sa_tour)
        print('first tour')
        print(first_tour)
        print('second tour')
        print(second_tour)
        print('third tour')
        print(third_tour)
        print(len(first_tour) + len(second_tour) + len(third_tour))'''
        manager = multiprocessing.Manager()
        proc_dic = manager.dict()
        processes = []
        for i in range(split):
            p = multiprocessing.Process(target=solve_each_sa, args=(i, proc_dic, cities, tours[i], city_distance_list,))
            processes.append(p)

        '''
        p1 = multiprocessing.Process(target=solve_each_sa, args=(1, proc_dic, cities, first_tour, city_distance_list,))
        p2 = multiprocessing.Process(target=solve_each_sa, args=(2, proc_dic, cities, second_tour, city_distance_list,))
        p3 = multiprocessing.Process(target=solve_each_sa, args=(3, proc_dic, cities, third_tour, city_distance_list,))
        p4 = multiprocessing.Process(target=solve_each_sa, args=(4, proc_dic, cities, fourth_tour, city_distance_list,))'''

        # start the processes
        for p in processes:
            p.start()

        '''
        p1.start()
        p2.start()
        p3.start()
        p4.start()'''
        
        # プロセス終了待ち合わせ
        for p in processes:
            p.join()

        '''
        p1.join()
        p2.join()
        p3.join()
        p4.join()'''

        #tour1 = proc_dic[str(1)] + second_tour + third_tour
        #tour2 = first_tour + proc_dic[str(2)] + third_tour
        #tour3 = first_tour + second_tour + proc_dic[str(3)]
        tour = proc_dic[str(0)]
        for i in range(1, split):
            tour += proc_dic[str(i)]
        #print(len(proc_dic[str(1)]))
        #print(proc_dic[str(2)])
        #print(proc_dic[str(3)])
        #print('number of cities : ' + str(N)  + '    len(tour) = : ' + str(len(tour)))
        '''
        tour1 = solve_2opt(cities, tour1)
        tour2 = solve_2opt(cities, tour2)
        tour3 = solve_2opt(cities, tour3)
        tour4 = solve_2opt(cities, tour4)
        
        dist1 = total_distance(cities, tour1)
        dist2 = total_distance(cities, tour2)
        dist3 = total_distance(cities, tour3)
        dist4 = total_distance(cities, tour4)
        
        if dist1 == min([dist1, dist2, dist3]):
            dist = dist1
            tour = tour1
        elif dist2 == min([dist1, dist2, dist3]):
            dist = dist2
            tour = tour2
        elif dist3 == min([dist1, dist2, dist3]):
            dist = dist3
            tour = tour3
        else:
            dist = dist4
            tour = tour4
            print('the best is tour4')'''
        
        tour = solve_2opt(cities, tour)
        dist = total_distance(cities, tour)

        print('the distance is {}'.format(dist))
        
        if dist < best_distance:
            best_distance = dist
            best_tour = tour


    print('the final distance is {}'.format(best_distance))
    return best_tour, best_distance


if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)