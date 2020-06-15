# Google STEP Program week5 homework
# Simulated Annealing Algorithm ans 2-opt algorithm
# 2-opt after SA

import sys
import math
import random
from common import print_tour, read_input
import solver_my_sa
from solver_greedy import distance


# calculate the total distance of the tour
def total_dist(cities, tour):
    N = len(tour)
    total_distance = 0
    for i in range(N):
        total_distance += distance(cities[tour[i % N]], cities[tour[(i + 1) % N]])
    return total_distance


# 2-opt after SA
def solve(cities):
    N = len(cities)
    best_distance = float('Inf')
    best_tour = None

    for _ in range(5):
        # initialize with the SA tour
        tour = solver_my_sa.solve(cities)
        
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

        dist = total_dist(cities, tour)
        if dist < best_distance:
            best_distance = dist
            best_tour = tour

    #print('the distance so far is {}'.format(best_distance))
    #print('------final SA process------')
    #best_tour = solver_my_sa.solve_with_initial(cities, best_tour, alpha=0.9999999, stopping_T=1e-100, stopping_iter=10000)
    #best_distance = total_dist(cities, best_tour)

    print('the final distance is {}'.format(best_distance))
    return best_tour, best_distance


if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)
    