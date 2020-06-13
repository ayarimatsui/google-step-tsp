# Google STEP Program week5 homework
# 2-opt法

import sys
import math
from common import print_tour, read_input
import solver_greedy
from solver_greedy import distance

#2-opt法
def solve(cities):
    N = len(cities)
    tour = solver_greedy.solve(cities)
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


if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)
    print_tour(tour)