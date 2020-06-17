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


    print('the final distance is {}'.format(best_distance))
    return best_tour, best_distance


if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)
    

######## result #######
'''
Challenge 0
output          :    3862.20
sample/random   :    3862.20
sample/greedy   :    3418.10
sample/sa       :    3291.62
my_output/2_opt :    3418.10
my_output/my_sa :    3418.10
my_output/sa_&_2opt:    3291.62

Challenge 1
output          :    6101.57
sample/random   :    6101.57
sample/greedy   :    3832.29
sample/sa       :    3778.72
my_output/2_opt :    3832.29
my_output/my_sa :    3832.29
my_output/sa_&_2opt:    3778.72

Challenge 2
output          :   13479.25
sample/random   :   13479.25
sample/greedy   :    5449.44
sample/sa       :    4494.42
my_output/2_opt :    4994.89
my_output/my_sa :    4494.42
my_output/sa_&_2opt:    4494.42

Challenge 3
output          :   47521.08
sample/random   :   47521.08
sample/greedy   :   10519.16
sample/sa       :    8150.91
my_output/2_opt :    8970.05
my_output/my_sa :    8436.89
my_output/sa_&_2opt:    8118.40

Challenge 4
output          :   92719.14
sample/random   :   92719.14
sample/greedy   :   12684.06
sample/sa       :   10675.29
my_output/2_opt :   11489.79
my_output/my_sa :   11207.26
my_output/sa_&_2opt:   10632.31

Challenge 5
output          :  347392.97
sample/random   :  347392.97
sample/greedy   :   25331.84
sample/sa       :   21119.55
my_output/2_opt :   21363.60
my_output/my_sa :   23681.04
my_output/sa_&_2opt:   20956.88

Challenge 6
output          : 1374393.14
sample/random   : 1374393.14
sample/greedy   :   49892.05
sample/sa       :   44393.89
my_output/2_opt :   42712.37
my_output/my_sa :   49282.53
my_output/sa_&_2opt:   42052.14
'''