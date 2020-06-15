# Google STEP Program week5 homework
# Tabu Search Algorithm

import sys
import math
import copy
from common import print_tour, read_input
import solver_greedy



class TS():
    def __init__(self, cities, initial_tour=None):
        self.cities = cities
        self.N = len(cities)
        self.dict_of_neighbours = self.generate_neighbours()
        self.n_opt = 1
        self.initial_tour = initial_tour


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


    # geenrates a distance matrix between all cities
    def generate_neighbours(self):
        dict_of_neighbours = {}

        for i in range(self.N):
            for j in range(i+1, self.N):
                if i not in dict_of_neighbours:
                    dict_of_neighbours[i] = {}
                    dict_of_neighbours[i][j]= self.distance(i, j)
                else:
                    dict_of_neighbours[i][j] = self.distance(i, j)
                    # dict_of_neighbours[i] = sorted(dict_of_neighbours[i].items(), key=lambda kv: kv[1])
                if j not in dict_of_neighbours:
                    dict_of_neighbours[j] = {}
                    dict_of_neighbours[j][i] = self.distance(j, i)
                else:
                    dict_of_neighbours[j][i] = self.distance(j, i)
                    # dict_of_neighbours[i] = sorted(dict_of_neighbours[i].items(), key=lambda kv: kv[1])

        return dict_of_neighbours


    # generate first solution by greedy algorithm
    def generate_first_tour(self):
        first_tour = solver_greedy.solve(self.cities)
        distance = self.total_dist(first_tour)
        return first_tour, distance


    def find_neighborhood(self, tour):
        neighborhood_of_solution = []
        for n in tour[1:-self.n_opt]:
            idx1 = []
            n_index = tour.index(n)
            for i in range(self.n_opt):
                idx1.append(n_index + i)

            for kn in tour[1:-self.n_opt]:
                idx2 = []
                kn_index = tour.index(kn)
                for i in range(self.n_opt):
                    idx2.append(kn_index + i)
                if bool(
                    set(tour[idx1[0]:(idx1[-1]+1)]) &
                    set(tour[idx2[0]:(idx2[-1]+1)])):
            
                    continue
            

                _tmp = copy.deepcopy(tour)
                for i in range(self.n_opt):
                    _tmp[idx1[i]] = tour[idx2[i]]
                    _tmp[idx2[i]] = tour[idx1[i]]

                distance = 0
                for k in _tmp[:-1]:
                    next_node = _tmp[_tmp.index(k) + 1]
                    distance = distance + self.dict_of_neighbours[k][next_node]
                    
                _tmp.append(distance)
                if _tmp not in neighborhood_of_solution:
                    neighborhood_of_solution.append(_tmp)

        indexOfLastItemInTheList = len(neighborhood_of_solution[0]) - 1

        neighborhood_of_solution.sort(key=lambda x: x[indexOfLastItemInTheList])
        return neighborhood_of_solution



    def tabu_search(self, iters=100, size=10):
        if self.initial_tour is None:
            first_solution, distance_of_first_solution = self.generate_first_tour()
        else:
            first_solution = self.initial_tour
            distance_of_first_solution = self.total_dist(first_solution)
        count = 1
        solution = first_solution
        tabu_list = []
        best_cost = distance_of_first_solution
        best_solution_ever = solution
        while count <= iters:
            neighborhood = self.find_neighborhood(solution)
            index_of_best_solution = 0
            best_solution = neighborhood[index_of_best_solution]
            best_cost_index = len(best_solution) - 1
            found = False
            while found is False:
                i = 0
                first_exchange_node, second_exchange_node = [], []
                n_opt_counter = 0
                while i < len(best_solution):
                    if best_solution[i] != solution[i]:
                        first_exchange_node.append(best_solution[i])
                        second_exchange_node.append(solution[i])
                        n_opt_counter += 1
                        if n_opt_counter == self.n_opt:
                            break
                    i = i + 1

                exchange = first_exchange_node + second_exchange_node
                if first_exchange_node + second_exchange_node not in tabu_list and second_exchange_node + first_exchange_node not in tabu_list:
                    tabu_list.append(exchange)
                    found = True
                    solution = best_solution[:-1]
                    cost = neighborhood[index_of_best_solution][best_cost_index]
                    if cost < best_cost:
                        best_cost = cost
                        best_solution_ever = solution
                elif index_of_best_solution < len(neighborhood):
                    best_solution = neighborhood[index_of_best_solution]
                    index_of_best_solution = index_of_best_solution + 1

            while len(tabu_list) > size:
                tabu_list.pop(0)

            count = count + 1
        #best_solution_ever.pop(-1)
        #print(best_cost)
        #print(best_solution_ever)
        print(self.total_dist(best_solution_ever))
        return best_solution_ever


# solve function
def solve(cities):
    ts = TS(cities)
    tour = ts.tabu_search()
    return tour


def solve_with_initial(cities, initial_tour):
    ts = TS(cities, initial_tour=initial_tour)
    tour = ts.tabu_search()
    return tour



if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = solve(cities)
    #print_tour(tour)