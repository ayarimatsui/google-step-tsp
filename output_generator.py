# Google STEP Program week5 homework
# generate my output files

from common import format_tour, read_input

import solver_2_opt
import solver_my_sa
import solver_sa_and_2opt
import solver_DSA

CHALLENGES = 8


def generate_output():
    for i in range(7, CHALLENGES):
        cities = read_input(f'input_{i}.csv')
        solver = solver_DSA
        name = 'dsa'
        #solver = solver_2_opt
        #name = '2_opt'
        record = [3292, 3779, 4495, 8150, 10676, 20273, 40000, 81000]
        tour, distance = solver.solve(cities)
        while distance > record[i]:
            tour, distance = solver.solve(cities)
            if distance <= record[i]:
                break
        with open(f'my_output/{name}_{i}_1.csv', 'w') as f:
            f.write(format_tour(tour) + '\n')
        '''
        for solver, name in ((solver_2_opt, '2_opt')):
            tour = solver.solve(cities)
            with open(f'my_output/{name}_{i}.csv', 'w') as f:
                f.write(format_tour(tour) + '\n')'''


if __name__ == '__main__':
    generate_output()