# Google STEP Program week5 homework
# generate my output files

from common import format_tour, read_input

import solver_2_opt
import solver_my_sa
import solver_sa_and_2opt

CHALLENGES = 7


def generate_output():
    for i in range(CHALLENGES):
        cities = read_input(f'input_{i}.csv')
        solver = solver_sa_and_2opt
        name = 'sa_&_2opt'
        #solver = solver_2_opt
        #name = '2_opt'
        tour = solver.solve(cities)
        with open(f'my_output/{name}_{i}.csv', 'w') as f:
            f.write(format_tour(tour) + '\n')
        '''
        for solver, name in ((solver_2_opt, '2_opt')):
            tour = solver.solve(cities)
            with open(f'my_output/{name}_{i}.csv', 'w') as f:
                f.write(format_tour(tour) + '\n')'''


if __name__ == '__main__':
    generate_output()