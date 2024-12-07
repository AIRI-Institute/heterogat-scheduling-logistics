from solvers.random_solver import random_solve
from solvers.greedy_solver import greedy_solve
from solvers.gnn_solver import gnn_solve
from utils import read_fatahi_dataset, delta_from_gamma, total_cost_from_gamma
import numpy as np
from tqdm.auto import tqdm
import logging
logging.basicConfig(level=logging.INFO, filename='results.log', format='%(asctime)s %(levelname)-8s %(message)s')

if __name__ == '__main__':
    logging.info('Reading the LMSC dataset...')
    dataset = read_fatahi_dataset('data/lmsc/dataset.xlsx')
    logging.info('Solving the problems...')
    for solver in ['optimal', 'random', 'greedy', 'gnn']:
        for problem in tqdm(dataset, desc=f'Solver: {solver}. Solving the problems...'):
            if solver == 'optimal':
                gamma = np.load(f'data/lmsc/optimal/{problem["name"]}/gamma.npy')
            if solver == 'random':
                gamma = random_solve(problem)
            if solver == 'greedy':
                gamma = greedy_solve(problem)
            if solver == 'gnn':
                n_tasks, n_operations, n_cities = problem["n_tasks"], problem["n_operations"], problem["n_cities"]
                out_dim = 16 if n_operations == 5 else 32
                n_layers = 1 if n_operations == 5 else 3
                gamma = gnn_solve(
                    problem,
                    path_to_ckpt=f'checkpoints/gnn-{n_tasks}-{n_operations}-{n_cities}.ckpt',
                    n_operations=n_operations,
                    out_dim=out_dim, 
                    n_layers=n_layers, 
                    n_iterations=50,
                    random_seed=0,
                )
            delta = delta_from_gamma(problem, gamma)
            cost = total_cost_from_gamma(problem, gamma, delta)
            logging.info(f'Solver: {solver}, Problem: {problem["name"]}, Total cost: {cost:.2f}')
