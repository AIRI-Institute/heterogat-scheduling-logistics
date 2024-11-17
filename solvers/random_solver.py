import numpy as np


def random_solve(problem):
    n_cities = problem["n_cities"]
    n_operations = problem["n_operations"]
    n_tasks = problem["n_tasks"]
    op_cost = problem['op_cost']
    operation = problem['operation']

    gamma = np.zeros((n_operations, n_tasks, n_cities))
    for i in range(n_operations):
        for j in range(n_tasks):
            random_city = np.random.choice(np.where(op_cost[i] < 999)[0])
            gamma[i, j, random_city] = 1
    gamma = gamma * operation[..., None]
    return gamma
