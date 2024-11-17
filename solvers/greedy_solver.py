import numpy as np


def _solve_task(
    available_operations, cost_operations,
    trans_cost, operation_number,
    gamma
):
    sub_problem_data = []
    for i, stage in enumerate(available_operations):
        if i == 0:
            city = np.argmin(cost_operations[stage])
            sub_problem_data.append([city, cost_operations[stage, city]])
        else:
            cost_total = cost_operations[stage] + trans_cost[:,city,:]
            # Here we calculate the min value of matrix in axis services-cities
            service, city = np.unravel_index(np.argmin(cost_total),
                                            cost_total.shape)
            sub_problem_data.append([city, cost_total[service, city]])
        gamma[stage, operation_number, city] = 1


def greedy_solve(problem):
    n_cities = problem["n_cities"]
    n_operations = problem["n_operations"]
    n_tasks = problem["n_tasks"]
    operations = problem["operation"]
    dist = problem["dist"]
    time_cost = problem["time_cost"]
    op_cost = problem["op_cost"]
    productivity = problem["productivity"]
    transportation_cost = problem["transportation_cost"]

    # Create cost matrices
    cost_operations = time_cost * op_cost / productivity
    trans_cost = dist[None, ...] * transportation_cost

    gamma = np.zeros((n_operations, n_tasks, n_cities))
    
    for task in range(n_tasks):
        available_operations = np.nonzero(operations[:, task])[0]
        _solve_task(
            available_operations, cost_operations,
            trans_cost, task, gamma
        )

    return gamma
