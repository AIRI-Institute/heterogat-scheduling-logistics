from solvers.graph_model import AttentionGNN, os_type, ss_type
from utils import graph_from_problem, gamma_from_target, delta_from_gamma, total_cost_from_gamma
import numpy as np

def gnn_solve(problem, path_to_ckpt, n_operations, out_dim, n_layers, n_iterations=10):
    graph = graph_from_problem(problem, max_operations=n_operations)
    graph.edata['feat'][os_type][:, 0] /= 10
    graph.edata['feat'][ss_type][:] /= 100

    model = AttentionGNN.load_from_checkpoint(
        checkpoint_path=path_to_ckpt,
        ins_dim=1,
        ino_dim=n_operations,
        out_dim=out_dim,
        n_layers=n_layers,
        lr=0.002,
    )
    model.eval()
    gammas = []
    min_cost = np.inf
    argmin_cost = 0
    for i in range(n_iterations):
        pred = model.predict(graph)
        gamma = gamma_from_target(pred, graph, problem)
        gammas.append(gamma)
        delta = delta_from_gamma(problem, gamma)
        cost = total_cost_from_gamma(problem, gamma, delta).item()
        if min_cost > cost:
            min_cost = cost
            argmin_cost = i
    return gammas[argmin_cost]
