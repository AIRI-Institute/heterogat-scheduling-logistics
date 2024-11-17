import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import numpy.random as random
import torch
import dgl
from dgl.data import DGLDataset
from itertools import product


ss_type = ('s', 'ss', 's')
os_type = ('o', 'os', 's')
so_type = ('s', 'so', 'o')


def graph_from_problem(problem, gamma=None, max_operations=None):
    n_tasks = problem['n_tasks']
    n_operations = problem['n_operations']
    operation = problem['operation']
    dist = problem['dist'] * problem['transportation_cost'][0]
    time_cost = problem['time_cost']
    op_cost = problem['op_cost']
    productivity = problem['productivity']
    if max_operations is None:
        max_operations = n_operations

    operation_index = []
    for i in range(n_tasks):
        for j in range(n_operations):
            if operation[j, i] == 1:
                operation_index.append((i, j))
    operation_index = np.array(operation_index)
    
    adj_operation = np.zeros((operation_index.shape[0], operation_index.shape[0]))
    for i in range(n_tasks):
        col_i = operation[:, i]
        path = np.where(col_i > 0)[0]
        for j in range(len(path) - 1):
            u = operation_index.tolist().index([i, path[j]])
            v = operation_index.tolist().index([i, path[j+1]])
            adj_operation[u, v] = 1

    full_time_cost = np.tile(time_cost, (n_tasks, 1))
    full_time_cost = full_time_cost[operation.T.reshape(-1).astype(bool)]

    full_op_cost = np.tile(op_cost, (n_tasks, 1))
    full_op_cost = full_op_cost[operation.T.reshape(-1).astype(bool)]

    graph_data = {
        ss_type: np.where(dist > 0),
        os_type: np.where(full_op_cost < 999),
        so_type: np.where(full_op_cost < 999)[::-1],
        ('o', 'forward', 'o'): np.where(adj_operation > 0),
        ('o', 'backward', 'o'): np.where(adj_operation > 0)[::-1],
    }
    g = dgl.heterograph(graph_data)
    g = dgl.add_self_loop(g, etype='ss')

    op_feat = torch.zeros(len(operation_index), max_operations)
    op_feat[range(len(operation_index)), operation_index[:, 1]] = 1

    g.ndata['feat'] = {
        'o': torch.FloatTensor(op_feat),
        's': torch.FloatTensor(productivity[:, None])
    }
    g.ndata['operation_index'] = {
        'o': torch.LongTensor(operation_index),
    }
    u_idx, v_idx = g.edges(etype='os')
    serves_feat = np.array([
        full_op_cost[u_idx, v_idx],
        full_time_cost[u_idx, v_idx],
    ])
    g.edata['feat'] = {
        'os': torch.FloatTensor(serves_feat.T),
        'ss': torch.FloatTensor(dist[g.edges(etype='ss')][:, None]),
    }
    g.edata['_feat'] = {
        'os': torch.FloatTensor(serves_feat.T),
        'ss': torch.FloatTensor(dist[g.edges(etype='ss')][:, None]),
    }

    target = []
    for full_o, c in zip(*np.where(full_op_cost < 999)):
        t, o = operation_index[full_o]
        if gamma is not None:
            target.append(gamma[o, t, c])
        else:
            target.append(0)
    g.edata['target'] = {
        'os': torch.FloatTensor(target)[:, None],
    }
    return g


def gamma_from_target(target, graph, problem):
    target_mask = target[:, 0] == 1
    u, v = graph.edges(etype=os_type)
    u, v = u[target_mask], v[target_mask]
    u = graph.ndata['operation_index']['o'][u]
    gamma = np.zeros((problem['n_operations'], problem['n_tasks'], problem['n_cities']))
    for i in range(len(u)):
        operation, task, city = u[i, 1], u[i, 0], v[i]
        gamma[operation, task, city] = 1
    return gamma


def delta_from_gamma(problem, gamma):
    n_cities = problem['n_cities']
    n_operations = problem['n_operations']
    n_tasks = problem['n_tasks']

    delta = np.zeros((1, n_cities, n_cities, n_operations - 1, n_tasks))
    for t in range(n_tasks):
        o_iter, c_iter = np.where(gamma[:, t] == 1)
        for i in range(len(o_iter)-1):
            o = o_iter[i]
            c_u, c_v = c_iter[i], c_iter[i+1]
            delta[0, c_u, c_v, o, t] = 1
    return delta


class GraphDataset(DGLDataset):
    def __init__(self, graphs):
        super().__init__(name='custom_dataset')
        self.graphs = graphs
        self.ids = torch.arange(len(graphs))
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.ids[idx]
    
    def __len__(self):
        return len(self.graphs)


def total_cost_from_gamma(problem, gamma, delta):
    time_cost = problem['time_cost']
    op_cost = problem['op_cost']
    productivity = problem['productivity']
    transportation_cost = problem['transportation_cost']
    dist = problem['dist']

    total_op_cost = np.sum(
        (time_cost * op_cost / productivity[None, :])[:, None, :] * gamma
    )
    total_logistic_cost = np.sum(
        (transportation_cost[:, None, None] * dist[None, ...])[..., None, None] * delta
    )
    return total_op_cost + total_logistic_cost

def total_cost_from_graph(graph, pred, transportation_cost=0.3):
    mask = pred.bool()[:, 0]
    o, s = graph.edges(etype=os_type)
    o, s = o[mask], s[mask]
    edata_feat = graph.edata['_feat'][os_type][mask]
    productivity = graph.dstdata['feat']['s'][s][:, 0]
    op_cost = edata_feat[:, 0]
    time_cost = edata_feat[:, 1]
    total_op_cost = sum(time_cost * op_cost / productivity)

    total_logistic_cost = 0
    for task in set(graph.ndata['operation_index']['o'][o, 0].numpy()):
        route = s[graph.ndata['operation_index']['o'][o, 0] == task]
        route_ids = graph.edge_ids(route[:-1], route[1:], etype=ss_type)
        dist = graph.edata['_feat'][ss_type][route_ids]
        total_logistic_cost += (dist * transportation_cost).sum()
    
    return total_op_cost + total_logistic_cost


def check_feasibility(gamma, delta, problem):
    n_operations = problem['n_operations']
    n_tasks = problem['n_tasks']
    operation = problem['operation']
    n_cities = problem['n_cities']
    for i, k in product(range(n_operations), range(n_tasks)):
        assert sum(gamma[i, k]) == operation[i, k]
    for i, k, m, m_ in product(
        range(n_operations-1), range(n_tasks), range(n_cities), range(n_cities)):
        seq = np.where(operation[i:, k] == 1)[0]
        if operation[i, k] and len(seq) > 1:
            assert gamma[i, k, m] + gamma[i+seq[1], k, m_] - 1 <= sum(delta[:, m, m_, i, k])


def read_fatahi_dataset(path_to_file, sheet_names=None):
    """
    Fatahi Valilai, Omid. “Dataset for Logistics and Manufacturing 
    Service Composition”. 17 Mar. 2021. Web. 9 June 2023.
    """
    if sheet_names is None:
        sheet_names = [
            '5,10,10-1',
            '5,10,10-2',
            '5,10,10-3',
            '10,10,10-1',
            '10,10,10-2',
            '10,10,10-3',
            '5,10,20-1',
            '5,10,20-2',
            '5,10,20-3',
            '5,20,10-1',
            '5,20,10-2',
            '5,20,10-3',
            '5,20,20-1',
            '5,20,20-2',
            '5,20,20-3',
            '5,5,5-1',
            '5,5,5-2',
            '5,5,5-3',
        ]
    res = []
    for sheet_name in tqdm(sheet_names, desc='Reading the dataset...'):
        res.append(_read_sheet(path_to_file, sheet_name))
    return res


def _read_sheet(path_to_file, sheet_name):
    n_services = 1
    n_tasks, n_operations, n_cities, _ = list(
            map(int, '-'.join(sheet_name.split(',')).split('-'))
        )
    operation = np.zeros((n_operations, n_tasks))
    dist = np.zeros((n_cities, n_cities))
    time_cost = np.zeros((n_operations, n_cities))
    op_cost = np.zeros((n_operations, n_cities))
    productivity = np.zeros((n_cities))
    transportation_cost = np.zeros((n_services))

    operation[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_tasks+1),
        skiprows=5,
        nrows=n_operations,
    )
    dist[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_cities+1), 
        skiprows=5*2+n_operations-1,
        nrows=n_cities,
    )
    time_cost[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_cities+1), 
        skiprows=5*3+n_operations+n_cities-1*2,
        nrows=n_operations,
    )
    time_cost[np.isinf(time_cost)] = 999
    op_cost[:, :] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(1, n_cities+1), 
        skiprows=5*4+n_operations+n_cities+n_operations-1*3,
        nrows=n_operations,
    )
    op_cost[np.isinf(op_cost)] = 999
    productivity[:] = pd.read_excel(
        path_to_file, 
        sheet_name=sheet_name, 
        header=None, 
        usecols=range(n_cities), 
        skiprows=5*5+n_operations+n_cities+n_operations+n_operations-1*4,
        nrows=1,
    )
    transportation_cost[:] = [0.3]
    return {
        'name': sheet_name,
        'n_tasks': n_tasks, 
        'n_operations': n_operations, 
        'n_cities': n_cities,
        'n_services': n_services,
        'operation': operation,
        'dist': dist,
        'time_cost': time_cost,
        'op_cost': op_cost,
        'productivity': productivity,
        'transportation_cost': transportation_cost,
    }


def sample_problem(
        n_tasks, 
        n_operations, 
        n_cities, 
        threshold=0.5, 
        max_iters=1000, 
        dirpath='../data/',
        random_seed=None):
    assert 0 < n_tasks
    assert 0 < n_operations < 21
    assert 0 < n_cities < 21
    if random_seed is not None:
        random.seed(random_seed)
    for i in range(max_iters):
        operation = random.rand(n_operations, n_tasks) > threshold
        operation = operation.astype(int)
        if np.all(operation.sum(axis=0) > 0):
            break
    assert np.all(operation.sum(axis=0) > 0)
    dist = np.load(f'{dirpath}dist.npy')[:n_cities, :n_cities]
    time_cost = np.load(f'{dirpath}time_cost.npy')[:n_operations, :n_cities]
    op_cost = np.load(f'{dirpath}op_cost.npy')[:n_operations, :n_cities]
    productivity = np.load(f'{dirpath}productivity.npy')[:n_cities]
    transportation_cost = np.array([0.3])
    return {
        'name': f'{n_tasks},{n_operations},{n_cities}',
        'n_tasks': n_tasks, 
        'n_operations': n_operations, 
        'n_cities': n_cities,
        'n_services': 1,
        'operation': operation,
        'dist': dist,
        'time_cost': time_cost,
        'op_cost': op_cost,
        'productivity': productivity,
        'transportation_cost': transportation_cost,
    }


def sample_dataset(
        n_problems, 
        n_tasks_range=[5,10], 
        n_operations_range=[5,20], 
        n_cities_range=[5,20], 
        threshold=0.5, 
        max_iters=1000, 
        dirpath='../data/',
        random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    problems = []
    for i in range(n_problems):
        n_tasks = random.randint(n_tasks_range[0], n_tasks_range[1]+1)
        n_operations = random.randint(n_operations_range[0], n_operations_range[1]+1)
        n_cities = random.randint(n_cities_range[0], n_cities_range[1]+1)
        problem = sample_problem(n_tasks, n_operations, n_cities, threshold, max_iters, dirpath)
        problem['name'] = problem['name']+f'-{i+1}'
        problems.append(problem)
    return problems
