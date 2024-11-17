from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from dgl.dataloading import GraphDataLoader
from solvers.graph_model import AttentionGNN
from utils import sample_dataset, graph_from_problem, os_type, ss_type, GraphDataset
from tqdm.auto import tqdm
import numpy as np

def prepare_train_val_dataset(n_tasks, n_operations, n_cities, path_to_dataset):
    dataset = sample_dataset(
        100, 
        [n_tasks, n_tasks],
        [n_operations, n_operations],
        [n_cities, n_cities],
        threshold=0.2,
        dirpath=f'{path_to_dataset}/{n_tasks}-{n_operations}-{n_cities}/',
        random_seed=0
    )
    train_dataset = dataset[:80]
    val_dataset = dataset[80:]
    return train_dataset, val_dataset

def prepare_graphs(dataset, n_tasks, n_operations, n_cities, path_to_dataset):
    graphs = []
    for problem in dataset:
        name = problem['name']
        gamma = np.load(f'{path_to_dataset}/{n_tasks}-{n_operations}-{n_cities}/{name}/gamma.npy')
        graph = graph_from_problem(problem, gamma, max_operations=n_operations)
        graph.edata['feat'][os_type][:, 0] /= 10
        graph.edata['feat'][ss_type][:] /= 100
        graphs.append(graph)
    return graphs

def train():
    cases = [
        [ 5,  5,  5],
        [ 5, 10, 10],
        [10, 10, 10],
        [ 5, 10, 20],
        [ 5, 20, 10],
        [ 5, 20, 20],
    ]

    for (n_tasks, n_operations, n_cities) in tqdm(cases, desc='Training GNN models...'):
        train_dataset, val_dataset = prepare_train_val_dataset(
            n_tasks, n_operations, n_cities, 'data/synthetic')
        train_graphs = prepare_graphs(
            train_dataset, n_tasks, n_operations, n_cities, 'data/synthetic')
        val_graphs = prepare_graphs(
            val_dataset, n_tasks, n_operations, n_cities, 'data/synthetic')

        train_graph_dataset = GraphDataset(train_graphs)
        val_graph_dataset = GraphDataset(val_graphs)

        train_dataloader = GraphDataLoader(train_graph_dataset, batch_size=8, shuffle=True)
        val_dataloader = GraphDataLoader(val_graph_dataset, batch_size=100)

        out_dim = 16 if n_operations == 5 else 32
        n_layers = 1 if n_operations == 5 else 3

        model = AttentionGNN(
            ins_dim=1,
            ino_dim=n_operations,
            out_dim=out_dim,
            n_layers=n_layers,
            lr=0.002,
        )
        trainer = Trainer(
            enable_progress_bar=False,
            max_epochs=100,
            log_every_n_steps=1,
            logger=CSVLogger(f'training_gnn/{n_tasks}-{n_operations}-{n_cities}'),
            accelerator='cpu',
        )
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

if __name__ == '__main__':
    train()
