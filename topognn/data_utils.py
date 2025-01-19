"""Utility functions for data sets."""

import csv
from itertools import combinations
import itertools
import math
import os
import pickle
from tqdm import tqdm
import time
import multiprocessing
from numba import vectorize, jit

import torch
import networkx as nx
import numpy as np
import pytorch_lightning as pl

from init import DATA_DIR, Tasks
from cli_utils import str2bool

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.collate import collate
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, Planetoid
from torch_geometric.transforms import BaseTransform, Compose, OneHotDegree
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils.convert import from_networkx
from torch_geometric import transforms

from torch_scatter import scatter

from torch.utils.data import random_split, Subset

from sklearn.model_selection import StratifiedKFold, train_test_split

from simplicial_utils import pyg_to_simplex_tree


def dataset_map_dict():
    DATASET_MAP = {
        'IMDB-BINARY': IMDB_Binary,
        'IMDB-MULTI': IMDB_Multi,
        'REDDIT-BINARY': REDDIT_Binary,
        'REDDIT-5K': REDDIT_5K,
        'PROTEINS': Proteins,
        'PROTEINS_full': Proteins_full,
        'ENZYMES': Enzymes,
        'DD': DD,
        'NCI1' : NCI,
        'MUTAG': MUTAG,
        'MNIST': MNIST,
        'CIFAR10': CIFAR10,
        'PATTERN': PATTERN,
        'CLUSTER': CLUSTER,
        'Necklaces': Necklaces,
        'Cycles': Cycles,
        'NoCycles': NoCycles,
        'CliquePlanting': CliquePlanting,
        'DBLP': DBLP,
        'Cora': Cora,
        'CiteSeer' : CiteSeer,
        'PubMed': PubMed,
        'StrongReg': StronglyRegularGraphs,
        #'Cornell': Cornell,
    }

    return DATASET_MAP


def remove_duplicate_edges(batch):

        with torch.no_grad():
            batch = batch.clone()
            device = batch.x.device
            # Computing the equivalent of batch over edges.
            edge_slices = torch.tensor(batch._slice_dict["edge_index"], device= device)
            edge_diff_slices = (edge_slices[1:]-edge_slices[:-1])
            n_batch = len(edge_diff_slices)
            batch_e = torch.repeat_interleave(torch.arange(
                n_batch, device = device), edge_diff_slices)

            correct_idx = batch.edge_index[0] <= batch.edge_index[1]
            #batch_e_idx = batch_e[correct_idx]
            n_edges = scatter(correct_idx.long(), batch_e, reduce = "sum")

            batch.edge_index = batch.edge_index[:,correct_idx]

            new_slices = torch.cumsum(torch.cat((torch.zeros(1,device=device, dtype=torch.long),n_edges)),0).tolist()

            batch._slice_dict["edge_index"] =  new_slices
            return batch


def get_dataset_class(**kwargs):

    if kwargs.get("paired", False):
        # (kwargs["dataset"],batch_size = kwargs["batch_size"], disjoint = not kwargs["merged"] )
        dataset_cls = PairedTUGraphDataset
    else:

        dataset_cls = dataset_map_dict()[kwargs["dataset"]]
    return dataset_cls


class MyData(Data):
    """ Custom Data class so that we can collate batches the right way with
    the 'clique_edge_index' attributes"""
    def __init__(self, data=None):
        """ From PyTorch Geometric data to MyData"""
        super(MyData, self).__init__()
        if data:
            self.edge_index = data.edge_index
            self.x = data.x
            for attr in data.__dict__["_store"]:
                if "triangle" in attr or "simplex" in attr or "clique" in attr or "pos" in attr or "attr" in attr or "complex" in attr:
                    setattr(self, attr, getattr(data, attr))
            if hasattr(data, "weight"):
                self.weight = data.weight
            self.y = data.y

    def __inc__(self, key, value, *args, **kwargs):
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if "clique" in key:
            dim = int(key.split("_")[-3])
            if dim == 2:
                return self.triangle_index.shape[1]
            else:
                return getattr(self, f"simplex_dim_{dim}_index").shape[1]
        else:
            if 'batch' in key:
                return int(value.max()) + 1
            elif 'index' in key or key == 'face':
                return self.num_nodes
            else:
                return 0

# Method 1: parallel un CPU
import multiprocessing
from multiprocessing.pool import ThreadPool
from functools import partial

# ATTENTION: the my_f function needs to be defined outside of any class/method
# so for example, define right at the top of the file right after the imports
def my_f(x, dim):
    (i, c1), (j, c2) = x
    if len(c1 & c2) == dim:
        return [i, j]
    else:
        return None

#@functional_transform('lift_to_simplex')
class LiftToSimplex(BaseTransform):
    def __init__(self, expansion_dim=2):
        assert expansion_dim >= 2
        self.expansion_dim = expansion_dim
        self.counter = 0
        self.dim = 0

    def __call__(self, data):
        st = pyg_to_simplex_tree(data.edge_index, data.num_nodes)
        print(data.edge_index[:, 0:100])
        print(data.num_nodes)
        print(data.edge_index.shape)
        st.expansion(self.expansion_dim)  # Computes the clique complex up to the desired dim.
        complex_dim = st.dimension()  # See what is the dimension of the complex now.

        data = MyData(data)
        data.complex_dim = complex_dim
        indexes = {dim: [[] for _ in range(dim+1)] for dim in range(2, self.expansion_dim+1)}
        for k, _ in st.get_simplices():
            if len(k) > 2 and len(k) <= complex_dim+1:
                for i, s in enumerate(k):
                    indexes[len(k)-1][i].append(s)
                    
        # We save the cliques of each dimension, and the edge index for the clique graph
        print('Graph Num: ', self.counter)
        self.counter += 1
        for dim, idx in indexes.items():
            start_time = time.time()
            self.dim = dim
            # We create a "clique_edge_index", which is the edge index for the clique graph (with cliques of size "dim")
            cliques = list(enumerate(set([idx[j][i] for j in range(dim+1)]) for i in range(len(idx[0]))))
            print(len(cliques))
            # Join cliques by an edge if they share k-1 nodes.
            clique_pairs = combinations(cliques, 2)
            #clique_ei = [ [i, j] for (i, c1), (j, c2) in clique_pairs if len(c1 & c2)==dim ]
            
            
            # the part below goes instead of the current way of computing clique_ei
            #num_call_same_time = 100
            my_call_func = partial(my_f, dim=dim)
            # try both optione below and see what works best
            # Option 1: use multithreading
            #pool = ThreadPool(num_call_same_time)
            # Option 2: use multiprocessing
            pool = multiprocessing.Pool(processes=20)
            responses = pool.map(my_call_func, clique_pairs)
            clique_ei = [x for x in responses if x]
            
            
            fake_edge = False
            if len(clique_ei) > 0:
                clique_edge_index = torch.tensor(clique_ei).transpose(0, 1)
            else: # if there are no "edges" between high-dim cliques we create a fake one for proper batching
                fake_edge = True
                clique_edge_index = torch.zeros((2, 1), dtype=data.edge_index.dtype)
            setattr(data, f"clique_graph_dim_{dim}_edge_index", clique_edge_index)
            
            # We also add an index which contains the cliques of size "dim" (i.e., the nodes of the clique graph above)
            if len(idx[0]) > 0:
                if fake_edge: # if added a fake edge we need to add a fake node; this makes it easier to account for the fake edge added for proper batching
                    for l in idx:
                        l.insert(0, 0)
                new_idx = torch.tensor(idx, dtype=data.edge_index.dtype)
            else:
                 # even if there are no high-dim simplices we still create a fake one (this is needed for proper batching)
                new_idx = torch.tensor([[0] for _ in range(dim+1)], dtype=data.edge_index.dtype)
            if dim == 2:
                data.triangle_index = new_idx
            else:
                setattr(data, f"simplex_dim_{dim}_index", new_idx)

            tt = time.time() - start_time
            print(tt)
        return data


class MakeUndirected(BaseTransform):
    def __call__(self, data):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        return data


class CliquePlantingDataset(InMemoryDataset):
    """Clique planting data set."""

    def __init__(
        self,
        root,
        n_graphs=1000,
        n_vertices=100,
        k_clique=17,
        random_d = 3,
        p_ER_graph = 0.5,
        pre_transform=None,
        transform=None,
        **kwargs
    ):
        """Initialise new variant of clique planting data set.

        Parameters
        ----------
        root : str
            Root directory for storing graphs.

        n_graphs : int
            How many graphs to create.

        n_vertices : int
            Size of graph for planting a clique.

        k : int
            Size of clique. Must be subtly 'compatible' with n, but the
            class will warn if problematic values are being chosen.
        """
        self.n_graphs = n_graphs
        self.n_vertices = n_vertices
        self.k = k_clique
        self.random_d = random_d
        self.p = p_ER_graph

        super().__init__(root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """No raw file names are required."""
        return []

    @property
    def processed_dir(self):
        """Directory to store data in."""
        # Following the other classes, we are relying on the client to
        # provide a proper path.
        return os.path.join(
            self.root,
            'processed'
        )

    @property
    def processed_file_names(self):
        """Return file names for identification of stored data."""
        N = self.n_graphs
        n = self.n_vertices
        k = self.k
        return [f'data_{N}_{n}_{k}.pt']

    def process(self):
        """Create data set and store it in memory for subsequent processing."""
        graphs = [self._make_graph() for i in range(self.n_graphs)]
        labels = [y for _, y in graphs]

        data_list = [from_networkx(g) for g, _ in graphs]
        for data, label in zip(data_list, labels):
            data.y = label
            data.x = torch.randn(data.num_nodes,self.random_d)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _make_graph(self):
        """Create graph potentially containing a planted clique."""
        G = nx.erdos_renyi_graph(self.n_vertices, p=self.p)
        y = 0
        #nx.classes.function.set_node_attributes(G,dict(G.degree),name="degree")

        if np.random.choice([True, False]):
            G = self._plant_clique(G, self.k)
            y = 1

        return G, y

    def _plant_clique(self, G, k):
        """Plant $k$-clique in a given graph G.

        This function chooses a random subset of the vertices of the graph and
        turns them into fully-connected subgraph.
        """
        n = G.number_of_nodes()
        vertices = np.random.choice(np.arange(n), k, replace=False)

        for index, u in enumerate(vertices):
            for v in vertices[index+1:]:
                G.add_edge(u, v)

        return G


class StronglyRegularGraphs(pl.LightningDataModule):
    task = Tasks.GRAPH_CLASSIFICATION

    def __init__(
        self,
        name,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        num_workers=4,
        add_node_degree=True,
        lift_to_simplex=False,
        max_simplex_dim=2,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.add_node_degree = add_node_degree
        self.lift_to_simplex = lift_to_simplex
        self.max_simplex_dim = max_simplex_dim
        self.num_classes = 3 # dummy value, there are no classes
        #self.prepare_data_per_node = 0

        self.pre_transform_list = []
        if self.add_node_degree:
            self.pre_transform_list.append(OneHotDegree(20))
            self.node_attributes = 21
        if self.lift_to_simplex:
            self.pre_transform_list.append(LiftToSimplex(self.max_simplex_dim))
        self.pre_transform = None if len(self.pre_transform_list)==0 else Compose(self.pre_transform_list)

    def prepare_data(self):
        """Load or create data set according to the provided parameters."""
        root=os.path.join(DATA_DIR, 'strongly_regular', self.name)
        processed_file = os.path.join(root, "processed.pt")
        if os.path.exists(processed_file):
            self._data_list = torch.load(processed_file)
        else:
            pyg_graphs = []
            for graph_file_name in os.listdir(root):
                file_path = os.path.join(root, graph_file_name)
                edges = []
                with open(file_path) as fg:
                    for line in fg:
                        edges.append(list(map(int, line.split(" "))))
                ei = torch.tensor(edges).transpose(0, 1)
                pyg_graph = Data(edge_index=ei, y=torch.tensor(0)) # add fake 0 label for making code work
                pyg_graphs.append(pyg_graph)

            if self.pre_transform is not None:
                pyg_graphs = [self.pre_transform(d) for d in pyg_graphs]

            self._data_list = pyg_graphs
            #data, slices, _ = collate(
                #pyg_graphs[0].__class__,
                #data_list=pyg_graphs,
                #increment=False,
                #add_batch=False,
            #)
            torch.save(self._data_list, processed_file)

        self.batch_size = len(self._data_list)
        self.train = self._data_list
        self.val = self._data_list
        self.test = self._data_list

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=bool, default=True)
        parser.add_argument('--add_node_degree', type=bool, default=True)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--name',type=str, default="cubic12")
        parser.add_argument('--lift_to_simplex', type=str2bool, default=False)
        parser.add_argument('--max_simplex_dim', type=int, default=2)
        return parser


class SyntheticBaseDataset(InMemoryDataset):
    def __init__(self, root=DATA_DIR, transform=None, pre_transform=None, **kwargs):
        super(SyntheticBaseDataset, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graphs.txt', 'labels.pt']

    @property
    def processed_file_names(self):
        return ['synthetic_data.pt']

    def process(self):
        # Read data into huge `Data` list.
        with open(f"{self.root}/graphs.txt", "rb") as fp:   # Unpickling
            x_list, edge_list = pickle.load(fp)

        labels = torch.load(f"{self.root}/labels.pt")
        data_list = [Data(x=x_list[i], edge_index=edge_list[i],
                          y=labels[i][None]) for i in range(len(x_list))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SyntheticDataset(pl.LightningDataModule):
    task = Tasks.GRAPH_CLASSIFICATION

    def __init__(
        self,
        name,
        batch_size,
        use_node_attributes=True,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        num_workers=4,
        add_node_degree=False,
        dataset_class=SyntheticBaseDataset,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.dataset_class = dataset_class
        self.kwargs = kwargs

        if add_node_degree:
            self.pre_transform = OneHotDegree(max_degrees[name])
        else:
            self.pre_transform = None

    def prepare_data(self):
        """Load or create data set according to the provided parameters."""
        dataset = self.dataset_class(
            root=os.path.join(DATA_DIR, 'SYNTHETIC', self.name),
            pre_transform=self.pre_transform,
            **self.kwargs
        )

        self.node_attributes = dataset.num_node_features
        self.num_classes = dataset.num_classes
        n_instances = len(dataset)
        n_train = math.floor(
            (1 - self.val_fraction) * (1 - self.test_fraction) * n_instances)
        n_val = math.ceil(
            (self.val_fraction) * (1 - self.test_fraction) * n_instances)
        n_test = n_instances - n_train - n_val

        self.train, self.val, self.test = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=bool, default=True)
        #parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--min_cycle',type=int,default = 3)
        parser.add_argument('--k_clique',type=int, default = 17)
        parser.add_argument('--p_ER_graph',type=float, default = 0.5, help = "Probability of an edge in the ER graph (only for CliquePlanting)")
        #parser.add_argument('--benchmark_idx',type=str2bool,default=True,help = "If True, uses the idx from the graph benchmarking paper.")
        return parser


def get_label_fromTU(dataset):
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i].y)
    return labels


def get_degrees_fromTU(name):

    dataset = TUDataset(
            root=os.path.join(DATA_DIR, name),
            use_node_attr=True,
            cleaned=True,
            name = name)
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]


    deg = torch.cat(degs, dim=0).to(torch.float)
    mean, std = deg.mean().item(), deg.std().item()

    print(f"Mean of degree of {name} = {mean} with std : {std}")

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class RandomAttributes(object):
    def __init__(self,d):
        self.d = d
    def __call__(self,data):
        data.x = torch.randn((data.x.shape[0],self.d))
        return data

class TUGraphDataset(pl.LightningDataModule):
    #task = Tasks.GRAPH_CLASSIFICATION

    def __init__(self, name, batch_size, use_node_attributes=True,
                 val_fraction=0.1, test_fraction=0.1, fold=0, seed=42,
                 num_workers=2, n_splits=5, legacy=True,
                 lift_to_simplex=False, max_simplex_dim=2, **kwargs):

        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.use_node_attributes = use_node_attributes
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.legacy = legacy
        self.lift_to_simplex = lift_to_simplex
        self.max_simplex_dim = max_simplex_dim

        if name == "DBLP_v1":
            self.task = Tasks.NODE_CLASSIFICATION_WEIGHTED
        else:
            self.task = Tasks.GRAPH_CLASSIFICATION

        max_degrees = {"IMDB-BINARY": 540,
                "COLLAB": 2000, 'PROTEINS': 50, 'ENZYMES': 18, "REDDIT-BINARY": 12200, "REDDIT-MULTI-5K":8000, "IMDB-MULTI":352}
        mean_degrees = {"REDDIT-BINARY":2.31,"REDDIT-MULTI-5K":2.34}
        std_degrees  = {"REDDIT-BINARY": 20.66,"REDDIT-MULTI-5K":12.50}

        self.has_node_attributes = use_node_attributes

        self.pre_transform = None

        if not use_node_attributes:
            self.transform = RandomAttributes(d=3)
            #self.transform = None
        else:
            if name in ['IMDB-BINARY','IMDB-MULTI','REDDIT-BINARY','REDDIT-MULTI-5K']:
                self.max_degree = max_degrees[name]
                if self.max_degree < 1000:
                    self.pre_transform = OneHotDegree(self.max_degree)
                    self.transform = None
                else:
                    self.transform = None
                    self.pre_transform = NormalizedDegree(mean_degrees[name],std_degrees[name])

            else:
                self.transform = None

        self.n_splits = n_splits
        self.fold = fold

        if name in ["PROTEINS_full", "ENZYMES", "DD"]:
            self.benchmark_idx = kwargs["benchmark_idx"]
        else:
            self.benchmark_idx = False

        if lift_to_simplex:
            if self.pre_transform is not None:
                #### Davide note: Compose can create issues if using DGL datasets () see tu_datasets.py)
                self.pre_transform = Compose([self.pre_transform, LiftToSimplex(self.max_simplex_dim)])
            else:
                self.pre_transform = LiftToSimplex(self.max_simplex_dim)

    def prepare_data(self):
        from tu_datasets import PTG_LegacyTUDataset

        if self.name == "PROTEINS_full" or self.name == "ENZYMES":
            cleaned = False
        else:
            cleaned = True


        if self.legacy:
            dataset = PTG_LegacyTUDataset(
                root=os.path.join(DATA_DIR, self.name + '_legacy'),
                # use_node_attr=self.has_node_attributes,
                # cleaned=cleaned,
                name=self.name,
                transform=self.transform,
                simplicial_pre_transform=None if not self.lift_to_simplex else LiftToSimplex(self.max_simplex_dim),
                simplicial_dim=None if not self.lift_to_simplex else self.max_simplex_dim
            )
            self.node_attributes = dataset[0].x.shape[1]
        else:
            dataset = TUDataset(
                root=os.path.join(DATA_DIR, self.name),
                use_node_attr=self.has_node_attributes,
                cleaned=cleaned,
                name=self.name,
                transform=self.transform,
                pre_transform = self.pre_transform
            )

            if self.has_node_attributes:
                self.node_attributes= dataset.num_node_features
            else:
                if self.max_degree<1000:
                    self.node_attributes = self.max_degree+1
                else:
                    self.node_attributes = dataset.num_node_features

        self.num_classes = dataset.num_classes

        if self.benchmark_idx:
            all_idx = {}
            for section in ['train', 'val', 'test']:
                with open(os.path.join(DATA_DIR, 'Benchmark_idx', self.name+"_"+section+'.index'), 'r') as f:
                    reader = csv.reader(f)
                    all_idx[section] = [list(map(int, idx)) for idx in reader]
            train_index = all_idx["train"][self.fold]
            val_index = all_idx["val"][self.fold]
            test_index = all_idx["test"][self.fold]
        else:
            n_instances = len(dataset)

            skf = StratifiedKFold(n_splits=self.n_splits,
                                  random_state=self.seed, shuffle=True)

            skf_iterator = skf.split(
                torch.tensor([i for i in range(n_instances)]), torch.tensor(get_label_fromTU(dataset)))


            train_index, test_index = next(
                itertools.islice(skf_iterator, self.fold, None))
            train_index, val_index = train_test_split(
                train_index, random_state=self.seed)

            train_index = train_index.tolist()
            val_index = val_index.tolist()
            test_index = test_index.tolist()

        self.train = Subset(dataset, train_index)
        self.val = Subset(dataset, val_index)
        self.test = Subset(dataset, test_index)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--legacy', type=str2bool, default=True)
        parser.add_argument('--benchmark_idx', type=str2bool, default=True,
                            help="If True, uses the idx from the graph benchmarking paper.")
        parser.add_argument('--lift_to_simplex', type=str2bool, default=False)
        parser.add_argument('--max_simplex_dim', type=int, default=2)
        return parser


class PairedTUGraphDatasetBase(TUDataset):
    """Pair graphs in TU data set."""

    def __init__(self, name, disjoint, **kwargs):
        """Create new paired graph data set from named TU data set.

        Parameters
        ----------
        name : str
            Name of the TU data set to use as the parent data set. Must
            be a data set with a binary classification task.

        disjoint : bool
            If set, performs a disjoint union between the two graphs
            that are supposed to be paired, resulting in two connected
            components.

        **kwargs : kwargs
            Optional set of keyword arguments that will be used for
            loading the parent TU data set.
        """
        # We only require this for the pairing routine; by default, the
        # disjoint union of graphs will be calculated.
        self.disjoint = disjoint

        if name == "PROTEINS_full" or name == "ENZYMES":
            cleaned = False
        else:
            cleaned = True

        root = os.path.join(DATA_DIR, name)

        super().__init__(name=name, root=root, cleaned=cleaned, **kwargs)

    def _pair_graphs(self):
        """Auxiliary function for performing graph pairing.

        Returns
        -------
        Tuple of data tensor and slices array, which can be saved to the
        disk or used for further processing.
        """
        y = self.data.y.numpy()

        # Some sanity checks before continuing with processing the data
        # set.
        labels = sorted(np.unique(self.data.y))
        n_classes = len(labels)
        if n_classes != 2:
            raise RuntimeError(
                'Paired data set is only defined for binary graph '
                'classification tasks.'
            )

        # Will contain the merged graphs as single `Data` objects,
        # consisting of proper pairings of the respective inputs.
        data = []

        for i, label in enumerate(y):
            partners = np.arange(len(y))
            partners = partners[i < partners]

            for j in partners:

                # FIXME
                #
                # Cannot use `int64` to access the data set. I am
                # reasonably sure that this is *wrong*.
                j = int(j)

                # Merge the two graphs into a single graph with two
                # connected components. This requires merges of all
                # the tensors (except for `y`, which we *know*, and
                # `edge_index`, which we have to merge in dimension
                # 1 instead of 0).

                merged = {}

                # Offset all nodes of the second graph correctly to
                # ensure that we will get new edges and no isolated
                # nodes.
                offset = self[i].num_nodes
                edge_index = torch.cat(
                    (self[i].edge_index, self[j].edge_index + offset),
                    1
                )

                new_label = int(label == y[j])

                # Only graphs whose components stem from the positive
                # class will be accepted here; put *all* other graphs
                # into the negative class.
                if label != 1:
                    new_label = 0

                # Check whether we are dealing with the positive label,
                # i.e. the last of the unique labels, when creating the
                # set of *merged* graphs.
                if not self.disjoint and new_label == 1:
                    u = torch.randint(0, self[i].num_nodes, (1,))
                    v = torch.randint(0, self[j].num_nodes, (1,)) + offset

                    edge = torch.tensor([[u], [v]], dtype=torch.long)
                    edge_index = torch.cat((edge_index, edge), 1)

                merged['edge_index'] = edge_index
                merged['y'] = torch.tensor([new_label], dtype=torch.long)

                for attr_name in dir(self[i]):

                    # No need to merge labels or edge_indices
                    if attr_name == 'y' or attr_name == 'edge_index':
                        continue

                    attr = getattr(self[i], attr_name)

                    if type(attr) == torch.Tensor:
                        merged[attr_name] = torch.cat(
                            (
                                getattr(self[i], attr_name),
                                getattr(self[j], attr_name)
                            ), 0
                        )

                data.append(Data(**merged))

        data, slices = self.collate(data)
        return data, slices

    def download(self):
        """Download data set."""
        super().download()

    @property
    def processed_dir(self):
        """Return name of directory for storing paired graphs."""
        name = 'paired{}{}'.format(
            '_cleaned' if self.cleaned else '',
            '_merged' if not self.disjoint else ''
        )
        return os.path.join(self.root, self.name, name)

    def process(self):
        """Process data set according to input parameters."""
        # First finish everything in the parent data set before starting
        # to pair the graphs and write them out.
        super().process()

        self.data, self.slices = self._pair_graphs()
        torch.save((self.data, self.slices), self.processed_paths[0])


class PairedTUGraphDataset(pl.LightningDataModule):
    task = Tasks.GRAPH_CLASSIFICATION

    def __init__(
        self,
        dataset,
        batch_size,
        use_node_attributes=True,
        merged=False,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        num_workers=4,
        **kwargs
    ):
        """Create new paired data set."""
        super().__init__()

        self.name = dataset
        self.disjoint = not merged
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.use_node_attributes = use_node_attributes

    def prepare_data(self):
        dataset = PairedTUGraphDatasetBase(
            self.name,
            disjoint=self.disjoint,
            use_node_attr=self.use_node_attributes,
        )

        self.node_attributes = dataset.num_node_features
        self.num_classes = dataset.num_classes
        n_instances = len(dataset)

        # FIXME: should this be updated?
        n_train = math.floor(
            (1 - self.val_fraction) * (1 - self.test_fraction) * n_instances)
        n_val = math.ceil(
            (self.val_fraction) * (1 - self.test_fraction) * n_instances)
        n_test = n_instances - n_train - n_val

        self.train, self.val, self.test = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        return parser


class IMDB_Binary(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='IMDB-BINARY', **kwargs)

class IMDB_Multi(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='IMDB-MULTI', **kwargs)

class REDDIT_Binary(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='REDDIT-BINARY', **kwargs)

class REDDIT_5K(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='REDDIT-MULTI-5K', **kwargs)

class Proteins(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='PROTEINS', **kwargs)

class Proteins_full(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='PROTEINS_full', **kwargs)

class Enzymes(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='ENZYMES', **kwargs)

class DD(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='DD', **kwargs)


class MUTAG(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='MUTAG', **kwargs)

class NCI(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='NCI1', **kwargs)

class DBLP(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='DBLP_v1', **kwargs)

class Cycles(SyntheticDataset):
    def __init__(self, min_cycle, **kwargs):
        name = "Cycles" + f"_{min_cycle}"
        super().__init__(name=name, **kwargs)

class NoCycles(SyntheticDataset):
    def __init__(self, **kwargs):
        super().__init__(name="NoCycles", **kwargs)

class Necklaces(SyntheticDataset):
    def __init__(self, **kwargs):
        super().__init__(name="Necklaces", **kwargs)

class CliquePlanting(SyntheticDataset):
    def __init__(self, **kwargs):

        super().__init__(
            name="CliquePlanting",
            dataset_class=CliquePlantingDataset,
            **kwargs
        )

def add_pos_to_node_features(instance: Data):
    instance.x = torch.cat([instance.x, instance.pos], axis=-1)
    return instance


class GNNBenchmark(pl.LightningDataModule):
    def __init__(self, name, batch_size, use_node_attributes, num_workers=4, lift_to_simplex=False, max_simplex_dim=2, **kwargs):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = os.path.join(DATA_DIR, self.name)
        self.lift_to_simplex = lift_to_simplex
        self.max_simplex_dim = max_simplex_dim

        self.pre_transform = None
        self.transforms_list = []
        if name == "MNIST":
            self.transforms_list.append(MakeUndirected())

        if name in ['MNIST', 'CIFAR10']:
            self.task = Tasks.GRAPH_CLASSIFICATION
            self.num_classes = 10
            if use_node_attributes:
                self.transforms_list.append(add_pos_to_node_features)
        elif name == 'PATTERN':
            self.task = Tasks.NODE_CLASSIFICATION_WEIGHTED
            self.num_classes = 2
            if use_node_attributes is False:
                self.transforms_list.append(RandomAttributes(d=3))
        elif name == 'CLUSTER':
            self.task = Tasks.NODE_CLASSIFICATION_WEIGHTED
            self.num_classes = 6
            if use_node_attributes is False:
                self.transforms_list.append(RandomAttributes(d=3))
        else:
            raise RuntimeError('Unsupported dataset')

        if lift_to_simplex:
            if self.pre_transform is not None:
                self.pre_transform = Compose([self.pre_transform, LiftToSimplex(self.max_simplex_dim)])
            else:
                self.pre_transform = LiftToSimplex(self.max_simplex_dim)

        if len(self.transforms_list)>0:
            self.transform  = transforms.Compose(self.transforms_list)
        else:
            self.transform = None

    def prepare_data(self):
        # Just download the data
        train = GNNBenchmarkDataset(
            self.root, self.name, split='train', transform=self.transform, pre_transform=self.pre_transform)

        self.node_attributes = train[0].x.shape[-1]
        GNNBenchmarkDataset(self.root, self.name, split='val', pre_transform=self.pre_transform)
        GNNBenchmarkDataset(self.root, self.name, split='test', pre_transform=self.pre_transform)

    def train_dataloader(self):
        return DataLoader(
            GNNBenchmarkDataset(
                self.root, self.name, split='train', transform=self.transform, pre_transform=self.pre_transform),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            GNNBenchmarkDataset(
                self.root, self.name, split='val', transform=self.transform, pre_transform=self.pre_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            GNNBenchmarkDataset(
                self.root, self.name, split='test', transform=self.transform, pre_transform=self.pre_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        parser.add_argument('--lift_to_simplex', type=str2bool, default=False)
        parser.add_argument('--max_simplex_dim', type=int, default=2)
        return parser


class MNIST(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('MNIST', **kwargs)


class CIFAR10(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('CIFAR10', **kwargs)


class PATTERN(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('PATTERN', **kwargs)


class CLUSTER(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('CLUSTER', **kwargs)

class PlanetoidDataset(pl.LightningDataModule):
    def __init__(self, name, use_node_attributes, num_workers=4, **kwargs):
        super().__init__()
        self.name = name
        self.num_workers = num_workers
        self.root = os.path.join(DATA_DIR, self.name)

        self.task = Tasks.NODE_CLASSIFICATION

        if use_node_attributes:
            self.random_transform = lambda x : x
        else:
            self.random_transform = RandomAttributes(d=3)

    def prepare_data(self):
        # Just download the data
        dummy_data = Planetoid(
                self.root, self.name, split='public', transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_train_transform]))
        self.num_classes = int(torch.max(dummy_data[0].y) + 1)
        self.node_attributes = dummy_data[0].x.shape[1]
        return

    def train_dataloader(self):
        return DataLoader(
            Planetoid(
                    self.root,
                    self.name,
                    split='public',
                    transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_train_transform])
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            Planetoid(
                self.root,
                self.name,
                split='public',
                transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_val_transform])
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            Planetoid(
                self.root,
                self.name,
                split='public',
                transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_test_transform])
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    @staticmethod
    def keep_train_transform(data):
        data.y[~data.train_mask] = -100
        return data

    def keep_val_transform(data):
        data.y[~data.val_mask] = -100
        return data

    def keep_test_transform(data):
        data.y[~data.test_mask] = -100
        return data

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        return parser


class Cora(PlanetoidDataset):
    def __init__(self, **kwargs):
        super().__init__(name='Cora', split = "public", **kwargs)

class CiteSeer(PlanetoidDataset):
    def __init__(self, **kwargs):
        super().__init__(name='CiteSeer', split = "public", **kwargs)

class PubMed(PlanetoidDataset):
    def __init__(self, **kwargs):
        super().__init__(name='PubMed', split = "public", **kwargs)
