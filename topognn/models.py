import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC

from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data

from init import Tasks
from cli_utils import str2bool, int_or_none
from layers import GCNLayer, GINLayer, GATLayer, GatedGCNLayer, SimpleSetTopoLayer, fake_persistence_computation, persistence_computation, InvariantDeepSet
from metrics import WeightedAccuracy
from data_utils import remove_duplicate_edges

import coord_transforms as coord_transforms
import numpy as np


try: # it's just to be able to test the code on my laptop where I can't install torchph
    from torch_persistent_homology import compute_persistence_homology_batched_mt
    TORCHPH = True
    print("Using TorchPH")
except:
    TORCHPH = False
    print("Not using TorchPH")


class TopologyLayer(torch.nn.Module):
    """Topological Aggregation Layer."""

    def __init__(self, features_in, features_out, num_filtrations,
                 num_coord_funs, filtration_hidden, num_coord_funs1=None,
                 dim1=False, residual_and_bn=False,
                 share_filtration_parameters=False, fake=False,
                 tanh_filtrations=False, relu_filtrations=False, swap_bn_order=False, dist_dim1=False,
                 separate_filtration_functions=False, higher_dims=[], num_coord_funsh=None, dist_dimh=False,
                 clique_persistence=False, mlp_combine_dims_clique_persistence=False):
        """
        num_coord_funs is a dictionary with the numbers of coordinate functions of each type.
        dim1 is a boolean. True if we have to return dim1 persistence.
        """
        super().__init__()

        self.dim1 = dim1

        self.features_in = features_in
        self.features_out = features_out

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.filtration_hidden = filtration_hidden
        self.residual_and_bn = residual_and_bn
        self.share_filtration_parameters = share_filtration_parameters
        self.fake = fake
        self.swap_bn_order = swap_bn_order
        self.dist_dim1 = dist_dim1

        self.separate_filtration_functions = separate_filtration_functions
        self.higher_dims = higher_dims
        self.num_coord_funsh = num_coord_funsh
        self.dist_dimh = dist_dimh
        self.clique_persistence = clique_persistence
        self.mlp_combine_dims_clique_persistence = mlp_combine_dims_clique_persistence
        if mlp_combine_dims_clique_persistence:
            # assumes same dimensionalities for all dimensions
            total_num_coord_funsh = np.array([i for i in num_coord_funsh[self.higher_dims[0]].values()]).sum()
            if self.dist_dimh:
                out_dim = total_num_coord_funsh * self.num_filtrations
                self.mlp_dim_combiner = nn.Sequential(nn.Linear(out_dim*2, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
            else:
                out_dim = total_num_coord_funsh * self.num_filtrations * 2
                self.mlp_dim_combiner = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        self.total_num_coord_funs = np.array(
            list(num_coord_funs.values())).sum()

        self.coord_fun_modules0 = torch.nn.ModuleList([
            getattr(coord_transforms, key)(output_dim=num_coord_funs[key])
            for key in num_coord_funs
        ])

        if self.dim1:
            assert num_coord_funs1 is not None
            self.coord_fun_modules1 = torch.nn.ModuleList([
                getattr(coord_transforms, key)(output_dim=num_coord_funs1[key])
                for key in num_coord_funs1
            ])

        if self.higher_dims:
            assert num_coord_funsh is not None
            self.coord_fun_modulesh = {}
            for dim in self.higher_dims:
                self.coord_fun_modulesh[str(dim)] = torch.nn.ModuleList([ ### ModuleDict requires strings as keys
                    getattr(coord_transforms, key)(output_dim=num_coord_funsh[dim][key])
                    for key in num_coord_funsh[dim]
                ])
            self.coord_fun_modulesh = torch.nn.ModuleDict(self.coord_fun_modulesh)

        if tanh_filtrations:
            final_filtration_activation = nn.Tanh()
        elif relu_filtrations:
            final_filtration_activation = nn.ReLU()
        else:
            final_filtration_activation = nn.Identity()
        if self.share_filtration_parameters:
            self.filtration_modules0 = torch.nn.Sequential(
                torch.nn.Linear(self.features_in, self.filtration_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filtration_hidden, num_filtrations),
                final_filtration_activation
            )
            if self.separate_filtration_functions:
                assert self.dim1
                self.filtration_modulesh = {}
                orders = [1]
                if self.higher_dims:
                    orders += self.higher_dims
                for i in orders:
                    self.filtration_modulesh[str(i)] = torch.nn.Sequential(
                        InvariantDeepSet(self.features_in, num_filtrations, "mean", 2),
                        torch.nn.ReLU(),
                    )
                self.filtration_modulesh = torch.nn.ModuleDict(self.filtration_modulesh)
        else:
            self.filtration_modules0 = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.features_in, self.filtration_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.filtration_hidden, 1),
                    final_filtration_activation
                ) for _ in range(num_filtrations)
            ])
            if self.separate_filtration_functions:
                assert self.dim1
                self.filtration_modulesh = {}
                orders = [1]
                if self.higher_dims:
                    orders += self.higher_dims
                for i in orders:
                    self.filtration_modulesh[str(i)] = torch.nn.ModuleList([
                            torch.nn.Sequential(
                            InvariantDeepSet(self.features_in, 1, "mean", 2),
                            final_filtration_activation,
                        ) for _ in range(num_filtrations)
                    ])
                self.filtration_modulesh = torch.nn.ModuleDict(self.filtration_modulesh)

        if self.residual_and_bn:
            in_out_dim = self.num_filtrations * self.total_num_coord_funs
            features_out = features_in
            self.bn = nn.BatchNorm1d(features_out)
            if self.dist_dim1 and self.dim1:
                self.out1 = torch.nn.Linear(self.num_filtrations * self.total_num_coord_funs, features_out)
            if self.dist_dimh and len(self.higher_dims) > 0:
                self.outh = {}
                for dim in self.higher_dims:
                    self.outh[str(dim)] = torch.nn.Linear(self.num_filtrations * self.total_num_coord_funs, features_out)
                self.outh = torch.nn.ModuleDict(self.outh)
        else:
            if self.dist_dim1 or self.dist_dimh:
                multiplier = 2 if self.dist_dim1 else 1
                if self.dist_dimh:
                    for _ in self.higher_dims:
                        multiplier += 1
                in_out_dim = self.features_in + multiplier * self.num_filtrations * self.total_num_coord_funs
            else:
                in_out_dim = self.features_in + self.num_filtrations * self.total_num_coord_funs
        
        self.out = torch.nn.Linear(in_out_dim, features_out)

    def compute_filtration(self, x, edge_index, higher_dim_indexes=None):
        if self.share_filtration_parameters:
            filtered_v_ = self.filtration_modules0(x)
        else:
            filtered_v_ = torch.cat([filtration_mod.forward(x)
                                     for filtration_mod in self.filtration_modules0], 1)

        filtered_e_, _ = torch.max(torch.stack(
                (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)
        if self.separate_filtration_functions:
            f_in = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1).reshape((edge_index.shape[1], 2, -1))
            if self.share_filtration_parameters:
                filtered_e_ = filtered_e_ + self.filtration_modulesh["1"](f_in)
            else:
                filtered_e_ = filtered_e_ + torch.cat([filtration_mod.forward(f_in)
                                         for filtration_mod in self.filtration_modulesh["1"]], 1)

        filtered_higher = None
        if self.higher_dims and higher_dim_indexes is not None:
            filtered_higher = {}
            for dim in self.higher_dims:
                idx = higher_dim_indexes[dim]
                filtered_dim_h_, _ = torch.max(torch.stack(tuple([filtered_v_[idx[i]] for i in range(idx.shape[0])])), axis=0)
                if self.separate_filtration_functions:
                    f_in = torch.cat([x[idx[i]] for i in range(dim+1)], dim=-1).reshape((idx.shape[1], idx.shape[0], -1))
                    if self.share_filtration_parameters:
                        filtered_higher[dim] = self.filtration_modulesh[str(dim)](f_in)
                    else:
                        filtered_higher[dim] = torch.cat([filtration_mod.forward(f_in)
                                    for filtration_mod in self.filtration_modulesh[str(dim)]], 1)
                    filtered_higher[dim] = filtered_higher[dim] + filtered_dim_h_
                else:
                    filtered_higher[dim] = filtered_dim_h_

        return filtered_v_, filtered_e_, filtered_higher

    def compute_persistence(self, x, batch, return_filtration=False, higher_dim_indexes=None):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        edge_index = batch.edge_index
        vertex_slices = batch._slice_dict['x'].clone().detach()
        edge_slices = torch.Tensor(batch._slice_dict['edge_index']).long()
        if self.higher_dims and higher_dim_indexes is not None:
            higher_order_slices = {}
            clique_edge_index_slices = {}
            for dim in self.higher_dims:
                if dim == 2:
                    idx_name = "triangle_index"
                else:
                    idx_name = f"simplex_dim_{dim}_index"
                higher_order_slices[dim] = torch.LongTensor(batch._slice_dict[idx_name])
                clique_edge_index_slices[dim] = torch.LongTensor(batch._slice_dict[f"clique_graph_dim_{dim}_edge_index"])

        filtered_v_, filtered_e_, filtered_higher = self.compute_filtration(x, edge_index, higher_dim_indexes)

        if self.fake:
            return fake_persistence_computation(filtered_v_, edge_index, vertex_slices, edge_slices, batch.batch), None
        else:
            if self.clique_persistence:
                if not TORCHPH:
                    p_og = fake_persistence_computation(filtered_v_, edge_index, vertex_slices, edge_slices, batch.batch)
                else:
                    vertex_slices = vertex_slices.cpu()
                    edge_slices = edge_slices.cpu()
                    filtered_v_ = filtered_v_.cpu().transpose(1, 0).contiguous()
                    filtered_e_ = filtered_e_.cpu().transpose(1, 0).contiguous()
                    edge_index = edge_index.cpu().transpose(1, 0).contiguous()
                    p_og_0, p_og_1 = compute_persistence_homology_batched_mt(filtered_v_, filtered_e_, edge_index, vertex_slices, edge_slices)
                    p_og = (p_og_0.to(x.device), p_og_1.to(x.device))
                persistenceh_new = {}
                for dim in self.higher_dims:
                    clique_ei = getattr(batch, f"clique_graph_dim_{dim}_edge_index")
                    num_nodes = higher_order_slices[dim][1:] - higher_order_slices[dim][:-1]
                    clique_batch = torch.repeat_interleave(torch.arange(num_nodes.shape[0], dtype=batch.edge_index.dtype), num_nodes)
                    if not TORCHPH:
                        persistenceh_new[dim] = fake_persistence_computation(filtered_higher[dim], clique_ei, higher_order_slices[dim], clique_edge_index_slices[dim], clique_batch)
                    else:
                        clique_filtered_e, _ = torch.max(torch.stack((filtered_higher[dim][clique_ei[0]], filtered_higher[dim][clique_ei[1]])), axis=0)
                        filtered_higher[dim] = filtered_higher[dim].transpose(1, 0).cpu().contiguous()
                        clique_filtered_e = clique_filtered_e.transpose(1, 0).cpu().contiguous()
                        clique_ei = clique_ei.cpu().transpose(1, 0).contiguous()

                        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(
                            filtered_higher[dim], clique_filtered_e, clique_ei, higher_order_slices[dim].cpu(), clique_edge_index_slices[dim].cpu()
                        )
                        zero_pos_e_idxs = (clique_ei[:,0]-clique_ei[:,1] == 0).nonzero()
                        if len(zero_pos_e_idxs) > 0: # it means that a fake edge was inserted (see method __call__ of class LiftToSimplex in data_utils.py)
                            # remove the persistence values for the fake nodes and edges
                            for zero_pos_e in zero_pos_e_idxs:
                                #zero_pos_e = zero_pos_e.item()
                                zero_pos_e = zero_pos_e.squeeze()
                                zero_pos_v = clique_ei[zero_pos_e][0]
                                persistence0_new[:, zero_pos_v, :] = persistence0_new[:, zero_pos_v, :] * 0
                                persistence1_new[:, zero_pos_e, :] = persistence1_new[:, zero_pos_e, :] * 0
                        persistenceh_new[dim] = (persistence0_new.to(x.device), persistence1_new.to(x.device))
                persistence0_new, persistence1_new = p_og
            else:
                persistences0, persistences1, persistencesh = [], [], {}
                if self.higher_dims:
                    persistencesh = {dim: [] for dim in self.higher_dims}

                def _fix_filtrations_for_batch(filtrations, slices):
                    lenghts = slices[1:] - slices[:-1]
                    filtrations_per_graph = torch.split(filtrations, lenghts.tolist(), dim=0)
                    new_filtrations = []
                    max_val = torch.zeros((filtrations.shape[1]), device=filtrations.device)
                    for i, f in enumerate(filtrations_per_graph):
                        if f.shape[0] == 0: # for higher order simplices which may not appear in every graph
                            continue
                        new_filtrations.append(f + max_val)
                        max_val = max_val + f.max(dim=0)[0] + 1e-8 # add epsilon for cases in which filtration is zero
                    new_filtrations = torch.cat(new_filtrations)
                    assert new_filtrations.shape == filtrations.shape
                    return new_filtrations
                copy_filtered_v_ = filtered_v_.detach().clone()
                adjusted_filtered_v_ = _fix_filtrations_for_batch(copy_filtered_v_, vertex_slices)
                copy_filtered_e_ = filtered_e_.detach().clone()
                adjusted_filtered_e_ = _fix_filtrations_for_batch(copy_filtered_e_, edge_slices)
                adjusted_filtered_h_ = {}
                if self.higher_dims:
                    for dim in self.higher_dims:
                        copy_filtered_h_ = filtered_higher[dim].detach().clone()
                        adjusted_filtered_h_[dim] = _fix_filtrations_for_batch(copy_filtered_h_, higher_order_slices[dim])
                for filtration_idx in range(filtered_v_.shape[1]):
                    f_h = {dim: adjusted_filtered_h_[dim][:, filtration_idx] for dim in adjusted_filtered_h_}
                    filtration_p0, filtration_p1, filtration_ph = persistence_computation(edge_index, adjusted_filtered_v_[:, filtration_idx], adjusted_filtered_e_[:, filtration_idx], higher_dim_indexes, f_h)
                    persistences0.append(filtration_p0)
                    persistences1.append(filtration_p1)
                    for dim in filtration_ph:
                        persistencesh[dim].append(filtration_ph[dim])
                persistence0_new = torch.stack(persistences0)
                persistence1_new = torch.stack(persistences1)
                persistenceh_new = {}
                for dim in persistencesh:
                    persistenceh_new[dim] = torch.stack(persistencesh[dim])

        if return_filtration:
            if (self.dim1 or self.higher_dims) and self.separate_filtration_functions:
                filtrations = {}
                filtrations[0] = filtered_v_
                if self.dim1:
                    filtrations[1] = filtered_e_
                if self.higher_dims:
                    for d, f in filtered_higher:
                        filtrations[d] = f
                return (persistence0_new, persistence1_new, persistenceh_new), filtrations
            else:
                return (persistence0_new, persistence1_new, persistenceh_new), filtered_v_
        else:
            return (persistence0_new, persistence1_new, persistenceh_new), None

    def compute_coord_fun(self, persistence, batch, coord_fun_modules):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """
        coord_activation = torch.cat(
            [mod.forward(persistence) for mod in coord_fun_modules], 1)
        return coord_activation

    def compute_coord_activations(self, persistences, batch, coord_fun_modules):
        """
        Return the coordinate functions activations pooled by graph.
        Output dims : list of length number of filtrations with elements : [N_graphs in batch, number fo coordinate functions]
        """
        coord_activations = [self.compute_coord_fun(
            persistence, batch=batch, coord_fun_modules=coord_fun_modules) for persistence in persistences]
        return torch.cat(coord_activations, 1)

    def collapse_dim(self, activations, mask, slices):
        """
        Takes a flattened tensor of activations along with a mask and collapses it (sum) to have a graph-wise features

        Inputs :
        * activations [N_edges,d]
        * mask [N_edge]
        * slices [N_graphs]
        Output:
        * collapsed activations [N_graphs,d]
        """
        collapsed_activations = []
        for el in range(len(slices)-1):
            activations_el_ = activations[slices[el]:slices[el+1]]
            mask_el = mask[slices[el]:slices[el+1]]
            activations_el = activations_el_[mask_el].sum(axis=0)
            collapsed_activations.append(activations_el)

        return torch.stack(collapsed_activations)

    def forward(self, x, batch, return_filtration = False):
        #Remove the duplicate edges.
        batch = remove_duplicate_edges(batch)

        if self.higher_dims and len(self.higher_dims) > 0:
            higher_dim_indexes = {}
            for dim in self.higher_dims:
                if dim == 2:
                    idx_name = "triangle_index"
                else:
                    idx_name = f"simplex_dim_{dim}_index"
                higher_dim_indexes[dim] = getattr(batch, idx_name)
            persistences, filtration = self.compute_persistence(x, batch, return_filtration, higher_dim_indexes)
        else:
            persistences, filtration = self.compute_persistence(x, batch, return_filtration)
        persistences0, persistences1, persistencesh = persistences

        coord_activations = self.compute_coord_activations(
            persistences0, batch, self.coord_fun_modules0)
        if self.dim1:
            persistence1_mask = (persistences1 != 0).any(2).any(0)
            # TODO potential save here by only computing the activation on the masked persistences
            coord_activations1 = self.compute_coord_activations(
                persistences1, batch, self.coord_fun_modules1)
            graph_activations1 = self.collapse_dim(coord_activations1, persistence1_mask, batch._slice_dict[
                "edge_index"])  # returns a vector for each graph
        else:
            graph_activations1 = None

        if self.higher_dims and len(self.higher_dims) > 0:
            graph_activationsh = {}
            for dim in self.higher_dims:
                if self.clique_persistence:
                    graph_acts = []
                    slices = []
                    if dim == 2:
                        idx_name = "triangle_index"
                    else:
                        idx_name = f"simplex_dim_{dim}_index"
                    slices.append(batch._slice_dict[idx_name])
                    slices.append(batch._slice_dict[f"clique_graph_dim_{dim}_edge_index"])
                    for pers, sl in zip(persistencesh[dim], slices):
                        persistenceh_mask = (pers != 0).any(2).any(0)
                        coord_activationsh = self.compute_coord_activations(
                            pers, batch, self.coord_fun_modulesh[str(dim)])
                        graph_acts.append(self.collapse_dim(coord_activationsh, persistenceh_mask, sl))
                    if self.dist_dimh:
                        if self.mlp_combine_dims_clique_persistence:
                            graph_activationsh[dim] = self.mlp_dim_combiner(torch.cat(graph_acts, dim=1))
                        else:
                            graph_activationsh[dim] = graph_acts[0] + graph_acts[1] # returns a vector for each graph
                    else:
                        if self.mlp_combine_dims_clique_persistence:
                            graph_activationsh[dim] = self.mlp_dim_combiner(torch.cat(graph_acts, dim=1))
                        else:
                            graph_activationsh[dim] = torch.cat(graph_acts, dim=1) # returns a vector for each graph
                else:
                    persistenceh_mask = (persistencesh[dim] != 0).any(2).any(0)
                    coord_activationsh = self.compute_coord_activations(
                        persistencesh[dim], batch, self.coord_fun_modulesh[str(dim)])
                    if dim == 2:
                        idx_name = "triangle_index"
                    else:
                        idx_name = f"simplex_dim_{dim}_index"
                    graph_activationsh[dim] = self.collapse_dim(coord_activationsh, persistenceh_mask, batch._slice_dict[idx_name])  # returns a vector for each graph
        else:
            graph_activationsh = None

        if self.residual_and_bn:
            out_activations = self.out(coord_activations)
            if self.dim1 and self.dist_dim1:
                out_activations += self.out1(graph_activations1)[batch.batch]
                graph_activations1 = None
            if self.higher_dims and self.dist_dimh:
                for dim in self.higher_dims:
                    out_activations += self.outh[str(dim)](graph_activationsh[dim])[batch.batch]
                graph_activationsh = None
            if self.swap_bn_order:
                out_activations = self.bn(out_activations)
                out_activations = x + F.relu(out_activations)
            else:
                out_activations = self.bn(out_activations)
                out_activations = x + out_activations
        else:
            #BUG
            print(coord_activations.shape)
            out_activations = self.out(coord_activations)
            if self.dist_dim1 and self.dim1_flag:
                out_activations1 = self.out1(graph_activations1)[batch.batch]
                x0 = torch.cat([out_activations, out_activations1], dim=-1)
                x1 = None
            if self.dist_dimh and self.higher_dims and len(self.higher_dims) > 0:
                for dim in self.higher_dims:
                    out_activationsh = self.outh[str(dim)](graph_activationsh[dim])[batch.batch]
                    x0 = torch.cat([x0, out_activationsh], dim=-1)
                xh = None
            x = self.out(torch.cat([x, x0], dim=-1))
            
            if self.dim1 and self.dist_dim1:
                concat_activations = torch.cat((x, coord_activations), 1)
                print(concat_activations.shape)
                out_activations = self.out(concat_activations)
                out_activations = F.relu(out_activations)

        return out_activations, graph_activations1, graph_activationsh, filtration


class PointWiseMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self,x, **kwargs):
        return self.mlp(x)


class LargerGCNModel(pl.LightningModule):
    def __init__(self, hidden_dim, depth, num_node_features, num_classes, task,
                 lr=0.001, dropout_p=0.2, GIN=False, GAT = False, GatedGCN = False, batch_norm=False,
                 residual=False, train_eps=True, save_filtration = False,
                 add_mlp=False, weight_decay = 0., dropout_input_p=0.,
                 dropout_edges_p=0., strong_reg_check=False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.save_filtration = save_filtration
        self.strong_reg_check = strong_reg_check

        num_heads = 1

        if GIN:
            def build_gnn_layer(is_first = False, is_last = False):
                return GINLayer(in_features = hidden_dim, out_features = hidden_dim, train_eps=train_eps, activation = nn.Identity() if is_last else F.relu, batch_norm = batch_norm, dropout = 0. if is_last else dropout_p, **kwargs)
            graph_pooling_operation = global_add_pool

        elif GAT:
            num_heads = kwargs["num_heads_gnn"]
            def build_gnn_layer(is_first = False, is_last = False):
                return GATLayer( in_features = hidden_dim * num_heads, out_features = (hidden_dim * num_heads) if is_last else hidden_dim, train_eps=train_eps, activation = nn.Identity() if is_last else F.relu, batch_norm = batch_norm, dropout = 0. if is_last else dropout_p, num_heads = 1 if is_last else num_heads, **kwargs)
            graph_pooling_operation = global_mean_pool

        elif GatedGCN:
            def build_gnn_layer(is_first = False, is_last = False):
                return GatedGCNLayer( in_features = hidden_dim , out_features = hidden_dim, train_eps=train_eps, activation = nn.Identity() if is_last else F.relu, batch_norm = batch_norm, dropout = 0. if is_last else dropout_p, **kwargs)
            graph_pooling_operation = global_mean_pool

        else:
            def build_gnn_layer(is_first=False, is_last=False):
                return GCNLayer(
                    hidden_dim,
                    #num_node_features if is_first else hidden_dim,
                    hidden_dim,
                    #num_classes if is_last else hidden_dim,
                    nn.Identity() if is_last else F.relu,
                    0. if is_last else dropout_p,
                    batch_norm, residual)
            graph_pooling_operation = global_mean_pool
        
        if dropout_input_p != 0.:
            self.embedding = torch.nn.Sequential(
                torch.nn.Dropout(dropout_input_p),
                torch.nn.Linear(num_node_features, hidden_dim * num_heads)
            )
        else:
            self.embedding = torch.nn.Linear(num_node_features, hidden_dim * num_heads)

        layers = [
            build_gnn_layer(is_first=i == 0, is_last=i == (depth-1))
            for i in range(depth)
        ]

        if add_mlp:
            #BUGFIX PointWiseMLP(hidden_dim) to PointWiseMLP(hidden_dim * num_heads) - GAT case
            mlp_layer = PointWiseMLP(hidden_dim * num_heads)
            layers.insert(1, mlp_layer)

        self.layers = nn.ModuleList(layers)

        if task is Tasks.GRAPH_CLASSIFICATION:
            self.pooling_fun = graph_pooling_operation
        elif task in [Tasks.NODE_CLASSIFICATION, Tasks.NODE_CLASSIFICATION_WEIGHTED]:
            def fake_pool(x, batch):
                return x
            self.pooling_fun = fake_pool
        else:
            raise RuntimeError('Unsupported task.')

        if (kwargs.get("dim1",False) and ("dim1_out_dim" in kwargs.keys()) and ( not kwargs.get("fake",False))):
            dim_before_class = hidden_dim + kwargs["dim1_out_dim"] #SimpleTopoGNN with dim1
        else:
            dim_before_class = hidden_dim * num_heads

        self.classif = nn.Identity()
        self.classif =  torch.nn.Sequential(
            nn.Linear(dim_before_class, hidden_dim // 2),
             nn.ReLU(),
             nn.Linear(hidden_dim // 2, hidden_dim // 4),
             nn.ReLU(),
             nn.Linear(hidden_dim // 4, num_classes)
         )

        self.task = task
        if task is Tasks.GRAPH_CLASSIFICATION:
            self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
            self.accuracy_val = Accuracy(task='multiclass', num_classes=num_classes)
            self.accuracy_test = Accuracy(task='multiclass', num_classes=num_classes)
            self.loss = torch.nn.CrossEntropyLoss()
        elif task is Tasks.NODE_CLASSIFICATION_WEIGHTED:
            self.accuracy = WeightedAccuracy(num_classes)
            self.accuracy_val = WeightedAccuracy(num_classes)
            self.accuracy_test = WeightedAccuracy(num_classes)

            def weighted_loss(pred, label):
                # calculating label weights for weighted loss computation
                with torch.no_grad():
                    n_classes = pred.shape[1]
                    V = label.size(0)
                    label_count = torch.bincount(label)
                    label_count = label_count[label_count.nonzero(
                        as_tuple=True)].squeeze()
                    cluster_sizes = torch.zeros(
                        n_classes, dtype=torch.long, device=pred.device)
                    cluster_sizes[torch.unique(label)] = label_count
                    weight = (V - cluster_sizes).float() / V
                    weight *= (cluster_sizes > 0).float()
                return F.cross_entropy(pred, label, weight)

            self.loss = weighted_loss
        elif task is Tasks.NODE_CLASSIFICATION:
            self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
            self.accuracy_val = Accuracy(task='multiclass', num_classes=num_classes)
            self.accuracy_test = Accuracy(task='multiclass', num_classes=num_classes)
            # Ignore -100 index as we use it for masking
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.lr = lr

        self.lr_patience = kwargs["lr_patience"]

        self.min_lr = kwargs["min_lr"]

        self.dropout_p = dropout_p
        #self.edge_dropout = EdgeDropout(dropout_edges_p)

        self.weight_decay = weight_decay

    def configure_optimizers(self):
        """Reduce learning rate if val_loss doesnt improve."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        scheduler =  {'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.lr_patience),

            "monitor":"val_loss",
            "frequency":1,
            "interval":"epoch"}

        return [optimizer], [scheduler]

    def forward(self, data):
        # Do edge dropout
        #data = self.edge_dropout(data)

        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, edge_index=edge_index, data=data)

        x = self.pooling_fun(x, data.batch)
        # only for strongly regular dataset: find number fo different graph embeddings in batch
        if self.strong_reg_check:
            print("tot graphs:", x.shape[0])
            print("unique", torch.unique(x, dim=0).shape[0])
            exit()
        x = self.classif(x)

        return x

    def training_step(self, batch, batch_idx):
        y = batch.y
        # Flatten to make graph classification the same as node classification
        y = y.view(-1)
        y_hat = self(batch)
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        loss = self.loss(y_hat, y)
        mask = y != -100

        self.accuracy(torch.nn.functional.softmax(y_hat,-1)[mask], y[mask])

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=y.shape[0])
        self.log("train_acc", self.accuracy, on_step=True, on_epoch=True, batch_size=y.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y
        # Flatten to make graph classification the same as node classification
        y = y.view(-1)
        y_hat = self(batch)
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        loss = self.loss(y_hat, y)
        mask = y != -100
        
        self.accuracy_val(torch.nn.functional.softmax(y_hat,-1)[mask], y[mask])

        self.log("val_loss", loss, on_epoch = True, batch_size=y.shape[0])

        self.log("val_acc", self.accuracy_val, on_epoch=True, batch_size=y.shape[0])

    def test_step(self, batch, batch_idx):
        y = batch.y

        if hasattr(self,"topo1") and self.save_filtration:
            y_hat, filtration = self(batch,return_filtration = True)
        else:
            y_hat = self(batch)
            filtration = None

        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, batch_size=y.shape[0])
        mask = y != -100

        self.accuracy_test(torch.nn.functional.softmax(y_hat,-1)[mask], y[mask])

        self.log("test_acc", self.accuracy_test, on_epoch=True, batch_size=y.shape[0])


        return {"y":y, "y_hat":y_hat, "filtration":filtration}

    # TODO should be fixed - on_test_epoch_end doen't have outputs
    #def on_test_epoch_end(self, outputs):

        #y = torch.cat([output["y"] for output in outputs])
        #y_hat = torch.cat([output["y_hat"] for output in outputs])

        #if hasattr(self,"topo1") and self.save_filtration:
        #    filtration = torch.nn.utils.rnn.pad_sequence([output["filtration"].T for output in outputs], batch_first = True)
            # TODO
            #if self.logger is not None:
                #torch.save(filtration,os.path.join(wandb.run.dir,"filtration.pt"))

        #y_hat_max = torch.argmax(y_hat,1)
            # TODO
        #if self.logger is not None:
            #self.logger.experiment.log_metrics({"conf_mat" : wandb.plot.confusion_matrix(preds=y_hat_max.cpu().numpy(), y_true = y.cpu().numpy())})

    @ classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent])
        parser.add_argument("--hidden_dim", type=int, default=146)
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--lr_patience", type=int, default=10)
        parser.add_argument("--min_lr", type=float, default=0.00001)
        parser.add_argument("--dropout_p", type=float, default=0.0)
        parser.add_argument('--GIN', type=str2bool, default=False)
        parser.add_argument('--GAT', type=str2bool, default=False)
        parser.add_argument('--GatedGCN', type=str2bool, default=False)
        parser.add_argument('--train_eps', type=str2bool, default=True)
        parser.add_argument('--batch_norm', type=str2bool, default=True)
        parser.add_argument('--residual', type=str2bool, default=True)
        parser.add_argument('--save_filtration', type=str2bool, default=False)
        parser.add_argument('--add_mlp', type=str2bool, default=False)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--dropout_input_p', type=float, default=0.)
        parser.add_argument('--dropout_edges_p', type=float, default=0.)
        parser.add_argument('--num_heads_gnn', type=int, default=1)
        parser.add_argument('--strong_reg_check', type=str2bool, default=False)
        return parser


class LargerTopoGNNModel(LargerGCNModel):
    def __init__(self, hidden_dim, depth, num_node_features, num_classes, task,
                 lr=0.001, dropout_p=0.2, GIN=False,
                 batch_norm=False, residual=False, train_eps=True,
                 residual_and_bn=False, aggregation_fn='mean',
                 dim0_out_dim=32, dim1_out_dim=32,
                 share_filtration_parameters=False, fake=False, deepset=False,
                 tanh_filtrations=False, relu_filtrations=False, deepset_type='full', full_deepset_highdims=False,
                 swap_bn_order=False,
                 dist_dim1=False, togl_position=1,
                 separate_filtration_functions=False, higher_dims=[], higher_dims_out_dim=[], dist_dimh=False,
                 clique_persistence=False, mlp_combine_dims_clique_persistence=False,
                 strong_reg_check=False,
                 **kwargs):
        super().__init__(hidden_dim = hidden_dim, depth = depth, num_node_features = num_node_features, num_classes = num_classes, task = task,
                 lr=lr, dropout_p=dropout_p, GIN=GIN,
                 batch_norm=batch_norm, residual=residual, train_eps=train_eps, **kwargs)

        self.save_hyperparameters()
        self.strong_reg_check = strong_reg_check

        self.residual_and_bn = residual_and_bn
        self.num_filtrations = kwargs["num_filtrations"]
        self.filtration_hidden = kwargs["filtration_hidden"]
        self.num_coord_funs = kwargs["num_coord_funs"]
        self.num_coord_funs1 = self.num_coord_funs #kwargs["num_coord_funs1"]

        self.dim1 = kwargs["dim1"]
        self.tanh_filtrations = tanh_filtrations
        self.relu_filtrations = relu_filtrations
        self.deepset_type = deepset_type
        self.full_deepset_highdims = full_deepset_highdims
        self.togl_position = depth if togl_position is None else togl_position

        self.separate_filtration_functions = separate_filtration_functions
        if higher_dims is not None:
            assert len(higher_dims) == len(higher_dims_out_dim)
            self.higher_dims = higher_dims
        else:
            self.higher_dims = []
        if higher_dims:
            self.higher_dims_out_dim = {dim: out_dim for dim, out_dim in zip(higher_dims, higher_dims_out_dim)}
        else:
            self.higher_dims_out_dim = {}
        self.num_coord_funsh = self.num_coord_funs #kwargs["num_coord_funsh"]
        self.dist_dimh = dist_dimh
        self.clique_persistence = clique_persistence
        self.mlp_combine_dims_clique_persistence = mlp_combine_dims_clique_persistence

        self.deepset = deepset

        if kwargs.get("GAT",False):
            hidden_dim = hidden_dim * kwargs["num_heads_gnn"]

        if self.deepset:
            self.topo1 = SimpleSetTopoLayer(
                n_features = hidden_dim,
                n_filtrations =  self.num_filtrations,
                mlp_hidden_dim = self.filtration_hidden,
                aggregation_fn=aggregation_fn,
                dim1=self.dim1,
                dim0_out_dim=dim0_out_dim,
                dim1_out_dim=dim1_out_dim,
                residual_and_bn=residual_and_bn,
                fake = fake,
                deepset_type=deepset_type,
                full_deepset_highdims=self.full_deepset_highdims,
                swap_bn_order=swap_bn_order,
                relu_filtrations=relu_filtrations,
                dist_dim1=dist_dim1,
                separate_filtration_functions=self.separate_filtration_functions,
                higher_dims=self.higher_dims,
                higher_dims_out_dim=self.higher_dims_out_dim,
                dist_dimh=self.dist_dimh,
                clique_persistence=self.clique_persistence,
                mlp_combine_dims_clique_persistence=self.mlp_combine_dims_clique_persistence
            )
        else:
            coord_funs = {"Triangle_transform": self.num_coord_funs,
                          "Gaussian_transform": self.num_coord_funs,
                          "Line_transform": self.num_coord_funs,
                          "RationalHat_transform": self.num_coord_funs
                          }

            coord_funs1 = {"Triangle_transform": self.num_coord_funs1,
                           "Gaussian_transform": self.num_coord_funs1,
                           "Line_transform": self.num_coord_funs1,
                           "RationalHat_transform": self.num_coord_funs1
                           }

            coord_funsh = {}
            if self.higher_dims:
                for dim in self.higher_dims:
                    coord_funsh[dim] = {"Triangle_transform": self.num_coord_funsh,
                                        "Gaussian_transform": self.num_coord_funsh,
                                        "Line_transform": self.num_coord_funsh,
                                        "RationalHat_transform": self.num_coord_funsh
                                        }

            self.topo1 = TopologyLayer(
                hidden_dim, hidden_dim, num_filtrations=self.num_filtrations,
                num_coord_funs=coord_funs, filtration_hidden=self.filtration_hidden,
                dim1=self.dim1, num_coord_funs1=coord_funs1,
                residual_and_bn=residual_and_bn, swap_bn_order=swap_bn_order,
                share_filtration_parameters=share_filtration_parameters, fake=fake,
                tanh_filtrations=tanh_filtrations,
                relu_filtrations=relu_filtrations,
                dist_dim1=dist_dim1,
                separate_filtration_functions=self.separate_filtration_functions,
                higher_dims=self.higher_dims,
                num_coord_funsh=coord_funsh,
                dist_dimh=self.dist_dimh,
                clique_persistence=self.clique_persistence,
                mlp_combine_dims_clique_persistence=self.mlp_combine_dims_clique_persistence
                )

        # number of extra dimension for each embedding from cycles (dim1)
        cycles_dim = 0
        if (self.dim1 and not dist_dim1) or (self.higher_dims and not dist_dimh):
            if self.deepset:
                if self.dim1 and not dist_dim1:
                    cycles_dim += dim1_out_dim
                if self.higher_dims and not dist_dimh:
                    for dim in self.higher_dims:
                        cycles_dim += self.higher_dims_out_dim[dim]
            else: #classical coordinate functions.
                if self.dim1 and not dist_dim1:
                    cycles_dim += self.num_filtrations * np.array(list(coord_funs1.values())).sum()
                if self.higher_dims and not dist_dimh:
                    for dim in self.higher_dims:
                        if self.clique_persistence:
                            cycles_dim += self.num_filtrations * np.array(list(coord_funsh[dim].values())).sum() * 2
                        else:
                            cycles_dim += self.num_filtrations * np.array(list(coord_funsh[dim].values())).sum()

        self.classif = torch.nn.Sequential(
             nn.Linear(hidden_dim + cycles_dim, hidden_dim // 2),
             nn.ReLU(),
             nn.Linear(hidden_dim // 2, hidden_dim // 4),
             nn.ReLU(),
             nn.Linear(hidden_dim // 4, num_classes)
         )


    def configure_optimizers(self):
        """Reduce learning rate if val_loss doesnt improve."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay =  self.weight_decay)
        scheduler =  {'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.lr_patience),
            "monitor":"val_loss",
            "frequency":1,
            "interval":"epoch"}

        return [optimizer], [scheduler]

    def forward(self, data, return_filtration=False):
        # Do edge dropout
        #data = self.edge_dropout(data)
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        
        for layer in self.layers[:self.togl_position]:
            x = layer(x, edge_index=edge_index, data=data)
        x, x_dim1, x_dimh, filtration = self.topo1(x, data, return_filtration)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        for layer in self.layers[self.togl_position:]:
            x = layer(x, edge_index=edge_index, data=data)

        # Pooling
        x = self.pooling_fun(x, data.batch)
        #Aggregating the dim1 topo info if dist_dim1 == False
        if x_dim1 is not None:
            if self.task in [Tasks.NODE_CLASSIFICATION, Tasks.NODE_CLASSIFICATION_WEIGHTED]:
                # Scatter graph level representation to nodes
                x_dim1 = x_dim1[data.batch]
            x_pre_class = torch.cat([x, x_dim1], axis=1)
        else:
            x_pre_class = x

        #Aggregating the higher dim topo info if dist_dimh == False
        if x_dimh is not None:
            if self.task in [Tasks.NODE_CLASSIFICATION, Tasks.NODE_CLASSIFICATION_WEIGHTED]:
                # Scatter graph level representation to nodes
                for dim in self.higher_dims:
                    x_dimh[dim] = x_dimh[dim][data.batch]
            x_pre_class_new = [x_pre_class]
            for dim in self.higher_dims:
                x_pre_class_new.append(x_dimh[dim])
            x_pre_class = torch.cat(x_pre_class_new, axis=1)

        # only for strongly regular dataset: find number fo different graph embeddings in batch
        if self.strong_reg_check:
            print("tot graphs:", x_pre_class.shape[0])
            #print(x_pre_class)
            print("unique", torch.unique(x_pre_class, dim=0).shape[0])
            #print(torch.unique(x_pre_class, dim=0))
            exit()

        #Final classification
        x = self.classif(x_pre_class)

        if return_filtration:
            return x, filtration
        else:
            return x

    @classmethod
    def add_model_specific_args(cls, parent):
        parser = super().add_model_specific_args(parent)
        parser.add_argument('--filtration_hidden', type=int, default=24)
        parser.add_argument('--num_filtrations', type=int, default=8)
        parser.add_argument('--tanh_filtrations', type=str2bool, default=False)
        parser.add_argument('--relu_filtrations', type=str2bool, default=False)
        parser.add_argument('--deepset_type', type=str, choices=['full', 'shallow', 'linear'], default='full')
        parser.add_argument('--full_deepset_highdims', type=str2bool, default=False, help='Use a larger DeepSet model than the default one for processing the persistence of higher-order dimensions.')
        parser.add_argument('--swap_bn_order', type=str2bool, default=False)
        parser.add_argument('--dim1', type=str2bool, default=False)
        parser.add_argument('--higher_dims', nargs='+', type=int, required=False)
        parser.add_argument('--num_coord_funs', type=int, default=3)
        #parser.add_argument('--num_coord_funs1', type=int, default=3)
        #parser.add_argument('--num_coord_funsh', type=int, default=3)
        parser.add_argument('--togl_position', type=int_or_none, default=1, help='Position of the TOGL layer, None means last.')
        parser.add_argument('--residual_and_bn', type=str2bool, default=True, help='Use residual and batch norm')
        parser.add_argument('--share_filtration_parameters', type=str2bool, default=True, help='Share filtration parameters of topo layer')
        parser.add_argument('--separate_filtration_functions', type=str2bool, default=False, help='Obtain filtrations for higher order simplices using separate functions (if False filtrations for higher order simplices are obtained taking the max of the values fo their faces).')
        parser.add_argument('--fake', type=str2bool, default=False, help='Fake topological computations.')
        parser.add_argument('--deepset', type=str2bool, default=False, help='Using DeepSet as coordinate function')
        parser.add_argument('--dim0_out_dim',type=int,default = 32, help = "Inner dim of the set function of the dim0 persistent features")
        parser.add_argument('--dim1_out_dim',type=int,default = 32, help = "Dimension of the ouput of the dim1 persistent features")
        parser.add_argument('--higher_dims_out_dim',type=int, nargs='+', help = "List with dimension of the ouput of the higher order persistent features")
        parser.add_argument('--dist_dim1', type=str2bool, default=False)
        # if dist_dim1==False concatenate the aggregated output of the togl layer for dim0 with the aggregated output for dim1
        # (for node classification every node in the graph gets concatenated with the output for dim1). If dist_dim1==True
        # output of dim1 is "mixed" with output of dim0 (the output of dim1 for every node in a graph is summed with the aggregated output of dim1 for the graph)
        parser.add_argument('--dist_dimh', type=str2bool, default=False)
        parser.add_argument('--clique_persistence', type=str2bool, default=False, help='Create clique graphs and perform ph on all clique graphs for the dimensions in higher_dims')
        parser.add_argument('--mlp_combine_dims_clique_persistence', type=str2bool, default=False, help='Use an mlp to combine the dimension 0 and dimension 1 ph outputs for the clique graphs')
        parser.add_argument('--aggregation_fn', type=str, default='mean')
        #parser.add_argument('--strong_reg_check', type=str2bool, default=False)
        return parser
