"""Implementation of layers following Benchmarking GNNs paper."""
import gudhi as gd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_scatter import scatter
from data_utils import remove_duplicate_edges
from torch_geometric.utils import sort_edge_index
from simplicial_utils import sort_higher_order_index

try: # it's just to be able to test the code on my laptop where I can't install torchph
    from torch_persistent_homology import compute_persistence_homology_batched_mt
    TORCHPH = True
except:
    TORCHPH = False


class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, dropout, batch_norm, residual=True
    ):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
        self.conv = GCNConv(in_features, out_features, add_self_loops=False)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GINLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        dropout,
        batch_norm,
        mlp_hidden_dim=None,
        residual=True,
        train_eps=False,
        **kwargs
    ):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = in_features

        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
        gin_net = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        self.conv = GINConv(gin_net, train_eps=train_eps)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GATLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        dropout,
        batch_norm,
        num_heads,
        residual=True,
        train_eps=False,
        **kwargs
    ):
        super().__init__()


        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features * num_heads) if batch_norm else nn.Identity()

        self.conv = GATConv(in_features, out_features, heads = num_heads, dropout = dropout)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)

class GatedGCNLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        dropout,
        batch_norm,
        residual=True,
        train_eps=False,
        **kwargs
    ):
        super().__init__()


        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()

        self.conv = ResGatedGraphConv(in_features, out_features)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        # Apply aggregation function over graph
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class InvariantDeepSet(nn.Module):
    """Simple invariant deep set."""

    def __init__(self, in_dim, out_dim, aggregation_fn, num_layers):
        super().__init__()
        self.gamma_layers = torch.nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(num_layers-1)]
                                             + [nn.Linear(in_dim, out_dim)])
        assert aggregation_fn in ["mean", "max", "sum"]
        self.Lambda = nn.Linear(out_dim, out_dim, bias=False)
        self.aggregation_fn = aggregation_fn

    def forward(self, x):
        """ x.shape = [num_simplices, dim_simplices, node_emb_dim] """
        for gamma_layer in self.gamma_layers:
            x = gamma_layer(x)
            x = F.relu(x)
        if self.aggregation_fn == "mean":
            xm = x.mean(dim=1)
        elif self.aggregation_fn == "max":
            xm = x.max(dim=1)
        else:
            xm = x.sum(dim=1)
        x_final = self.Lambda(xm)
        return x_final


class DeepSetLayerDimh(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn

    def forward(self, x, idx_slices, mask=None):
        '''
        Mask is True where the persistence (x) is observed.
        '''
        # Apply aggregation function over graph

        # Computing the equivalent of batch over higher order(>0) simplices.
        idx_diff_slices = (idx_slices[1:]-idx_slices[:-1]).to(x.device)
        n_batch = len(idx_diff_slices)
        batch_e = torch.repeat_interleave(torch.arange(
            n_batch, device=x.device), idx_diff_slices)
        # Only aggregate over simplices with non zero persistence pairs.
        if mask is not None:
            batch_e = batch_e[mask]

        xm = scatter(x, batch_e, dim=0,
                     reduce=self.aggregation_fn, dim_size=n_batch)

        xm = self.Lambda(xm)

        # xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        # xm = self.Lambda(xm)
        # x = self.Gamma(x)
        # x = x - xm[batch, :]
        return xm


def persistence_computation(edge_index, filtered_v, filtered_e, higher_dim_indexes=None, filtered_higher=None):
    """
    higher_dim_indexes = dictionary dim: index (tensor of shape (dim+1, num_simplices_of_dimension_dim))
    filtered_higher = dictionary dim: tensor with filtrations (position i has filtration for i-th simplex of dimension dim)
    """
    st = gd.SimplexTree()
    idx_filtration_for_simplex = {}
    for node, filtration in enumerate(filtered_v):
        st.insert([node], filtration)
        idx_filtration_for_simplex[tuple([node])] = node
    for i, filtration in enumerate(filtered_e):
        edge = [edge_index[0][i].item(), edge_index[1][i].item()]
        st.insert(edge, filtration)
        #edge.sort()
        idx_filtration_for_simplex[tuple(edge)] = i
        idx_filtration_for_simplex[tuple(edge[::-1])] = i
    if higher_dim_indexes is not None:
        for dim, filtration in filtered_higher.items():
            for i in range(filtration.shape[0]):
                simplex = [higher_dim_indexes[dim][j][i].item() for j in range(higher_dim_indexes[dim].shape[0])]
                st.insert(simplex, filtration[i])
                simplex.sort()
                idx_filtration_for_simplex[tuple(simplex)] = i

    barcode = st.persistence(min_persistence=-1, persistence_dim_max=True)
    persistence_pairs = st.persistence_pairs()

    persistence0 = torch.zeros(filtered_v.shape[0], 2).to(filtered_v.device)
    persistence1 = torch.zeros(edge_index.shape[1], 2).to(edge_index.device)
    persistenceh = {}
    if higher_dim_indexes:
        for dim, idx in higher_dim_indexes.items():
            persistenceh[dim] = torch.zeros(idx.shape[1], 2).to(idx.device)
    #sorted_edge_index = sort_edge_index(edge_index, num_nodes=filtered_v.shape[0])
    #if higher_dim_indexes:
        #sorted_h_inedxes = {dim: sort_higher_order_index(higher_dim_indexes[dim], filtered_v.shape[0]) for dim in higher_dim_indexes}
    infinity_for_dim = {0: filtered_v.max()}
    infinity_for_dim[1] = filtered_e.max()
    if higher_dim_indexes:
        for h in higher_dim_indexes.keys():
            infinity_for_dim[h] = filtered_higher[h].max()
    for pair in persistence_pairs:
        s1, s2 = pair

        def _get_filtration_for_simplex(s):
            dim = len(s) - 1
            if dim == 0:
                idx = s[0]
                f = filtered_v[idx]
            elif dim == 1:
                #s.sort()
                idx = idx_filtration_for_simplex[tuple(s)]
                f = filtered_e[idx]
            else:
                #sorted_h_idx = sorted_h_inedxes[dim]
                s.sort()
                idx = idx_filtration_for_simplex[tuple(s)]
                f = filtered_higher[dim][idx]
            return idx, f

        p1, f1 = _get_filtration_for_simplex(s1)
        if len(s1) == 1:
            persistence0[p1][0] = f1
        elif len(s1) == 2:
            persistence1[p1][0] = f1
        else:
            persistenceh[len(s1)-1][p1][0] = f1
        if len(s2) > 0:
            p2, f2 = _get_filtration_for_simplex(s2)
            if len(s1) == 1:
                persistence0[p1][1] = f2
            elif len(s1) == 2:
                persistence1[p1][1] = f2
            else:
                persistenceh[len(s1)-1][p1][1] = f2
        # TODO: What to do for the simplices for which Gudhi does not five a barcode element? (now: left to zero and no backprop)
        else: # TODO: What happens if the second is infinity? (now: set to max value for filtration)
            if len(s1) == 1:
                persistence0[p1][1] = infinity_for_dim[len(s1)-1]
            elif len(s1) == 2:
                persistence1[p1][1] = infinity_for_dim[len(s1)-1]
            else:
                persistenceh[len(s1)-1][p1][1] = infinity_for_dim[len(s1)-1]

    return persistence0, persistence1, persistenceh


def fake_persistence_computation(filtered_v_, edge_index, vertex_slices, edge_slices, batch):
    #print("Filtered_v shape", filtered_v_.shape)
    device = filtered_v_.device
    num_filtrations = filtered_v_.shape[1]
    filtered_e_, _ = torch.max(torch.stack(
        (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)
    #print("filtered_e_ shape", filtered_e_.shape) # (num_edges, num_filtrations)

    # Make fake tuples for dim 0
    persistence0_new = filtered_v_.unsqueeze(-1).expand(-1, -1, 2)
    #print("persistence0_new shape", persistence0_new.shape) # (num_nodes, num_filtrations, 2) each filtration is duplicated in the last dimension (like if every nodes borns and dies at same time)

    edge_slices = edge_slices.to(device)
    bs = edge_slices.shape[0] - 1
    #print("bs", bs) # batch size
    # Make fake dim1 with unpaired values
    # unpaired_values = scatter(filtered_v_, batch, dim=0, reduce='max')
    unpaired_values = torch.zeros((bs, num_filtrations), device=device)
    persistence1_new = torch.zeros(
        edge_index.shape[1], filtered_v_.shape[1], 2, device=device)
    #print("persistence1_new shape", persistence1_new.shape) # (num_edges, num_filtrations, 2)

    n_edges = edge_slices[1:] - edge_slices[:-1]
    #print("n_edges", n_edges)
    #print(edge_slices[0:-1].unsqueeze(-1))
    #print(edge_slices[0:-1].unsqueeze(-1).shape) # (batch_size, 1)
    #print((torch.rand(size=(bs, num_filtrations), device=device) * n_edges.float().unsqueeze(-1)))
    #print((torch.rand(size=(bs, num_filtrations), device=device) * n_edges.float().unsqueeze(-1)).shape) # (batch_size, num_filtrations)
    if n_edges[-1] == 0:
        # problem when clique graph is empty
        edge_slices[-2] = edge_slices[-2]-1
    random_edges = (
        edge_slices[0:-1].unsqueeze(-1) +
        torch.floor(
            torch.rand(size=(bs, num_filtrations), device=device)
            * n_edges.float().unsqueeze(-1)
        )
    ).long()
    #print("random edges", random_edges) # (batch_size, num_filtrations)
    #print("random edges", random_edges.shape) # (batch_size, num_filtrations)
    #exit()

    #print(torch.stack([
    #        unpaired_values,
    #        filtered_e_[
    #                random_edges, torch.arange(num_filtrations).unsqueeze(0)]
    #    ], -1).shape) # (batch_size, num_filtrations, 2)
    #print(persistence1_new[random_edges, torch.arange(num_filtrations).unsqueeze(0), :].shape) # (batch_size, num_filtrations, 2)

    #print("aaa")
    #print(edge_slices[0:-1].unsqueeze(-1))
    #print(torch.floor(
    #        torch.rand(size=(bs, num_filtrations), device=device)
    #        * n_edges.float().unsqueeze(-1)
    #    ))
    #print("ciao")
    #print(random_edges.shape, random_edges)
    #print(bs, n_edges)
    #print(filtered_v_.shape)
    #print(filtered_e_.shape)
    #print(edge_index.shape, edge_index)
    #print(batch.shape)
    #print(unpaired_values.shape, unpaired_values)

    persistence1_new[random_edges, torch.arange(num_filtrations).unsqueeze(0), :] = (
        torch.stack([
            unpaired_values,
            filtered_e_[
                    random_edges, torch.arange(num_filtrations).unsqueeze(0)]
        ], -1)
    )

    #if higher_dim_indexes is not None:
        #higher_dims_outputs = {}
        #for dim, idx in higher_dim_indexes.items():
            #filtered_dim_i_, _ = torch.max(torch.stack(tuple([filtered_v_[idx[i]] for i in range(idx.shape[0])])), axis=0)

    #print("final persistence0_new", persistence0_new.permute(1,0,2).shape) # (num_filtrations, num_nodes, 2)
    #exit()
    return persistence0_new.permute(1, 0, 2), persistence1_new.permute(1, 0, 2)


class SimpleSetTopoLayer(nn.Module):
    def __init__(self, n_features, n_filtrations, mlp_hidden_dim,
                 aggregation_fn, dim0_out_dim, dim1_out_dim, dim1,
                 residual_and_bn, fake, deepset_type='full', full_deepset_highdims=False,
                 swap_bn_order=False, relu_filtrations=False, dist_dim1=False,
                 separate_filtration_functions=False, higher_dims=[], higher_dims_out_dim={}, dist_dimh=False,
                 clique_persistence=False, mlp_combine_dims_clique_persistence=False):
        super().__init__()
        assert deepset_type in ['linear', 'shallow', 'full']

        self.num_filtrations = n_filtrations
        self.residual_and_bn = residual_and_bn
        self.swap_bn_order = swap_bn_order
        self.relu_filtrations = relu_filtrations
        self.dist_dim1 = dist_dim1

        self.separate_filtration_functions = separate_filtration_functions
        self.higher_dims = higher_dims
        self.higher_dims_out_dim = higher_dims_out_dim
        self.dist_dimh = dist_dimh
        self.clique_persistence = clique_persistence
        self.mlp_combine_dims_clique_persistence = mlp_combine_dims_clique_persistence
        if mlp_combine_dims_clique_persistence:
            # assumes same dimensionalities for all higher dims
            inout_dim = n_features if residual_and_bn and dist_dimh else self.higher_dims_out_dim[self.higher_dims[0]]
            self.mlp_dim_combiner = nn.Sequential(nn.Linear(inout_dim*2, inout_dim), nn.ReLU(), nn.Linear(inout_dim, inout_dim))

        if self.relu_filtrations:
            self.filtration_modules0 = nn.Sequential(
                nn.Linear(n_features, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, n_filtrations),
                nn.ReLU()
            )
        else:
            self.filtration_modules0 = nn.Sequential(
                nn.Linear(n_features, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, n_filtrations),
            )
        if separate_filtration_functions:
            assert dim1
            self.filtration_modulesh = {}
            orders = [1]
            if self.higher_dims:
                orders += self.higher_dims
            for i in orders:
                final_activation = torch.nn.ReLU() if self.relu_filtrations else torch.nn.Identity()
                self.filtration_modulesh[str(i)] = torch.nn.Sequential( ### ModuleDict requires string keys
                    InvariantDeepSet(n_features, n_filtrations, "mean", 2),
                    final_activation
                )
            self.filtration_modulesh = torch.nn.ModuleDict(self.filtration_modulesh)

        self.dim1_flag = dim1
        if self.dim1_flag:
            if full_deepset_highdims:
                self.set_fn1 = nn.ModuleList([
                    nn.Linear(n_filtrations * 2, dim1_out_dim),
                    nn.ReLU(),
                    DeepSetLayer(
                        in_dim=dim1_out_dim, out_dim=dim1_out_dim, aggregation_fn=aggregation_fn),
                    nn.ReLU(),
                    DeepSetLayerDimh(
                        in_dim=dim1_out_dim, out_dim=n_features if residual_and_bn and dist_dim1 else dim1_out_dim, aggregation_fn=aggregation_fn),
                ])
            else:
                self.set_fn1 = nn.ModuleList([
                    nn.Linear(n_filtrations * 2, dim1_out_dim),
                    nn.ReLU(),
                    DeepSetLayerDimh(
                        in_dim=dim1_out_dim, out_dim=n_features if residual_and_bn and dist_dim1 else dim1_out_dim, aggregation_fn=aggregation_fn),
                ])

        if self.higher_dims and len(self.higher_dims) > 0:
            self.set_fnh = {}
            for dim in self.higher_dims:
                if full_deepset_highdims:
                    self.set_fnh[str(dim)] = nn.ModuleList([
                        nn.Linear(n_filtrations * 2, self.higher_dims_out_dim[dim]),
                        nn.ReLU(),
                        DeepSetLayer(
                            in_dim=self.higher_dims_out_dim[dim],
                            out_dim=self.higher_dims_out_dim[dim],
                            aggregation_fn=aggregation_fn),
                        nn.ReLU(),
                        DeepSetLayerDimh(
                            in_dim=self.higher_dims_out_dim[dim],
                            out_dim=n_features if residual_and_bn and dist_dimh else self.higher_dims_out_dim[dim],
                            aggregation_fn=aggregation_fn),])
                else:
                    self.set_fnh[str(dim)] = nn.ModuleList([
                        nn.Linear(n_filtrations * 2, self.higher_dims_out_dim[dim]),
                        nn.ReLU(),
                        DeepSetLayerDimh(
                            in_dim=self.higher_dims_out_dim[dim],
                            out_dim=n_features if residual_and_bn and dist_dimh else self.higher_dims_out_dim[dim],
                            aggregation_fn=aggregation_fn),])
            self.set_fnh = torch.nn.ModuleDict(self.set_fnh)

        if deepset_type == 'linear': ### TODO why not same options for higher order?
            self.set_fn0 = nn.ModuleList([nn.Linear(
                n_filtrations * 2,
                n_features if residual_and_bn else dim0_out_dim, aggregation_fn)
            ])
        elif deepset_type == 'shallow':
            self.set_fn0 = nn.ModuleList(
                [
                    nn.Linear(n_filtrations * 2, dim0_out_dim),
                    nn.ReLU(),
                    DeepSetLayer(
                        dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn),
                ]
            )
        else:
            self.set_fn0 = nn.ModuleList(
                [
                    nn.Linear(n_filtrations * 2, dim0_out_dim),
                    nn.ReLU(),
                    DeepSetLayer(dim0_out_dim, dim0_out_dim, aggregation_fn),
                    nn.ReLU(),
                    DeepSetLayer(
                        dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn),
                ]
            )

        if residual_and_bn:
            self.bn = nn.BatchNorm1d(n_features)
        else:
            in_dim = dim0_out_dim + n_features
            if dist_dim1:
                in_dim += dim1_out_dim
            if dist_dimh:
                for dim in self.higher_dims:
                    in_dim += self.higher_dims_out_dim[dim]
            self.out = nn.Sequential(
                nn.Linear(in_dim, n_features),
                nn.ReLU()
            )
        self.fake = fake

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

        self.share_filtration_parameters = True # always True when using DeepSet
        filtered_v_, filtered_e_, filtered_higher = self.compute_filtration(x, edge_index, higher_dim_indexes)

        if self.fake:
            return fake_persistence_computation(
                filtered_v_, edge_index, vertex_slices, edge_slices, batch)
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

    def forward(self, x, data, return_filtration):
        # Remove the duplucate edges
        data = remove_duplicate_edges(data)
        edge_slices = torch.Tensor(data._slice_dict['edge_index']).cpu().long()
        if self.higher_dims and len(self.higher_dims) > 0:
            h_slices = {}
            for dim in self.higher_dims:
                if dim == 2:
                    idx_name = "triangle_index"
                else:
                    idx_name = f"simplex_dim_{dim}_index"
                h_slices[dim] =  torch.LongTensor(data._slice_dict[idx_name])

        if self.higher_dims and len(self.higher_dims) > 0:
            higher_dim_indexes = {}
            for dim in self.higher_dims:
                if dim == 2:
                    idx_name = "triangle_index"
                else:
                    idx_name = f"simplex_dim_{dim}_index"
                higher_dim_indexes[dim] = getattr(data, idx_name)
            persistences, filtration = self.compute_persistence(x, data, return_filtration, higher_dim_indexes)
        else:
            persistences, filtration = self.compute_persistence(x, data, return_filtration)
        persistences0, persistences1, persistencesh = persistences

        x0 = persistences0.permute(1, 0, 2).reshape(persistences0.shape[1], -1) # x0 has shape (num_nodes, num_filtrations*2) i.e. every row has all the filtrations (born, died), one after the other, for the node)

        for layer in self.set_fn0:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, data.batch)
            else:
                x0 = layer(x0)

        if self.dim1_flag:
            # Dim 1 computations.
            persistences1_reshaped = persistences1.permute(1, 0, 2).reshape(persistences1.shape[1], -1)
            persistences1_mask = ~((persistences1_reshaped == 0).all(-1))
            x1 = persistences1_reshaped[persistences1_mask]
            for layer in self.set_fn1:
                if isinstance(layer, DeepSetLayer): # full_deepset_highdims case
                    idx_diff_slices = (edge_slices[1:]-edge_slices[:-1]).to(x.device)
                    n_els = len(idx_diff_slices)
                    b = torch.repeat_interleave(torch.arange(n_els, device=x.device), idx_diff_slices)
                    b = b[persistences1_mask]
                    x1 = layer(x1, b)
                elif isinstance(layer, DeepSetLayerDimh):
                    x1 = layer(x1, edge_slices, mask=persistences1_mask)
                else:
                    x1 = layer(x1)
        else:
            x1 = None

        if self.higher_dims and len(self.higher_dims) > 0:
            # Higher dims computations.
            xh = {}
            for dim in self.higher_dims:
                if self.clique_persistence:
                    graph_acts = []
                    for pers, sl in zip(persistencesh[dim], [h_slices[dim], data._slice_dict[f"clique_graph_dim_{dim}_edge_index"]]):
                        persistencesh_reshaped = pers.permute(1, 0, 2).reshape(pers.shape[1], -1)
                        persistencesh_mask = ~((persistencesh_reshaped == 0).all(-1))
                        act = persistencesh_reshaped[persistencesh_mask]
                        for layer in self.set_fnh[str(dim)]:
                            if isinstance(layer, DeepSetLayer): # full_deepset_highdims case
                                idx_diff_slices = (sl[1:]-sl[:-1]).to(x.device)
                                n_els = len(idx_diff_slices)
                                b = torch.repeat_interleave(torch.arange(n_els, device=x.device), idx_diff_slices)
                                b = b[persistencesh_mask]
                                act = layer(act, b)
                            elif isinstance(layer, DeepSetLayerDimh):
                                act = layer(act, sl, mask=persistencesh_mask)
                            else:
                                act = layer(act)
                        graph_acts.append(act)
                    if self.mlp_combine_dims_clique_persistence:
                        xh[dim] = self.mlp_dim_combiner(torch.cat(graph_acts, dim=1))
                    else:
                        xh[dim] = graph_acts[0] + graph_acts[1]
                else:
                    persistencesh_reshaped = persistencesh[dim].permute(1, 0, 2).reshape(persistencesh[dim].shape[1], -1)
                    persistencesh_mask = ~((persistencesh_reshaped == 0).all(-1))
                    xh[dim] = persistencesh_reshaped[persistencesh_mask]
                    for layer in self.set_fnh[str(dim)]:
                        if isinstance(layer, DeepSetLayerDimh):
                            xh[dim] = layer(xh[dim], h_slices[dim], mask=persistencesh_mask)
                        else:
                            xh[dim] = layer(xh[dim])
        else:
            xh = None

        if self.residual_and_bn:
            if self.dist_dim1 and self.dim1_flag:
                x0 = x0 + x1[data.batch]
                x1 = None
            if self.dist_dimh and self.higher_dims and len(self.higher_dims) > 0:
                for dim in self.higher_dims:
                    x0 = x0 + xh[dim][data.batch]
                xh = None
            if self.swap_bn_order:
                x = x + F.relu(self.bn(x0))
            else:
                x = x + self.bn(F.relu(x0))
        else:
            if self.dist_dim1 and self.dim1_flag:
                x0 = torch.cat([x0, x1[data.batch]], dim=-1)
                x1 = None
            if self.dist_dimh and self.higher_dims and len(self.higher_dims) > 0:
                for dim in self.higher_dims:
                    #DEBUG
                    x0 = torch.cat([x0, xh[dim][data.batch]], dim=-1)
                xh = None
            x = self.out(torch.cat([x, x0], dim=-1))

        return x, x1, xh, filtration
