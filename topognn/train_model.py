#!/usr/bin/env python
"""Train a model using the same routine as used in the GNN Benchmarks dataset."""
import argparse
import sys
import numpy as np
import os
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_info
import data_utils as topo_data
import models as models
from pytorch_lightning import seed_everything
from cli_utils import str2bool

MODEL_MAP = {
    'TopoGNN': models.LargerTopoGNNModel,
    'GNN': models.LargerGCNModel,
}


class StopOnMinLR(Callback):
    """Callback to stop training as soon as the min_lr is reached.

    This is to mimic the training routine from the publication
    `Benchmarking Graph Neural Networks, V. P. Dwivedi, K. Joshi et al.`
    """

    def __init__(self, min_lr):
        super().__init__()
        self.min_lr = min_lr

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        """Check if lr is lower than min_lr.

        This is the closest to after the update of the lr where we can
        intervene via callback. The lr logger also uses this hook to log
        learning rates.
        """

        for scheduler in trainer.lr_scheduler_configs:
            opt = scheduler.scheduler.optimizer
            param_groups = opt.param_groups
            for pg in param_groups:
                lr = pg.get('lr')
                if lr < self.min_lr:
                    trainer.should_stop = True
                    rank_zero_info(
                        'lr={} is lower than min_lr={}. '
                        'Stopping training.'.format(lr, self.min_lr)
                    )


def main(model_cls, dataset_cls, args):
    #args.training_seed = seed_everything(args.training_seed)
    # Instantiate objects according to parameters

    PRINT_STATS = vars(args)["print_stats"]
    if PRINT_STATS:
        args_dict = vars(args)
        if args_dict["dataset"] == "MNIST":
            args_dict["batch_size"] = 1
            dataset = dataset_cls(**args_dict)
        else:
            dataset = dataset_cls(**vars(args))
    else:
        dataset = dataset_cls(**vars(args))
    dataset.prepare_data()

    if PRINT_STATS:
        print("_______")
        print("Dataset Stats")
        global_stat_dict = {i:[] for i in range(dataset.max_simplex_dim+1)}
        global_stat_cliquegraphedges_dict = {i:[] for i in range(2, dataset.max_simplex_dim+1)}
        if args_dict["dataset"] == "MNIST":
            data_split_list = [dataset.train_dataloader().dataset, dataset.val_dataloader().dataset, dataset.test_dataloader().dataset]
        else:
            data_split_list = [dataset.train, dataset.val, dataset.test]
        for split_name, split in zip(["train", "val", "test"], data_split_list):
            print(f"Split: {split_name}")
            stat_dict = {i:[] for i in range(dataset.max_simplex_dim+1)}
            stat_cliquegraphedges_dict = {i:[] for i in range(2, dataset.max_simplex_dim+1)}
            for g in split:
                if args_dict["dataset"] == "MNIST":
                    if hasattr(g, "edge_attr"):
                        del g.edge_attr
                #print(g)
                stat_dict[0].append(g.num_nodes)
                global_stat_dict[0].append(g.num_nodes)
                stat_dict[1].append(g.num_edges/2 if g.is_undirected() else g.num_edges)
                global_stat_dict[1].append(g.num_edges/2 if g.is_undirected() else g.num_edges)
                for dim in range(2, dataset.max_simplex_dim+1):
                    if getattr(g, f"clique_graph_dim_{dim}_edge_index").shape[1] != 1 or (getattr(g, f"clique_graph_dim_{dim}_edge_index")[0][0].sum().item() != 0):
                        stat_cliquegraphedges_dict[dim].append(getattr(g, f"clique_graph_dim_{dim}_edge_index").shape[1])
                        global_stat_cliquegraphedges_dict[dim].append(getattr(g, f"clique_graph_dim_{dim}_edge_index").shape[1])
                    else:
                        stat_cliquegraphedges_dict[dim].append(0)
                        global_stat_cliquegraphedges_dict[dim].append(0)
                    if dim == 2:
                        if g.triangle_index.shape[1] != 1 or g.triangle_index.sum().item() != 0:
                            stat_dict[dim].append(g.triangle_index.shape[1])
                            global_stat_dict[dim].append(g.triangle_index.shape[1])
                        else:
                            stat_dict[dim].append(0)
                            global_stat_dict[dim].append(0)
                    else:
                        if getattr(g, f"simplex_dim_{dim}_index").shape[1] != 1 or (getattr(g, f"simplex_dim_{dim}_index").sum().item() != 0):
                            stat_dict[dim].append(getattr(g, f"simplex_dim_{dim}_index").shape[1])
                            global_stat_dict[dim].append(getattr(g, f"simplex_dim_{dim}_index").shape[1])
                        else:
                            stat_dict[dim].append(0)
                            global_stat_dict[dim].append(0)
            for dim in stat_dict:
                print("Dim {}: {}+-{}".format(dim, np.array(stat_dict[dim]).mean(), np.array(stat_dict[dim]).std()))
                if dim >= 2:
                    print("  Clique Graph Edges: {} +- {}".format(np.array(stat_cliquegraphedges_dict[dim]).mean(), np.array(stat_cliquegraphedges_dict[dim]).std()))
            print("")
        print("________")
        print("Global Stats")
        for dim in global_stat_dict:
            print("Dim {}: $\mathlarger{{{:.1f}}} \pm \mathsmaller{{{:.1f}}}$".format(dim, np.array(global_stat_dict[dim]).mean(), np.array(global_stat_dict[dim]).std()))
            if dim >= 2:
                print("  Clique Graph Edges: $\mathlarger{{{:.1f}}} \pm \mathsmaller{{{:.1f}}}$".format(np.array(global_stat_cliquegraphedges_dict[dim]).mean(), np.array(global_stat_cliquegraphedges_dict[dim]).std()))

    res_repeats = []
    for repeat in range(args.num_repeats):
        model = model_cls(
            **vars(args),
            num_node_features=dataset.node_attributes,
            num_classes=dataset.num_classes,
            task=dataset.task
        )
        print('Running with hyperparameters:')
        print(model.hparams)

        stop_on_min_lr_cb = StopOnMinLR(args.min_lr)
        lr_monitor = LearningRateMonitor('epoch')
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join("results", f"{args.model}_{args.dataset}"),
            monitor='val_loss',
            mode='min',
            verbose=True
        )

        GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
        trainer = pl.Trainer(
            accelerator='auto',
            logger=CSVLogger(os.path.join("results", f"{args.model}_{args.dataset}"), name="log"),
            log_every_n_steps=5,
            max_epochs=args.max_epochs,
            callbacks=[stop_on_min_lr_cb, checkpoint_cb, lr_monitor],
            #profiler="advanced"
        )

        trainer.fit(model, datamodule=dataset)

        print("Performance on validation set:")
        val_results = trainer.test(test_dataloaders=dataset.val_dataloader())[0]
        print(val_results)

        print("Performance on test set:")
        test_results = trainer.test(test_dataloaders=dataset.test_dataloader())[0]
        res_repeats.append(test_results["test_acc"])

        # Just for interest see if loading the state with lowest val loss actually
        # gives better generalization performance.
        """
        checkpoint_path = checkpoint_cb.best_model_path
        trainer2 = pl.Trainer(logger=False)

        model = model_cls.load_from_checkpoint(
            checkpoint_path)
        val_results = trainer2.test(
            model,
            dataloaders=dataset.val_dataloader()
        )[0]

        val_results = {
            name.replace('test', 'val'): value
            for name, value in val_results.items()
        }

        test_results = trainer2.test(
            model,
            dataloaders=dataset.test_dataloader()
        )[0]

        for name, value in {**test_results}.items():
            wandb_logger.experiment.summary['restored_' + name] = value
            """
    print("--- Results over repeats:")
    print("Test Acc:", np.array(res_repeats).mean(), "+-", np.array(res_repeats).std())
    print(res_repeats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys())
    parser.add_argument('--dataset', type=str, choices=topo_data.dataset_map_dict().keys())
    parser.add_argument('--print_stats', type=str2bool, default=False)
    parser.add_argument('--training_seed', type=int, default=42)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument("--paired", type=str2bool, default=False)
    parser.add_argument("--merged", type=str2bool, default=False)

    partial_args, _ = parser.parse_known_args()

    if partial_args.model is None or partial_args.dataset is None:
        parser.print_usage()
        sys.exit(1)
    model_cls = MODEL_MAP[partial_args.model]
    #dataset_cls = DATASET_MAP[partial_args.dataset]
    dataset_cls = topo_data.get_dataset_class(**vars(partial_args))

    parser = model_cls.add_model_specific_args(parser)
    parser = dataset_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, dataset_cls, args)
