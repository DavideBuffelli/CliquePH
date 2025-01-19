import os
import random
import yaml
import argparse
import wandb
from train_model import MODEL_MAP, StopOnMinLR
import data_utils as topo_data
import models as models
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything

torch.set_float32_matmul_precision(precision='high')

def read_yaml(yaml_file_path):
    with open(f'{yaml_file_path}','r') as f:
        output = yaml.safe_load(f)
    return output

def dump_yaml(data, yaml_file_path):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def to_text(data, file_path):
    with open(file_path, 'w') as file:
        # Write the data to the file
        file.write(data)

def model_pipline(num_repeats, config):
    #wandb.init(config=config)
    #config = wandb.config
    model_cls = MODEL_MAP[config["model"]]
    
    res_repeats = []
    for repeat in range(num_repeats):
        config["training_seed"] = np.random.randint(100000000)
        seed_everything(config["training_seed"])
        
        dataset_cls = topo_data.get_dataset_class(**config)
        
        print(f'Num Repeat: {repeat}/{num_repeats}')
        
        dataset = dataset_cls(**config)
        dataset.prepare_data()
        
        model = model_cls(
            **config,
            num_node_features=dataset.node_attributes,
            num_classes=dataset.num_classes,
            task=dataset.task
        )

        print(model)
        print('Running with hyperparameters:')
        print(model.hparams)
    
        # Loggers and callbacks
        #wandb_logger = WandbLogger(
        #name=f"{config['model']}_{config['dataset']}",
        #project="topo_gnn",
        #log_model=True,
        #tags=[config['model'], config['dataset']],
        #save_dir=os.path.join(wandb_logger.experiment.dir, f"{config['model']}_{config['dataset']}")
        #)
    
        stop_on_min_lr_cb = StopOnMinLR(config["min_lr"])
        lr_monitor = LearningRateMonitor('epoch')
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join("results", f"{config['model']}_{config['dataset']}"),
            #dirpath=wandb_logger.experiment.dir,
            monitor='val_loss',
            mode='min',
            verbose=True
        )
        trainer = pl.Trainer(
            accelerator='auto',
            devices = -1,
            strategy='ddp_find_unused_parameters_true',
            logger=CSVLogger(os.path.join("results", f"{config['model']}_{config['dataset']}"), name="log"),
            #logger=wandb_logger,
            log_every_n_steps=5,
            max_epochs=config['max_epochs'],
            callbacks=[stop_on_min_lr_cb, checkpoint_cb, lr_monitor],
            #profiler="advanced"
        )
        
        trainer.fit(model, datamodule=dataset)

        print("Performance on validation set:")
        val_results = trainer.validate(dataloaders=dataset.val_dataloader())[0]
        print(val_results)
        
        print("Performance on test set:")
        test_results = trainer.test(dataloaders=dataset.test_dataloader())[0]
        
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

    acc_dict = {"mean_var_acc": str(np.array(res_repeats).mean())+"+-"+str(np.array(res_repeats).std())}
    for count, acc in enumerate(res_repeats):
        acc_dict[f"result_{count}"] = acc
        
    file_path = f"./best_mean_acc/out_{config['dataset']}_GAT_{config['GAT']}_GCN_{config['GCN']}_GIN_{config['GIN']}_use_node_attributes_{config['use_node_attributes']}.txt"
    to_text(str(acc_dict), file_path)

def test(num_repeats=10, use_node_attributes=True, config_path=" "):
    fixed_config = read_yaml(config_path)
    fixed_config["use_node_attributes"] = use_node_attributes
    model_pipline(num_repeats, fixed_config)

dataset = ["IMDB-BINARY", "IMDB-MULTI", "ENZYMES", "MNIST", "PROTEINS", "PROTEINS_full", "DD", "REDDIT-5K", "PATTERN", "CLUSTER", "Necklaces", "Cycles"]
GAT = False
GCN = False
GIN = False

gnn_model = "GCN"

if gnn_model == "GAT":
    GAT = True    
if gnn_model == "GCN":
    GCN = True
if gnn_model == "GIN":
    GIN = True
    
use_node_attributes = False

config_path = f"best_config/hparams_{dataset[4]}_GAT_{GAT}_GCN_{GCN}_GIN_{GIN}_use_node_attributes_{use_node_attributes}.yaml"
test(10, use_node_attributes, config_path)