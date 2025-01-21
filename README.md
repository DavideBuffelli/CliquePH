# CliquePH: Higher-Order Information for Graph Neural Networks through Persistent Homology on Clique Graphs

This repository contains the code for the LoG 2024 paper "CliquePH: Higher-Order Information for Graph Neural Networks through Persistent Homology on Clique Graphs".
The code is based on `https://github.com/BorgwardtLab/TOGL` and `https://github.com/ExpectationMax/torch_persistent_homology`

Warning: we are in the process of clearning the repository, please forgive us if the current code is a bit messy

## Installation
```bash
conda install python==3.10
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install lightning -c conda-forge or pip install lightning
conda install pytorch-scatter pytorch-sparse pyg -c pyg
conda install -c dglteam/label/cu117 dgl
conda install -c conda-forge gudhi
conda install -c conda-forge graph-tool

cd repos/torch_persistent_homology/torch_persistent_homology
python setup.py install
```

## Training models
The repository implements two models `TopoGNN` and `GCN`.  Additional
parameters can be passed to the script depending on the model and dataset
selected. For example, the `TopoGNN` model and the `MNIST` dataset have the
following configuration options:
```bash
$ python topognn/train_model.py --model TopoGNN --dataset MNIST --help
usage: train_model.py [-h] [--model {TopoGNN,GCN}]
                      [--dataset {IMDB-BINARY,REDDIT-BINARY,REDDIT-5K,PROTEINS,PROTEINS_full,ENZYMES,DD,MUTAG,MNIST,CIFAR10,PATTERN,CLUSTER,Necklaces,Cycles,NoCycles}]
                      [--training_seed TRAINING_SEED]
                      [--max_epochs MAX_EPOCHS] [--paired PAIRED]
                      [--merged MERGED] [--logger {wandb,tensorboard}]
                      [--gpu GPU] [--hidden_dim HIDDEN_DIM] [--depth DEPTH]
                      [--lr LR] [--lr_patience LR_PATIENCE] [--min_lr MIN_LR]
                      [--dropout_p DROPOUT_P] [--GIN GIN]
                      [--train_eps TRAIN_EPS] [--batch_norm BATCH_NORM]
                      [--residual RESIDUAL] [--batch_size BATCH_SIZE]
                      [--use_node_attributes USE_NODE_ATTRIBUTES]

optional arguments:
  -h, --help show this help message and exit
  --model {TopoGNN,GCN}
  --dataset {IMDB-BINARY,REDDIT-BINARY,REDDIT-5K,PROTEINS,PROTEINS_full,ENZYMES,DD,MUTAG,MNIST,CIFAR10,PATTERN,CLUSTER,Necklaces,Cycles,NoCycles}
  --training_seed TRAINING_SEED
  --max_epochs MAX_EPOCHS
  --paired PAIRED
  --merged MERGED
  --logger {wandb,tensorboard}
  --gpu GPU
  --hidden_dim HIDDEN_DIM
  --depth DEPTH
  --lr LR
  --lr_patience LR_PATIENCE
  --min_lr MIN_LR
  --dropout_p DROPOUT_P
  --GIN GIN
  --train_eps TRAIN_EPS
  --batch_norm BATCH_NORM
  --residual RESIDUAL
  --batch_size BATCH_SIZE
  --use_node_attributes USE_NODE_ATTRIBUTES
```

### Citation
If you use this code, please cite our paper.
```
@inproceedings{buffelli2024CliquePH,
    title={CliquePH: Higher-Order Information for Graph Neural Networks through Persistent Homology on Clique Graphs},
    author={Davide Buffelli and Farzin Soleymani and Bastian Rieck},
    year={2024},
    booktitle={Learning on Graphs Conference (LoG)}
}
```
