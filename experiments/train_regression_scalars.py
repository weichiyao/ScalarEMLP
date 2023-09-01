from oil.utils.utils import cosLr, FixedNumpySeed, FixedPytorchSeed
from torch.utils.data import DataLoader
from oil.datasetup.datasets import split_dataset
from emlp.datasets import Inertia,O5Synthetic,ParticleInteraction
import torch
import pytorch_lightning as pl


from trainer.trainer_scalars_nn import train_pl_model, RandomFixedLengthSampler
from trainer.trainer_scalars_nn import EquivarianceNet, InvarianceNet
from scalars_nn import dataset_transform

def makeTrainerScalars(
    dataset=Inertia,
    ndata=1000+2000,
    seed=2021,
    bs=512,
    epoch_samples=4096,
    device='cuda',
    split={'train':-1,'val':1000,'test':1000},
    trainer_config={
        'log_dir':"/home/",
        'lr':0.001,
        'milestones':[30,200,400,700],
        'gamma':0.1,
        'num_gpus':1,
        'max_epochs':300,
        'min_epochs':0,
        'check_val_every_n_epoch':1,
        'n_hidden_mlp':100, 
        'n_layers_mlp':2,
        'layer_norm_mlp':False
    },
    permutation=False,
    progress_bar=True
):
     
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_dataset = dataset(ndata)
        ## transform the dataset
        base_trans = dataset_transform(
            base_dataset
        ) 
        datasets = split_dataset(base_trans['dataset'], splits=split)
     
    device = torch.device(device)
    # kwargs_gpu = {"num_workers": 1, "pin_memory": True} if device=='cuda' else {}
    print(f"Using {device}, number of gpus is {trainer_config['num_gpus']}.")

    
    
    dataloaders = {
        k: DataLoader(
            v,
            batch_size=min(bs,len(v)),
            shuffle=(k=='train'), 
            num_workers=0,
            pin_memory=False

        ) for k,v in datasets.items()
    }
    if len(dataloaders['train'].dataset) < epoch_samples:
        train_loader = DataLoader(
            dataloaders['train'].dataset,
            sampler=RandomFixedLengthSampler(dataloaders['train'].dataset, epoch_samples),
            batch_size=dataloaders['train'].batch_size,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_loader = dataloaders['train']
    
    # compute the test metrics
    test_metrics = train_pl_model(
        train_loader=train_loader,
        test_loader=dataloaders['test'],
        validation_loader=dataloaders['val'],
        symname=base_dataset.symname,
        n_in_net=base_trans['dim_scalars'],
        n_hidden_mlp=trainer_config['n_hidden_mlp'], 
        n_layers_mlp=trainer_config['n_layers_mlp'],
        layer_norm_mlp=trainer_config['layer_norm_mlp'], 
        num_gpus=trainer_config['num_gpus'],
        max_epochs=trainer_config['max_epochs'],
        min_epochs=trainer_config['min_epochs'],
        learning_rate=trainer_config['lr'],
        milestones=trainer_config['milestones'],
        gamma=trainer_config['gamma'],
        check_val_every_n_epoch=trainer_config['check_val_every_n_epoch'],
        path_logs=trainer_config['log_dir'],
        permutation=permutation,
        progress_bar=progress_bar
    )
    return test_metrics


def restoreResults(
    path_ckpt,
    dataset=Inertia,
    ndata=1000+2000,
    seed=2021,
    bs=512,
    device='cuda',
    num_gpus=1,
    split={'train':-1,'val':1000,'test':1000},
    permutation=True,
    trainer_config=None
):
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_dataset = dataset(ndata)
        ## transform the dataset
        base_trans = dataset_transform(
            base_dataset
        ) 
        datasets = split_dataset(base_trans['dataset'], splits=split)
     
    device = torch.device(device)
    print(f"Using {device}, number of gpus is {num_gpus}.")
    
    dataloaders = {
        k: DataLoader(
            v,
            batch_size=min(bs,len(v)),
            shuffle=(k=='train'), 
            num_workers=0,
            pin_memory=False

        ) for k,v in datasets.items()
    }
    if base_dataset.symname == "O5invariant":
        litmodel = InvarianceNet(
            n_in_net=base_trans['dim_scalars']
        )  
    elif base_dataset.symname == "O3equivariant":
        litmodel = EquivarianceNet(
            n_in_net=base_trans['dim_scalars'],
            permutation=permutation
        ) 
    elif base_dataset.symname == "Lorentz":
        litmodel = InvarianceNet(
            n_in_net=base_trans['dim_scalars']
        ) 
    
    if trainer_config is None:
        model_load = litmodel.load_from_checkpoint(path_ckpt)
    else:
        model_load = litmodel.load_from_checkpoint(
            path_ckpt,
            n_in_net=base_trans['dim_scalars'],
            n_out_net=1, 
            n_hidden_mlp=trainer_config['n_hidden_mlp'], 
            n_layers_mlp=trainer_config['n_layers_mlp'],
            layer_norm_mlp=trainer_config['layer_norm_mlp'], 
            learning_rate=trainer_config['lr'],
            milestones=trainer_config['milestones'],
            gamma=trainer_config['gamma'] 
        )

    # test it on the loaded model
    trainer = pl.Trainer()
    train_metrics=trainer.test(
        model_load, 
        dataloaders=dataloaders['train']
    )[0]
    test_metrics=trainer.test(
        model_load, 
        dataloaders=dataloaders['test']
    )[0]
    return {'train_MSE':train_metrics['MSE'], 
            'train_R2':train_metrics['R2'], 
            'test_MSE':test_metrics['MSE'],
            'test_R2':test_metrics['R2']}


