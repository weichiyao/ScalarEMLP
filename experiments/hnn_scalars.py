from scalaremlp.nn import ScalarMLP, InvarianceLayer_objax, ScalarTransformer
from trainer.hamiltonian_dynamics import IntegratedDynamicsTrainer,DoubleSpringPendulum,hnnScalars_trial
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr,FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config 
import logging
import objax
import numpy as np


levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
          'warn': logging.WARNING,'warning': logging.WARNING,
          'info': logging.INFO,'debug': logging.DEBUG}


def makeTrainerScalars(*,dataset=DoubleSpringPendulum,num_epochs=250,ndata=30600,seed=2022, 
                       bs=512,lr=1e-3,max_grad_norm=0.5,device='cuda',split={'train':-1,'val':100,'test':500},
                       rescale_config={'rand_lower':3, 'rand_upper':7},
                       data_config={'chunk_len':10,'dt':0.2,'integration_time':12},
                       transformer_config={
                           'method':'none', 'dimensionless':True, 'n_rad':50, 
                           'n_quantiles':200, 'transform_distribution':'uniform'
                       },
                       net_config={'n_layers':5,'n_hidden':100,'div':1}, log_level='info',
                       trainer_config={'log_dir':'/home','log_args':{'minPeriod':.02,'timeFrac':.75},},
                       save=False,trial=1):
    logging.getLogger().setLevel(levels[log_level])

    # Prep the datasets splits, model, and dataloaders
    # Generate the train and validation sets 
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(
            n_systems=ndata,
            **data_config 
        )
        datasets = split_dataset(base_ds,splits=split)
          
    # Generate the original test sets 
    with FixedNumpySeed(seed),FixedPytorchSeed(seed): 
        test_ds0 = dataset(
            n_systems=split['test'], 
            **data_config  
        )
        datasets['test'] = test_ds0
         
        scale = np.random.uniform(
            rescale_config['rand_lower'], 
            rescale_config['rand_upper'],
            size=(split['test'],)
        )
          
    # Generate the additional test sets 
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        # scale mass-related inputs in the test set by rescaleKG1
        test_ds1 = dataset(
            n_systems=split['test'], 
            **data_config,  
            rescaleKG=True,
            scale=scale
        )
        # scale mass-related inputs in the test set randomly by rescaleKG2 
        test_ds2 = dataset(
            n_systems=split['test'], 
            **data_config,
            changedist=True
        )
        datasets['test1'] = test_ds1
        datasets['test2'] = test_ds2      
          
    # Create the transformer and the model 
    zs_train = base_ds.Zs[datasets['train']._ids].reshape(-1,4,3)
    zps_train = np.repeat(
        base_ds.ZPs[datasets['train']._ids], 
        data_config['chunk_len'], 
        axis=0
    ) 
    stransformer = ScalarTransformer(
        zs = zs_train, 
        zps = zps_train,   
        **transformer_config
    ) 
    model = InvarianceLayer_objax(transformer=stransformer, **net_config)
          
    # Create data loaders
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                   num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
          
    # Trainer 
    opt_constr = objax.optimizer.Adam 
    lr_sched = lambda e: lr  
    return IntegratedDynamicsTrainer(
        model,dataloaders,opt_constr,lr_sched,max_grad_norm,**trainer_config
    )

if __name__ == "__main__":
    Trial = hnnScalars_trial(makeTrainerScalars)
    cfg, outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))
    print(outcome)

