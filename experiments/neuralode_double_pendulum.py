from emlp.nn import EquivarianceLayerDP, ScalarTransformerDP 
from trainer.hamiltonian_dynamics import IntegratedODETrainer,DoubleSpringPendulum,TrialNODE
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo 
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import logging
import emlp.nn
import emlp.groups
import objax
import numpy as np
 


levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                    'warn': logging.WARNING,'warning': logging.WARNING,
                    'info': logging.INFO,'debug': logging.DEBUG}

def makeTrainer(*,dataset=DoubleSpringPendulum,num_epochs=2000,ndata=5000,seed=2021,aug=False,
                bs=500,lr=5e-3,max_grad_norm=float('inf'),device='cuda',split={'train':500,'val':.1,'test':.1},
                data_config={'chunk_len':5,'dt':0.2,'integration_time':30,'regen':False},
                transformer_config={
                    'method':'rbf', 'dimensionless':False, 'n_rad':50, 
                    'n_quantiles':1000, 'transform_distribution':'uniform'
                },
                net_config={'n_layers':3,'n_hidden':100,'div':1},log_level='warn',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.75},}, 
                save=False, trial=1):

    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(n_systems=ndata,**data_config)
        datasets = split_dataset(base_ds,splits=split)
    
        zs_train = base_ds.Zs[datasets['train']._ids].reshape(-1,4,3)
        pv_train = np.repeat(base_ds.PV[datasets['train']._ids], data_config['chunk_len'], axis=0) 
        ps_train = np.repeat(base_ds.PS[datasets['train']._ids], data_config['chunk_len'], axis=0) 
       
        stransformer = ScalarTransformerDP(
            zs = zs_train, 
            pv = pv_train,
            ps = ps_train,
            **transformer_config
        )
    model = EquivarianceLayerDP(transformer=stransformer, **net_config)
    
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                   num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
     
    opt_constr = objax.optimizer.Adam
    # lr_sched = lambda e: lr#*cosLr(num_epochs)(e)#*min(1,e/(num_epochs/10))
    lr_sched = lambda e: lr if e < 300 else (lr*0.5 if e < 1200 else lr*0.2)
    return IntegratedODETrainer(
        model,dataloaders,opt_constr,lr_sched,max_grad_norm,**trainer_config
    )

if __name__ == "__main__":
    trial_node = TrialNODE(makeTrainer)
    cfg,outcome = trial_node(argupdated_config(makeTrainer.__kwdefaults__))
    print(outcome)
