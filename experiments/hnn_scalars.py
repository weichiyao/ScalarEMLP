from emlp.nn import ScalarMLP, InvarianceLayer_objax, ScalarTransformer
from trainer.hamiltonian_dynamics import IntegratedDynamicsTrainer,DoubleSpringPendulum,hnnScalars_trial
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr,FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config 
import logging
import emlp
import emlp.reps
import objax
import numpy as np


levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
          'warn': logging.WARNING,'warning': logging.WARNING,
          'info': logging.INFO,'debug': logging.DEBUG}


def makeTrainerScalars(*,dataset=DoubleSpringPendulum,num_epochs=2000,ndata=5000,seed=2021, 
                bs=500,lr=5e-3,max_grad_norm=0.5,device='cuda',split={'train':-1,'val':100,'test':100},
                data_config={'chunk_len':10,'dt':0.2,'integration_time':6,'regen':False},
                transformer_config={
                    'method':'none', 'dimensionless':True, 'n_rad':50, 
                    'n_quantiles':200, 'transform_distribution':'uniform'
                },
                net_config={'n_layers':4,'n_hidden':200,'div':2}, log_level='info',
                trainer_config={'log_dir':'/home/','log_args':{'minPeriod':.02,'timeFrac':.75},},
                save=False,trial=1):
    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(n_systems=ndata, **data_config)
        datasets = split_dataset(base_ds,splits=split)
    
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
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                   num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = objax.optimizer.Adam
    # lr_sched = lambda e: lr#*cosLr(num_epochs)(e)#*min(1,e/(num_epochs/10))
    lr_sched = lambda e: lr if (e < 200) else (lr*0.4 if e < 1000 else (lr*0.1))   
    return IntegratedDynamicsTrainer(
        model,dataloaders,opt_constr,lr_sched,max_grad_norm,**trainer_config
    )

if __name__ == "__main__":
    Trial = hnnScalars_trial(makeTrainerScalars)
    i = makeTrainerScalars.__kwdefaults__['trial']
    cfg,outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__), i)
    print(outcome)

