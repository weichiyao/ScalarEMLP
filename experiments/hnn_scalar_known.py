from emlp.nn import InvarianceKnown
from trainer.hamiltonian_dynamics import KnownDynamicsTrainer,hnnScalarsKnown_trial
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr,FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import torch.nn as nn
import logging
import emlp
import emlp.reps
import objax

levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
          'warn': logging.WARNING,'warning': logging.WARNING,
          'info': logging.INFO,'debug': logging.DEBUG}


def makeTrainerScalars(*,num_epochs=100,seed=2022, 
                       data_config={'datasource':'/home/data.pickle',
                                    'itest':[1], 'chunk_len':5, 'ntrain_out':None},
                       bs=16,lr=1e-3,device='cuda',  
                       net_config={'n_layers':3,'n_hidden':10}, log_level='info',
                       trainer_config={'log_dir':'/home/','log_args':{'minPeriod':.02,'timeFrac':.75},},
                       save=False,):
    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        knowndata = KnownData(**data_config)
    
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                   num_workers=0,pin_memory=False)) for k,v in knowndata().items()}
    dataloaders['Train'] = dataloaders['train']
    
    model = InvarianceKnown(**net_config)
     
    opt_constr = objax.optimizer.Adam
    # lr_sched = lambda e: lr#*cosLr(num_epochs)(e)#*min(1,e/(num_epochs/10))
    lr_sched = lambda e: lr if (e < 200) else (lr*0.4 if e < 1000 else (lr*0.1))   
    return KnownDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = hnnScalarsKnown_trial(makeTrainerScalars)
    cfg,outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__,namespace=(emlp.groups,emlp.nn)))
    print(outcome)
