from emlp.nn import InvarianceLayerGeneral, ScalarTransformerGeneral
from trainer.hamiltonian_dynamics import GeneralDynamicsTrainer,GeneralData,TrialHNN
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr,FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.tuning.args import argupdated_config 
import logging
import emlp
import emlp.reps
import objax
import numpy as np


levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
          'warn': logging.WARNING,'warning': logging.WARNING,
          'info': logging.INFO,'debug': logging.DEBUG}

def makeTrainer(data_config={'datasource':'/home/data.pickle',
                             'itest':[1], 'chunk_len':5, 'ntrain_out':None},
                seed=2022, 
                bs=16,lr=1e-3,max_grad_norm=0.5,device='cuda',
                transformer_config={
                    'method':'none', 'dimensionless':False, 'n_rad':50, 
                    'n_quantiles':200, 'transform_distribution':'uniform'
                },
                net_config={'n_layers':3,'n_hidden':10}, 
                trainer_config={'log_dir':'/home/',
                                'log_args':{'minPeriod':.02,'timeFrac':.75},},
                save=False):   
  
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        gdata = GeneralData(**data_config)
    
    # Create the transformer 
    stransformer = ScalarTransformerDP(zs=gdata.Zs, pv=gdata.PV, ps=gdata.PS, **transformer_config)
    
    # Create data loaders
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                   num_workers=0,pin_memory=False)) for k,v in gdata().items()}
    dataloaders['Train'] = dataloaders['train']
    
    # Trainer
    model = InvarianceGeneral(**net_config)
    opt_constr = objax.optimizer.Adam 
    lr_sched = lambda e: lr  
    
    return GeneralDynamicsTrainer(
        model,dataloaders,opt_constr,lr_sched,max_grad_norm,**trainer_config
    )
  
if __name__ == "__main__":
    trial_hnn = TrialHNN(makeTrainer)
    cfg, outcome = trial_hnn(argupdated_config(makeTrainer.__kwdefaults__))
    print(outcome)

