from emlp.nn import EquivarianceLayer_objax,compute_scalars,radial_basis_transform
from trainer.hamiltonian_dynamics import IntegratedODETrainer,DoubleSpringPendulum,odeScalars_trial
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo 
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import logging
import emlp.nn
import emlp.groups
import objax
 


levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                    'warn': logging.WARNING,'warning': logging.WARNING,
                    'info': logging.INFO,'debug': logging.DEBUG}

def makeTrainerScalars(*,dataset=DoubleSpringPendulum,num_epochs=2000,ndata=5000,seed=2021,aug=False,
                n_rad=200,bs=500,lr=5e-3,device='cuda',split={'train':500,'val':.1,'test':.1},
                data_config={'chunk_len':5,'dt':0.2,'integration_time':30,'regen':False},
                net_config={'n_layers':3,'n_hidden':100},log_level='warn',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.75},}, 
                save=True,trial=1):

    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(n_systems=ndata,**data_config)
        datasets = split_dataset(base_ds,splits=split)
    
    z0_train = base_ds.Zs[datasets['train']._ids,0,:]
    zps_train = base_ds.ZPs[datasets['train']._ids]
    scalars_z0 = compute_scalars(z0_train.reshape(-1,4,3), zps_train)
    trans_mu, trans_gamma = radial_basis_transform(scalars_z0, nrad = n_rad) 
    model = EquivarianceLayer_objax(
        n_layers=net_config['n_layers'], 
        n_hidden=net_config['n_hidden'],
        mu=trans_mu,
        gamma=trans_gamma
    )
    
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                   num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
     
    opt_constr = objax.optimizer.Adam
    # lr_sched = lambda e: lr#*cosLr(num_epochs)(e)#*min(1,e/(num_epochs/10))
    lr_sched = lambda e: lr if e < 300 else (lr*0.5 if e < 1200 else lr*0.2)
    return IntegratedODETrainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = odeScalars_trial(makeTrainerScalars)
    cfg,outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))
    print(outcome)
