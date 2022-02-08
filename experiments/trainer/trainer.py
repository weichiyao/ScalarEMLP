import dill
from oil.logging.lazyLogger import LazyLogger
from oil.utils.utils import Eval, Named
from oil.utils.mytqdm import tqdm
from oil.tuning.study import guess_metric_sign
import copy, os, random
import glob
import numpy as np
from natsort import natsorted
import jax
from jax import lax
from jax import numpy as jnp
from jax.example_libraries import optimizers as jeoptim
from jax.tree_util import tree_map

import logging
from functools import partial
import objax 

class Trainer(object,metaclass=Named):
    """ Base trainer
        """
    def __init__(self, model, dataloaders, optim = objax.optimizer.Adam, lr_sched =lambda e:1, 
                 max_grad_norm=float('inf'), log_dir=None, log_suffix='',log_args={},early_stop_metric=None):
        # Setup model, optimizer, and dataloaders
        self.model = model#
        #self.model= objax.Jit(objax.ForceArgs(model,training=True)) #TODO: figure out static nums
        #self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
        #self.model.predict = objax.ForceArgs(model.__call__,training=False)
        #self._model = model
        #self.model = objax.ForceArgs(model,training=True)
        #self.model.predict = objax.ForceArgs(model.__call__,training=False)
        #self.model = objax.Jit(lambda x, training: model(x,training=training),model.vars(),static_argnums=(1,))
        #self.model = objax.Jit(model,static_argnums=(1,))
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim(model.vars())
        self.lr_sched= lr_sched
        self.dataloaders = dataloaders # A dictionary of dataloaders
        self.epoch = 0

        self.logger = LazyLogger(log_dir, log_suffix, **log_args)
        #self.logger.add_text('ModelSpec','model: {}'.format(model))
        self.hypers = {}
        self.ckpt = None# copy.deepcopy(self.state_dict()) #TODO fix model saving
        self.early_stop_metric = early_stop_metric
        #fastloss = objax.Jit(self.loss,model.vars())
        
        self.gradvals = objax.GradValues(self.loss,self.model.vars())
        #self.gradvals = objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
    def metrics(self,loader):
        return {}

    def loss(self,minibatch):
        raise NotImplementedError

    def train_to(self, final_epoch=100):
        assert final_epoch>=self.epoch, "trying to train less than already trained"
        self.train(final_epoch-self.epoch)

    def train(self, num_epochs=100):
        """ The main training loop"""
        start_epoch = self.epoch
        steps_per_epoch = len(self.dataloaders['train']); step=0
        for self.epoch in tqdm(range(start_epoch, start_epoch + num_epochs),desc='train'):
            for i, minibatch in enumerate(self.dataloaders['train']):
                step = i + self.epoch*steps_per_epoch
                loss_val = self.step(self.epoch+i/steps_per_epoch,minibatch) 
                # print(f"\n epoch {self.epoch} step {step}, loss = {round(loss_val.tolist(),3)}\n")
                with self.logger as do_log:
                    if do_log: self.logStuff(step, minibatch)
        self.epoch+=1
        self.logStuff(step)

    def step(self, epoch, minibatch):
        grad_val, loss_val = self.gradvals(minibatch)
        if jnp.isnan(grad_val[1]).any(): 
            return loss_val[0]
        # Clip the gradients according to global max norm.
        grad_val = jeoptim.clip_grads(grad_val, self.max_grad_norm)
        self.optimizer(self.lr_sched(epoch), grad_val)
        return loss_val[0] 

    def logStuff(self, step, minibatch=None):
        metrics = {}
        # if minibatch is not None and hasattr(self,'loss'):
        #     try: metrics['Minibatch_Loss'] = self.loss(self.model_params,minibatch)
        #     except (NotImplementedError, TypeError): pass
        for loader_name,dloader in self.dataloaders.items(): # Ignore metrics on train
            if loader_name=='train' or len(dloader)==0 or loader_name[0]=='_': continue
            for metric_name, metric_value in self.metrics(dloader).items():
                metrics[loader_name+'_'+metric_name] = metric_value
        self.logger.add_scalars('metrics', metrics, step)
        # for name,m in self.model.named_modules():
        #     if hasattr(m, 'log_data'):
        #         m.log_data(self.logger,step,name)
        self.logger.report()
        # update the best checkpoint
        if self.early_stop_metric is not None:
            maximize = guess_metric_sign(self.early_stop_metric)
            sign = 2*maximize-1
            best = (sign*self.logger.scalar_frame[self.early_stop_metric].values).max()
            current = sign*self.logger.scalar_frame[self.early_stop_metric].iloc[-1]
            if current >= best: self.ckpt = copy.deepcopy(self.state_dict())
        else: self.ckpt = copy.deepcopy(self.state_dict())

    def evalAverageMetrics(self,loader,metrics):
        num_total, loss_totals = 0, 0
        for minibatch in loader:
            try: mb_size = loader.batch_size
            except AttributeError: mb_size=1
            loss_totals += mb_size*metrics(minibatch)
            num_total += mb_size
        if num_total==0: raise KeyError("dataloader is empty")
        return loss_totals/num_total

    def state_dict(self):
        #TODO: handle saving and loading state
        state = {
            'outcome':self.logger.scalar_frame[-1:],
            'epoch':self.epoch,
            # 'model_state':self.model.state_dict(),
            # 'optim_state':self.optimizer.state_dict(),
            # 'logger_state':self.logger.state_dict(),
        }
        return state

    # def load_state_dict(self,state):
    #     self.epoch = state['epoch']
    #     self.model.load_state_dict(state['model_state'])
    #     self.optimizer.load_state_dict(state['optim_state'])
    #     self.logger.load_state_dict(state['logger_state'])

    def load_checkpoint(self,path=None):
        """ Loads the checkpoint from path, if None gets the highest epoch checkpoint"""
        if not path:
            chkpts = glob.glob(os.path.join(self.logger.log_dirr,'checkpoints/c*.state'))
            path = natsorted(chkpts)[-1] # get most recent checkpoint
            print(f"loading checkpoint {path}")
        with open(path,'rb') as f:
            self.load_state_dict(dill.load(f))

    def save_checkpoint(self):
        return self.logger.save_object(self.ckpt,suffix=f'checkpoints/c{self.epoch}.state')

# def safe_clip_grads(grad_tree, max_norm):
#     """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
#     norm = jeoptim.l2_norm(grad_tree)
#     eps = 1e-7
#     normalize = lambda g: jnp.where(norm < max_norm, g, g * max_norm / (norm + eps))
#     return tree_map(normalize, grad_tree)
# jax.config.update('jax_debug_nans', True)
# safe_clip_grads(jnp.array([0]), 1)
