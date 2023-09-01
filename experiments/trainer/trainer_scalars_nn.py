import pytorch_lightning as pl
import torch
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import torch.utils.data as data
from scalars_nn import BasicMLP, EquivariancePermutationLayer, EquivarianceLayer
from pytorch_lightning.callbacks import ModelCheckpoint
#################################################################################
class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.

    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    def __init__(self, dataset: data.Dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(range(len(self.dataset)))

        return iter((torch.randperm(self.target_length) % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length


def comp_R2(y,yhat):
    ybar = torch.mean(y,dim=0,keepdim=True)
    den = y-ybar
    nom = yhat-ybar 
    return torch.sum(nom*nom)/torch.sum(den*den)

class InvarianceNet(pl.LightningModule):
    def __init__(
        self, 
        n_in_net=3,
        n_hidden_mlp=100, 
        n_layers_mlp=2,
        layer_norm_mlp=True, 
        learning_rate=1e-3,
        milestones=[30,80,120],
        gamma=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = BasicMLP( 
            n_in=n_in_net, 
            n_out=1,
            n_hidden=n_hidden_mlp, 
            n_layers=n_layers_mlp,
            layer_norm=layer_norm_mlp
        )
        self.learning_rate = learning_rate
        self.milestones=milestones
        self.gamma=gamma
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
      
    def forward(self, x):
        
        yhat = self.model(x) # [_, n_out]
        
        
        return yhat

    def custom_weights_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        scalars, X, Y = batch
        # Train the model
        Yhat = self(x=scalars)
        
        # compute the train loss    
        train_MSE = F.mse_loss(Yhat, Y)
        # train_R2 = comp_R2(Y,Yhat)
        
        self.training_step_outputs.append(train_MSE)
        # Logging to TensorBoard by default
        self.log('loss', train_MSE) 
        return {'loss':  train_MSE}         
    
    def on_train_epoch_end(self): 
        avg_MSE = torch.stack(self.training_step_outputs).mean() 
        # logging histograms
        self.custom_weights_histogram_adder()
        
        # logging using tensorboard logger
        self.logger.experiment.add_scalars("Train epoch", 
                                           {'loss'   :  avg_MSE}, 
                                           self.current_epoch) 

        self.training_step_outputs.clear()  # free memory
                
    def configure_optimizers(self):
        """
        # Note that for ReduceLROnPlateau, step should be called after validate():
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     scheduler.step(val_loss)
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate) 
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 


    def validation_step(self, batch, batch_idx):
        """
        model.eval() and torch.no_grad() are called automatically for validation
        """
        scalars, X, Y = batch
        # Train the model
        Yhat = self(x=scalars) 
        # compute the loss
        val_MSE = F.mse_loss(Yhat, Y)
        val_R2 = comp_R2(Y,Yhat)

        self.validation_step_outputs.append([val_MSE, val_R2])
        # Logging to TensorBoard by default
        self.log('val_MSE', val_MSE, prog_bar=True)
        self.log('val_R2',   val_R2,   prog_bar=True)
        return {'MSE':  val_MSE, 'R2': val_R2}
    
    def on_validation_epoch_end(self):
        # calculating average loss 
        avg_MSE = torch.stack([x[0] for x in self.validation_step_outputs]).mean()
        avg_R2  = torch.stack([x[1] for x in self.validation_step_outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalars("Validation epoch", 
                                           {'MSE'   :  avg_MSE,
                                            'R2'     : avg_R2}, 
                                           self.current_epoch) 
        self.validation_step_outputs.clear()  # free memory

    
    def test_step(self, batch, batch_idx):
        scalars, X, Y = batch
        # Train the model
        Yhat = self(x=scalars)
        
        # compute the loss
        test_MSE = F.mse_loss(Yhat, Y)
        test_R2 = comp_R2(Y,Yhat)

        self.test_step_outputs.append([test_MSE, test_R2])
        return  {'MSE':  test_MSE, 'R2': test_R2}

    def on_test_epoch_end(self):
        avg_MSE = torch.stack([x[0] for x in self.test_step_outputs]).mean()
        avg_R2  = torch.stack([x[1] for x in self.test_step_outputs]).mean()
        metrics = {'MSE': avg_MSE, 'R2': avg_R2}
        self.log_dict(metrics)
        self.test_step_outputs.clear()  # free memory


 
class EquivarianceNet(pl.LightningModule):
    def __init__(
        self, 
        n_in_net=3,
        n_hidden_mlp=100, 
        n_layers_mlp=2,
        layer_norm_mlp=True, 
        learning_rate=1e-3,
        milestones=[30,80,120],
        gamma=0.5,
        permutation=True
    ):
        super().__init__()
        self.save_hyperparameters()
        if permutation == True:
            self.model = nn.Sequential(
                EquivariancePermutationLayer(
                    n_in=n_in_net, 
                    n_hidden=n_hidden_mlp, 
                    n_layers=n_layers_mlp, 
                    layer_norm=layer_norm_mlp
                ),
            )
        else: 
            self.model = nn.Sequential(
                EquivarianceLayer( 
                    n_in=20, 
                    n_hidden=n_hidden_mlp, 
                    n_layers=n_layers_mlp,
                    layer_norm=layer_norm_mlp
                )
            )
        
       
        self.learning_rate = learning_rate
        self.milestones=milestones
        self.gamma=gamma

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        
        yhat = self.model(x) # [_, n_out]
        return yhat

    def custom_weights_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        scalars, X, Y = batch
        # Train the model
        Yhat = self(x=(scalars, X))


        # compute the train loss    
        train_MSE = F.mse_loss(Yhat, Y) 
         
        self.training_step_outputs.append(train_MSE)
        # Logging to TensorBoard by default
        self.log('loss', train_MSE) 
        return {'loss':  train_MSE} 

    def on_train_epoch_end(self): 
        avg_MSE = torch.stack(self.training_step_outputs).mean() 
        # logging histograms
        self.custom_weights_histogram_adder()
        
        # logging using tensorboard logger
        self.logger.experiment.add_scalars("Train epoch", 
                                           {'loss'   :  avg_MSE}, 
                                           self.current_epoch) 

        self.training_step_outputs.clear()  # free memory
            
    def configure_optimizers(self):
        """
        # Note that for ReduceLROnPlateau, step should be called after validate():
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     scheduler.step(val_loss)
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 

    def validation_step(self, batch, batch_idx):
        """
        model.eval() and torch.no_grad() are called automatically for validation
        """
        scalars, X, Y = batch
        # Train the model
        Yhat = self(x=(scalars, X)) 
        # compute the loss
        val_MSE = F.mse_loss(Yhat, Y)
        val_R2 = comp_R2(Y,Yhat)
        
        # Logging to TensorBoard by default
        self.log('val_MSE', val_MSE, prog_bar=True)
        self.log('val_R2',   val_R2,   prog_bar=True)
         
        self.validation_step_outputs.append([val_MSE, val_R2])
        return {'MSE': val_MSE, 'R2': val_R2}

    def on_validation_epoch_end(self):
        # calculating average loss 
        avg_MSE = torch.stack([x[0] for x in self.validation_step_outputs]).mean()
        avg_R2  = torch.stack([x[1] for x in self.validation_step_outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalars("Validation epoch", 
                                           {'MSE'   :  avg_MSE,
                                            'R2'     : avg_R2}, 
                                           self.current_epoch) 
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        scalars, X, Y = batch
        # Train the model
        Yhat = self(x=(scalars, X))
        # compute the loss
        test_MSE = F.mse_loss(Yhat, Y)
        test_R2 = comp_R2(Y,Yhat)

        self.test_step_outputs.append([test_MSE, test_R2])
        return  {'MSE':  test_MSE, 'R2': test_R2}

    def on_test_epoch_end(self):
        avg_MSE = torch.stack([x[0] for x in self.test_step_outputs]).mean()
        avg_R2  = torch.stack([x[1] for x in self.test_step_outputs]).mean()
        metrics = {'MSE': avg_MSE, 'R2': avg_R2}
        self.log_dict(metrics)
        self.test_step_outputs.clear()  # free memory

        
def train_pl_model(
    n_in_net, 
    n_hidden_mlp=100, 
    n_layers_mlp=2, 
    layer_norm_mlp=False,
    train_loader: DataLoader=None,
    test_loader: DataLoader=None,
    validation_loader: DataLoader=None,
    symname="O3equivariant",
    num_gpus: int=1,
    max_epochs=100,
    min_epochs=0,
    learning_rate=0.001,
    milestones=[30,80,120],
    gamma=0.1,
    check_val_every_n_epoch=1,
    path_logs="./logs/",
    permutation=False,
    progress_bar=True
):  

    if progress_bar:
        progress_bar_refresh_rate=None
    else:
        progress_bar_refresh_rate=0 
    if symname == "O3equivariant":
        litmodel = EquivarianceNet(
            n_in_net=n_in_net, 
            n_hidden_mlp=n_hidden_mlp, 
            n_layers_mlp=n_layers_mlp,
            layer_norm_mlp=layer_norm_mlp, 
            learning_rate=learning_rate,
            milestones=milestones,
            gamma=gamma,
            permutation=permutation
        )
    elif symname=="O5invariant":
        litmodel = InvarianceNet(
            n_in_net=n_in_net,
            n_hidden_mlp=n_hidden_mlp, 
            n_layers_mlp=n_layers_mlp,
            layer_norm_mlp=layer_norm_mlp, 
            learning_rate=learning_rate,
            milestones=milestones,
            gamma=gamma
        )
    elif symname == "Lorentz":
        litmodel = InvarianceNet(
            n_in_net=n_in_net,
            n_hidden_mlp=n_hidden_mlp, 
            n_layers_mlp=n_layers_mlp,
            layer_norm_mlp=layer_norm_mlp, 
            learning_rate=learning_rate,
            milestones=milestones,
            gamma=gamma
        )
    else:
        raise ValueError("wrong symname???")
    

    # Save the top five check points that give the smallest values of validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_MSE',
        dirpath=path_logs,
        filename=symname+'-{epoch:02d}-{val_MSE:.6f}',
        save_top_k=3,
        mode='min',
    )
 
    kwargs = {
        'max_epochs': max_epochs, 
        'min_epochs': min_epochs,
        'accelerator':'gpu', 
        'devices': num_gpus,
        'check_val_every_n_epoch': check_val_every_n_epoch,
        'default_root_dir': path_logs
    }
    
    trainer = pl.Trainer(**kwargs,callbacks=[checkpoint_callback])
    
    trainer.fit(
        model=litmodel, 
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader
    )
    test_metrics = trainer.test(dataloaders=test_loader)[0]
    return test_metrics
