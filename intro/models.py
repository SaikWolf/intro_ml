
import torch
from torch import nn,optim
import lightning as lp
import numpy as np

class Encoder_v0(nn.Module):
    def __init__(self,initial=28*28,steps=[64,3],activation=nn.ReLU()):
        assert(len(steps) > 0)
        super().__init__()
        if isinstance(activation,str) and hasattr(nn,activation):
            activation = getattr(nn,activation)()
        self.hyperparams = {"initial":initial,"steps":steps,"activation":activation.__class__.__name__}
        self.linear_layers = nn.ModuleList([
            nn.Linear(initial,steps[0])
        ])
        self.linear_layers.extend([
            nn.Linear(steps[idx],steps[idx+1]) for idx in range(len(steps)-1)
        ])
        self._steps = len(steps)
        self._looped = self._steps-1
        self.activation = activation
        self.out_size = steps[-1]
        self.in_size = initial

    def forward(self, x):
        for idx in range(self._looped):
            x = self.activation(self.linear_layers[idx](x))
        return self.linear_layers[-1](x)


class Decoder_v0(nn.Module):
    def __init__(self,initial=3,steps=[64,28*28],activation=nn.ReLU()):
        assert(len(steps) > 0)
        super().__init__()
        if isinstance(activation,str) and hasattr(nn,activation):
            activation = getattr(nn,activation)()
        self.hyperparams = {"initial":initial,"steps":steps,"activation":activation.__class__.__name__}
        self.linear_layers = nn.ModuleList([
            nn.Linear(initial,steps[0])
        ])
        self.linear_layers.extend([
            nn.Linear(steps[idx],steps[idx+1]) for idx in range(len(steps)-1)
        ])
        self._steps = len(steps)
        self._looped = self._steps-1
        self.activation = activation
        self.out_size = steps[-1]
        self.in_size = initial

    def forward(self, x):
        for idx in range(self._looped):
            x = self.activation(self.linear_layers[idx](x))
        return self.linear_layers[-1](x)


class AutoEncoder_v0(lp.LightningModule):
    def __init__(self,
                 encoder=Encoder_v0(),
                 decoder=Decoder_v0(),
                 loss_fcn=nn.functional.mse_loss,
                 optim_fcn=lambda params: optim.Adam(params,lr=1e-3)
                ):
        super().__init__()
        if isinstance(encoder,dict):
            encoder = Encoder_v0(**encoder)
        else:
            self.save_hyperparameters({"encoder":encoder.hyperparams})
        if isinstance(decoder,dict):
            decoder = Decoder_v0(**decoder)
        else:
            self.save_hyperparameters({"decoder":decoder.hyperparams})
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = loss_fcn
        self.optim_maker = optim_fcn

    def training_step(self, batch, batch_idx):
        observation, label_or_target = batch
        del label_or_target ### not used in autoencoder

        # size(0) -> batch size, so flatten observation in a vector
        x = observation.view(observation.size(0),-1)
        y = self.encoder(x)
        x_hat = self.decoder(y)
        loss = self.criterion(x_hat,x)
        self.log('train_loss',loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        observation, label_or_target = batch
        del label_or_target ### not used in autoencoder

        # size(0) -> batch size, so flatten observation in a vector
        x = observation.view(observation.size(0),-1)
        y = self.encoder(x)
        x_hat = self.decoder(y)
        loss = self.criterion(x_hat,x)
        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        observation, label_or_target = batch
        del label_or_target ### not used in autoencoder

        # size(0) -> batch size, so flatten observation in a vector
        x = observation.view(observation.size(0),-1)
        y = self.encoder(x)
        x_hat = self.decoder(y)
        loss = self.criterion(x_hat,x)
        self.log('test_loss', loss)

    
    def configure_optimizers(self):
        optimizer = self.optim_maker(self.parameters())
        return optimizer



class Backbone_v0(nn.Module):
    def __init__(self,initial=(28,28),steps=[3,13,3],activation=nn.ReLU()):
        # initial shape (H,W)->C=1 or (C,H,W)
        assert(len(steps)>0)
        super().__init__()
        if isinstance(activation,str) and hasattr(nn,activation):
            activation = getattr(nn,activation)()
        self.hyperparams = {"initial":initial,"steps":steps,"activation":activation.__class__.__name__}
        self.in_size = initial if len(initial) > 2 else (1,)+tuple(initial)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=self.in_size[0],
                      out_channels=2*self.in_size[0],
                      kernel_size=(steps[0],steps[0]),
                      stride=1,
                      bias=True)
        ])
        self.conv_layers.extend([
            nn.Conv2d(in_channels=self.in_size[0]*2**(idx),
                      out_channels=self.in_size[0]*2**(idx+1),
                      kernel_size=(steps[idx],steps[idx]),
                      stride=1,
                      bias=True) for idx in range(1,len(steps))
        ])
        self._steps = len(steps)
        self._looped = self._steps-1
        self.activation = activation
        self.out_size = (
            self.in_size[0]*2**(self._steps),
            self.in_size[1] - sum([x-1 for x in steps]),
            self.in_size[2] - sum([x-1 for x in steps]))

    def forward(self, x):
        for idx in range(self._looped):
            x = self.activation(self.conv_layers[idx](x))
        return self.conv_layers[-1](x)


class DecisionHead_v0(nn.Module):
    def __init__(self,initial=(8,12,12),steps=[256,10],activation=nn.ReLU()):
        # initial shape (H,W)->C=1 or (C,H,W)
        assert(len(steps)>0)
        super().__init__()
        if isinstance(activation,str) and hasattr(nn,activation):
            activation = getattr(nn,activation)()
        self.hyperparams = {"initial":initial,"steps":steps,"activation":activation.__class__.__name__}
        self.in_size = initial if len(initial) > 2 else (1,)+tuple(initial)
        self.linear_layers = nn.ModuleList([
            nn.Linear(np.prod(self.in_size),steps[0],bias=True)
        ])
        self.linear_layers.extend([
            nn.Linear(steps[idx],steps[idx+1],bias=True) for idx in range(len(steps)-1)
        ])
        self._steps = len(steps)
        self._looped = self._steps-1
        self.activation = activation
        self.out_size = steps[-1]

    def forward(self, x):
        for idx in range(self._looped):
            x = self.activation(self.linear_layers[idx](x))
        return self.linear_layers[-1](x)


class Classifier_v0(lp.LightningModule):
    def __init__(self,
                 backbone=Backbone_v0(),
                 head=DecisionHead_v0(),
                 loss_fcn=nn.functional.cross_entropy,
                 optim_fcn=lambda params: optim.Adam(params,lr=1e-3)
                ):
        super().__init__()
        if isinstance(backbone,dict):
            backbone = Backbone_v0(**backbone)
        else:
            self.save_hyperparameters({"backbone":backbone.hyperparams})
        if isinstance(head,dict):
            head = DecisionHead_v0(**head)
        else:
            self.save_hyperparameters({"head":head.hyperparams})
        self.backbone = backbone
        self.head = head
        self.criterion = loss_fcn
        self.optim_maker = optim_fcn

    def training_step(self, batch, batch_idx):
        observation, label_or_target = batch

        # size(0) -> batch size, so flatten observation in a vector
        x = self.backbone(observation)
        y = x.view(observation.size(0),-1)
        z = self.head(y)
        loss = self.criterion(z,label_or_target)
        l_hat = torch.argmax(z,1)
        acc = torch.eq(l_hat,label_or_target).float().mean()
        self.log_dict({'train_loss':loss,"train_acc":acc},prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        observation, label_or_target = batch

        # size(0) -> batch size, so flatten observation in a vector
        x = self.backbone(observation)
        y = x.view(observation.size(0),-1)
        z = self.head(y)
        loss = self.criterion(z,label_or_target)
        l_hat = torch.argmax(z,1)
        acc = torch.eq(l_hat,label_or_target).float().mean()
        self.log_dict({'val_loss':loss,"val_acc":acc},prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        observation, label_or_target = batch

        # size(0) -> batch size, so flatten observation in a vector
        x = self.backbone(observation)
        y = x.view(observation.size(0),-1)
        z = self.head(y)
        loss = self.criterion(z,label_or_target)
        l_hat = torch.argmax(z,1)
        acc = torch.eq(l_hat,label_or_target).float().mean()
        self.log_dict({'test_loss':loss,"test_acc":acc})


    def configure_optimizers(self):
        optimizer = self.optim_maker(self.parameters())
        return optimizer

