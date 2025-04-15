
from torch import nn,optim
import lightning as lp

class Encoder_v0(nn.Module):
    def __init__(self,initial=28*28,steps=[64,3],activation=nn.ReLU()):
        assert(len(steps) > 0)
        super().__init__()
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
