#!/usr/bin/env python3

try:
    import intro
except ModuleNotFoundError:
    import os,sys
    sys.path.insert(0,os.path.join(os.environ['VIRTUAL_ENV'],'src','intro_dev'))
    sys.path.insert(0,os.path.join(os.environ['VIRTUAL_ENV'],'src','intro_ml'))
    import intro
from torchvision.datasets import QMNIST as MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import data as datahandling
import lightning as lp

import argparse

from intro.loggers import LoggerLearning as LL

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('stage',default=0,type=int,help='Select teaching stage to run')
    p.add_argument('--batch',default=200,type=int,help="Specify the batch size to use (def: %(default)s)")
    p.add_argument('--save',action='store_true',help='Save off the final model as a checkpoint')
    return p.parse_args()



################################################
## Does it train?
################################################
def stage0(batchsize:int=200,pathfile:str=None):
    dataset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor())
    dataloader = DataLoader(dataset,num_workers=7,batch_size=batchsize)

    autoencoder = intro.models.AutoEncoder_v0()

    logg = LL('ae_qmnist',"stage_0")
    ## disable auto checkpointing for learning purposes
    trainer = lp.Trainer(max_epochs=10,accelerator='auto',enable_checkpointing=False,logger=logg)
    trainer.fit(autoencoder,dataloader)
    if pathfile is not None:
        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir,'model.ckpt'))

    trainer.test(autoencoder,dataloader)


################################################
## Was the result above biased?
################################################
def stage1(batchsize:int=200,pathfile:str=None):
    dataset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor(), train=True)
    dataloader = DataLoader(dataset,num_workers=7,batch_size=batchsize)

    autoencoder = intro.models.AutoEncoder_v0()

    logg = LL('ae_qmnist',"stage_1")
    ## disable auto checkpointing for learning purposes
    trainer = lp.Trainer(max_epochs=10,accelerator='auto',logger=logg,enable_checkpointing=False)
    trainer.fit(autoencoder,dataloader)
    if pathfile is not None:
        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir,'model.ckpt'))

    testset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor(), train=False)
    testloader = DataLoader(testset,num_workers=7,batch_size=batchsize)

    trainer.test(autoencoder,testloader)

################################################
## Can we better understand data bias?
################################################
def stage2(batchsize:int=200,pathfile:str=None):
    dataset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor())

    train_data, valid_data = datahandling.random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    dataloader = DataLoader(train_data,num_workers=7,batch_size=batchsize)
    validloader = DataLoader(valid_data,num_workers=7,batch_size=batchsize)
    autoencoder = intro.models.AutoEncoder_v0()

    logg = LL('ae_qmnist',"stage_2")
    ## disable auto checkpointing for learning purposes
    trainer = lp.Trainer(max_epochs=10,accelerator='auto',logger=logg,enable_checkpointing=False)
    trainer.fit(autoencoder,dataloader,validloader)
    if pathfile is not None:
        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir,'model.ckpt'))

    testset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor(), train=False)
    testloader = DataLoader(testset,num_workers=7,batch_size=batchsize)

    trainer.test(autoencoder,testloader)

################################################
## Can we automate these bias search?
################################################
def stage3(batchsize:int=200,pathfile:str=None):
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

    dataset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor())

    train_data, valid_data = datahandling.random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    dataloader = DataLoader(train_data,num_workers=7,batch_size=batchsize)
    validloader = DataLoader(valid_data,num_workers=7,batch_size=batchsize)
    autoencoder = intro.models.AutoEncoder_v0()

    enable_ckpt = True
    early_stopper = EarlyStopping(monitor='val_loss',mode='min',patience=3,min_delta=0.001)
    checkpointer = ModelCheckpoint(monitor='val_loss',mode='min')
    callbacks = [checkpointer,early_stopper]
    logg = LL('ae_qmnist',"stage_3")
    trainer = lp.Trainer(max_epochs=30,accelerator='auto',callbacks=callbacks,enable_checkpointing=enable_ckpt,logger=logg)
    trainer.fit(autoencoder,dataloader,validloader)
    autoencoder = intro.models.AutoEncoder_v0.load_from_checkpoint(checkpointer.best_model_path)
    if pathfile is not None:
        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir,'model.ckpt'))

    testset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor(), train=False)
    testloader = DataLoader(testset,num_workers=7,batch_size=batchsize)

    trainer.test(autoencoder,testloader)


def main():
    args = parse_args()
    if args.stage == 0:
        stage0(args.batch,args.save)
    elif args.stage == 1:
        stage1(args.batch,args.save)
    elif args.stage == 2:
        stage2(args.batch,args.save)
    elif args.stage == 3:
        stage3(args.batch,args.save)
        

if __name__ == '__main__':
    main()
