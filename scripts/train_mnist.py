#!/usr/bin/env python3

try:
    import intro
except ModuleNotFoundError:
    import os,sys
    sys.path.insert(0,os.path.join(os.environ['VIRTUAL_ENV'],'src','intro_dev'))
    import intro
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import data as datahandling
import lightning as lp

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('stage',default=0,type=int,help='Select teaching stage to run')
    p.add_argument('--batch',default=200,type=int,help="Specify the batch size to use (def: %(default)s)")
    return p.parse_args()



################################################
## Does it train?
################################################
def stage0(batchsize=200):
    dataset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor())
    dataloader = DataLoader(dataset,num_workers=7,batch_size=batchsize)

    autoencoder = intro.models.AutoEncoder_v0()

    trainer = lp.Trainer(max_epochs=10,accelerator='auto')
    trainer.fit(autoencoder,dataloader)

    trainer.test(autoencoder,dataloader)


################################################
## Was the result above biased?
################################################
def stage1(batchsize=200):
    dataset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor(), train=True)
    dataloader = DataLoader(dataset,num_workers=7,batch_size=batchsize)

    autoencoder = intro.models.AutoEncoder_v0()

    trainer = lp.Trainer(max_epochs=10,accelerator='auto')
    trainer.fit(autoencoder,dataloader)

    testset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor(), train=False)
    testloader = DataLoader(testset,num_workers=7,batch_size=batchsize)

    trainer.test(autoencoder,testloader)

################################################
## Can we better understand data bias?
################################################
def stage2(batchsize=200):
    dataset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor())

    train_data, valid_data = datahandling.random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    dataloader = DataLoader(train_data,num_workers=7,batch_size=batchsize)
    validloader = DataLoader(valid_data,num_workers=7,batch_size=batchsize)
    autoencoder = intro.models.AutoEncoder_v0()

    trainer = lp.Trainer(max_epochs=10,accelerator='auto')
    trainer.fit(autoencoder,dataloader,validloader)

    testset = MNIST(intro.get_dataset_dir(),transform=transforms.ToTensor(), train=False)
    testloader = DataLoader(testset,num_workers=7,batch_size=batchsize)

    trainer.test(autoencoder,testloader)

def stage3(batchsize=200):
    pass

def main():
    args = parse_args()
    if args.stage == 0:
        stage0(args.batch)
    elif args.stage == 1:
        stage1(args.batch)
    elif args.stage == 2:
        stage2(args.batch)
    elif args.stage == 3:
        stage3(args.batch)

if __name__ == '__main__':
    main()
