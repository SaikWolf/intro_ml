#!/usr/bin/env python3

try:
    import intro
except ModuleNotFoundError:
    import os,sys
    sys.path.insert(0,os.path.join(os.environ['VIRTUAL_ENV'],'src','intro_dev'))
    import intro
from torchvision.datasets import FashionMNIST, KMNIST, MNIST, QMNIST

def main():
    datasets = intro.get_dataset_dir()

    FashionMNIST(datasets,download=True,train=True)
    FashionMNIST(datasets,download=True,train=False)
    KMNIST(datasets,download=True,train=True)
    KMNIST(datasets,download=True,train=False)
    MNIST(datasets,download=True,train=True)
    MNIST(datasets,download=True,train=False)
    QMNIST(datasets,download=True,train=True)
    QMNIST(datasets,download=True,train=False)

if __name__ == '__main__':
    main()
